"""TurboQuant benchmark: compression ratio vs recall vs speed.

Run:  python benchmarks/bench.py [--n 50000] [--dim 128] [--queries 200]

Measures every method against exact float32 ground truth on synthetic
clustered embeddings and prints the markdown tables in the README:

  1. L2 nearest-neighbor search -- the main memory/recall/latency table.
  2. Maximum inner-product search -- where AnisotropicPQ earns its place.
  3. Filtered search -- what probe escalation is worth under a selective
     filter, measured against exact *filtered* ground truth.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from turboquant import (  # noqa: E402
    AdaptiveBitQuantizer,
    AnisotropicPQ,
    FlatIndex,
    IVFPQIndex,
    OPQProductQuantizer,
    ProductQuantizer,
    QuantizedFlatIndex,
    ScalarQuantizer,
    TurboIndex,
    recall_at_k,
    top_k_max,
    train_base_query_split,
)

K = 10


def timed_search(index, queries, k, **kwargs):
    t0 = time.perf_counter()
    ids, _ = index.search(queries, k, **kwargs)
    ms = (time.perf_counter() - t0) * 1000 / len(queries)
    return ids, ms


def table(header: list[str], rows: list[tuple]) -> None:
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--n-lists", type=int, default=128)
    ap.add_argument("--n-probe", type=int, default=16)
    args = ap.parse_args()

    n_train = min(args.n, 20_000)
    print(
        f"dataset: n={args.n} dim={args.dim} queries={args.queries} "
        "(synthetic clustered)"
    )
    train, base, queries = train_base_query_split(
        n_train, args.n, args.queries, args.dim, n_clusters=100, seed=0
    )
    raw_bytes = base.nbytes

    flat = FlatIndex(args.dim)
    flat.add(base)
    gt, flat_ms = timed_search(flat, queries, K)
    print(
        f"ground truth built (flat float32: {raw_bytes / 1e6:.1f} MB, "
        f"{flat_ms:.2f} ms/query)\n"
    )

    # ------------------------------------------------- 1. L2 search
    rows: list[tuple] = []

    def bench(name, index, mem_bytes, **search_kwargs):
        ids, ms = timed_search(index, queries, K, **search_kwargs)
        r = recall_at_k(ids, gt, K)
        rows.append(
            (
                name,
                f"{raw_bytes / mem_bytes:.1f}x",
                f"{mem_bytes / 1e6:.1f}",
                f"{r:.3f}",
                f"{ms:.2f}",
            )
        )

    rows.append(("Flat float32 (exact)", "1.0x", f"{raw_bytes / 1e6:.1f}", "1.000",
                 f"{flat_ms:.2f}"))

    for bits in (8, 4):
        idx = QuantizedFlatIndex(ScalarQuantizer(args.dim, bits=bits).train(train))
        idx.add(base)
        bench(f"Flat SQ{bits}", idx, idx.memory_bytes)

    idx = QuantizedFlatIndex(AdaptiveBitQuantizer(args.dim, avg_bits=4.0).train(train))
    idx.add(base)
    bench("Flat AdaptiveBits (avg 4b)", idx, idx.memory_bytes)

    # PQ and OPQ side by side: identical bytes/vector, learned rotation
    # is the only difference.
    for m in (args.dim // 16, args.dim // 8):
        idx = QuantizedFlatIndex(ProductQuantizer(args.dim, n_subspaces=m).train(train))
        idx.add(base)
        bench(f"Flat PQ M={m}", idx, idx.memory_bytes)
        idx = QuantizedFlatIndex(
            OPQProductQuantizer(args.dim, n_subspaces=m).train(train)
        )
        idx.add(base)
        bench(f"Flat OPQ M={m}", idx, idx.memory_bytes)

    for m in (args.dim // 16, args.dim // 8, args.dim // 4):
        ivf = IVFPQIndex(args.dim, n_lists=args.n_lists, n_subspaces=m).train(train)
        ivf.add(base)
        bench(
            f"IVFPQ M={m} nprobe={args.n_probe}",
            ivf,
            ivf.memory_bytes,
            n_probe=args.n_probe,
        )

    for budget in (16, 24, 32, 64):
        turbo = TurboIndex(args.dim, budget_bytes=budget, n_lists=args.n_lists)
        turbo.train(train)
        turbo.add(base)
        split_name = (
            f"PQ{turbo.ivfpq.pq.M}+PQ{turbo.refiner.M}"
            if turbo.refine_kind == "pq"
            else f"PQ{turbo.ivfpq.pq.M}+adaptive"
        )
        bench(
            f"TurboIndex {budget}B ({split_name})",
            turbo,
            turbo.memory_bytes,
            n_probe=args.n_probe,
            rerank_factor=8,
        )

    print("### L2 nearest-neighbor search\n")
    table(["Method", "Compression", "Memory (MB)", f"Recall@{K}", "ms/query"], rows)

    # ----------------------------------- 2. MIPS (inner product) search
    mips_gt, _ = top_k_max(queries @ base.T, K)
    mips_rows = []
    for m in (args.dim // 16, args.dim // 8):
        for label, quant in (
            ("PQ", ProductQuantizer(args.dim, n_subspaces=m)),
            ("AnisotropicPQ eta=8", AnisotropicPQ(args.dim, n_subspaces=m, eta=8.0)),
        ):
            quant.train(train)
            codes = quant.encode(base)
            t0 = time.perf_counter()
            scores = quant.adc_distances(quant.compute_ip_lut(queries), codes)
            ids, _ = top_k_max(scores, K)
            ms = (time.perf_counter() - t0) * 1000 / len(queries)
            mse = float(((base - quant.decode(codes)) ** 2).sum(axis=1).mean())
            mips_rows.append(
                (
                    f"{label} M={m}",
                    f"{m}",
                    f"{recall_at_k(ids, mips_gt, K):.3f}",
                    f"{mse:.2f}",
                    f"{ms:.2f}",
                )
            )
    print("### Maximum inner-product search (unnormalized vectors)\n")
    table(
        ["Method", "Bytes/vector", f"Recall@{K}", "Recon MSE", "ms/query"],
        mips_rows,
    )

    # ------------------------------------------- 3. Filtered search
    ivf = IVFPQIndex(args.dim, n_lists=args.n_lists, n_subspaces=args.dim // 8)
    ivf.train(train)
    ivf.add(base)
    filt_rows = []
    for keep_every in (10, 100, 1000):
        mask = np.zeros(args.n, dtype=bool)
        mask[::keep_every] = True
        eligible = np.flatnonzero(mask)
        sub = FlatIndex(args.dim)
        sub.add(base[eligible])
        local, _ = sub.search(queries, K)
        truth = eligible[local]

        for label, kwargs in (
            ("post-filter only", {}),
            ("+ probe escalation", {"max_probe": args.n_lists}),
        ):
            ids, ms = timed_search(
                ivf, queries, K, n_probe=4, filter_mask=mask, **kwargs
            )
            filt_rows.append(
                (
                    f"{100 / keep_every:g}% selective, {label}",
                    f"{(ids >= 0).sum(axis=1).mean():.1f} / {K}",
                    f"{recall_at_k(ids, truth, K):.3f}",
                    f"{ms:.2f}",
                )
            )
    print("### Filtered search (nprobe=4, vs exact filtered ground truth)\n")
    table(["Filter", "Results returned", f"Recall@{K}", "ms/query"], filt_rows)


if __name__ == "__main__":
    main()
