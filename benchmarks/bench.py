"""TurboQuant benchmark: compression ratio vs recall vs speed.

Run:  python benchmarks/bench.py [--n 50000] [--dim 128] [--queries 200]

Measures every method against exact float32 ground truth on synthetic
clustered embeddings and prints a markdown table (the one in the README).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from turboquant import (
    AdaptiveBitQuantizer,
    FlatIndex,
    IVFPQIndex,
    ProductQuantizer,
    QuantizedFlatIndex,
    ScalarQuantizer,
    TurboIndex,
    recall_at_k,
    train_base_query_split,
)

K = 10


def timed_search(index, queries, k, **kwargs):
    t0 = time.perf_counter()
    ids, _ = index.search(queries, k, **kwargs)
    ms = (time.perf_counter() - t0) * 1000 / len(queries)
    return ids, ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--n-lists", type=int, default=128)
    ap.add_argument("--n-probe", type=int, default=16)
    args = ap.parse_args()

    n_train = min(args.n, 20_000)
    print(f"dataset: n={args.n} dim={args.dim} queries={args.queries} (synthetic clustered)")
    train, base, queries = train_base_query_split(
        n_train, args.n, args.queries, args.dim, n_clusters=100, seed=0
    )
    raw_bytes = base.nbytes

    flat = FlatIndex(args.dim)
    flat.add(base)
    gt, flat_ms = timed_search(flat, queries, K)
    print(f"ground truth built (flat float32: {raw_bytes / 1e6:.1f} MB, {flat_ms:.2f} ms/query)\n")

    rows: list[tuple[str, float, float, float, float]] = []

    def bench(name, index, mem_bytes, **search_kwargs):
        ids, ms = timed_search(index, queries, K, **search_kwargs)
        r = recall_at_k(ids, gt, K)
        rows.append((name, raw_bytes / mem_bytes, mem_bytes / 1e6, r, ms))

    rows.append(("Flat float32 (exact)", 1.0, raw_bytes / 1e6, 1.0, flat_ms))

    for bits in (8, 4):
        sq = ScalarQuantizer(args.dim, bits=bits).train(train)
        idx = QuantizedFlatIndex(sq)
        idx.add(base)
        bench(f"Flat SQ{bits}", idx, idx.memory_bytes)

    aq = AdaptiveBitQuantizer(args.dim, avg_bits=4.0).train(train)
    idx = QuantizedFlatIndex(aq)
    idx.add(base)
    bench("Flat AdaptiveBits (avg 4b)", idx, idx.memory_bytes)

    for m in (args.dim // 16, args.dim // 8):
        pq = ProductQuantizer(args.dim, n_subspaces=m).train(train)
        idx = QuantizedFlatIndex(pq)
        idx.add(base)
        bench(f"Flat PQ M={m}", idx, idx.memory_bytes)

    for m in (args.dim // 16, args.dim // 8, args.dim // 4):
        ivf = IVFPQIndex(args.dim, n_lists=args.n_lists, n_subspaces=m).train(train)
        ivf.add(base)
        bench(f"IVFPQ M={m} nprobe={args.n_probe}", ivf, ivf.memory_bytes, n_probe=args.n_probe)

    for budget in (16, 24, 32, 64):
        turbo = TurboIndex(args.dim, budget_bytes=budget, n_lists=args.n_lists).train(train)
        turbo.add(base)
        split = (
            f"PQ{turbo.ivfpq.pq.M}+PQ{turbo.refiner.M}"
            if turbo.refine_kind == "pq"
            else f"PQ{turbo.ivfpq.pq.M}+adaptive"
        )
        bench(
            f"TurboIndex {budget}B ({split})",
            turbo,
            turbo.memory_bytes,
            n_probe=args.n_probe,
            rerank_factor=8,
        )

    print(f"| Method | Compression | Memory (MB) | Recall@{K} | ms/query |")
    print("|---|---|---|---|---|")
    for name, ratio, mb, r, ms in rows:
        print(f"| {name} | {ratio:.1f}x | {mb:.1f} | {r:.3f} | {ms:.2f} |")


if __name__ == "__main__":
    main()
