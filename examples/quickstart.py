"""TurboQuant quickstart: index 100k vectors under a 32-byte/vector budget.

Run:  python examples/quickstart.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from turboquant import FlatIndex, TurboIndex, recall_at_k, train_base_query_split

DIM, K = 128, 10

print("generating synthetic embeddings (clusters + variance decay)...")
train, base, queries = train_base_query_split(
    n_train=20_000, n_base=100_000, n_query=100, dim=DIM
)

print("building exact ground truth (float32 brute force)...")
flat = FlatIndex(DIM)
flat.add(base)
gt, _ = flat.search(queries, K)

print("training TurboIndex (budget: 32 bytes/vector)...")
index = TurboIndex(dim=DIM, budget_bytes=32, n_lists=256)
index.train(train)
index.add(base)

ids, dists = index.search(queries, k=K, n_probe=16, rerank_factor=8)

raw_mb = base.nbytes / 1e6
idx_mb = index.memory_bytes / 1e6
print(
    f"\nraw float32:  {raw_mb:8.1f} MB"
    f"\ncompressed:   {idx_mb:8.1f} MB  ({raw_mb / idx_mb:.0f}x smaller,"
    f" {index.bytes_per_vector:.0f} bytes/vector)"
    f"\nrecall@{K}:    {recall_at_k(ids, gt, K):8.3f}"
)
