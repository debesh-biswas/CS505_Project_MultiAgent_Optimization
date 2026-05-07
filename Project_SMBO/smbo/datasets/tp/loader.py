import random
from typing import List, Any, Optional


def load_batch(
    tp_split: str = "validation",
    batch_size: int = 20,
    batch_seed: int = 42,
    stratify: bool = False,
    per_level: Optional[int] = None,
) -> List[Any]:
    """Load a seeded mini-batch of TravelPlanner rows from Hugging Face.

    Args:
        stratify: If True, sample equally from easy/medium/hard levels.
            Uses ``per_level`` per difficulty; ``batch_size`` is ignored.
        per_level: Queries per difficulty level when ``stratify=True``.
            Defaults to ``batch_size // 3``.
    """
    from datasets import load_dataset

    if tp_split not in ("train", "validation", "test"):
        raise ValueError(f"tp_split must be train|validation|test, got {tp_split!r}")

    print(f"Loading TravelPlanner ({tp_split})...")
    ds = load_dataset("osunlp/TravelPlanner", tp_split)[tp_split]
    rng = random.Random(batch_seed)

    if stratify:
        n = per_level if per_level is not None else max(1, batch_size // 3)
        levels = ["easy", "medium", "hard"]
        batch = []
        for lvl in levels:
            pool = [i for i in range(len(ds)) if ds[i].get("level", "").lower() == lvl]
            k = min(n, len(pool))
            chosen = rng.sample(pool, k)
            batch.extend([ds[i] for i in chosen])
            print(f"  {lvl}: {k} queries selected (pool size: {len(pool)})")
        print(f"Selected {len(batch)} queries total (stratified: {n} per level).")
    else:
        k = min(batch_size, len(ds))
        indices = rng.sample(range(len(ds)), k)
        batch = [ds[i] for i in indices]
        print(f"Selected {len(batch)} queries for evaluation.")

    return batch
