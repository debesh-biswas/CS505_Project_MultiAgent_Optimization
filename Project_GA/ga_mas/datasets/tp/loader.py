import random
from typing import List, Any


def load_batch(
    tp_split: str = "validation",
    batch_size: int = 20,
    batch_seed: int = 42,
) -> List[Any]:
    """Load a seeded mini-batch of TravelPlanner rows from Hugging Face."""
    from datasets import load_dataset

    if tp_split not in ("train", "validation", "test"):
        raise ValueError(f"tp_split must be train|validation|test, got {tp_split!r}")

    print(f"Loading TravelPlanner ({tp_split})...")
    ds = load_dataset("osunlp/TravelPlanner", tp_split)[tp_split]
    if batch_size == 9 and tp_split == "train":
        import random
        random.seed(batch_seed)
        easy_idx = [i for i, x in enumerate(ds["level"]) if x == "easy"]
        medium_idx = [i for i, x in enumerate(ds["level"]) if x == "medium"]
        hard_idx = [i for i, x in enumerate(ds["level"]) if x == "hard"]
        
        selected = (
            random.sample(easy_idx, 3) +
            random.sample(medium_idx, 3) +
            random.sample(hard_idx, 3)
        )
        ds = ds.select(selected)
        print(f"Loaded 9 stratified samples from {tp_split} (3 Easy, 3 Medium, 3 Hard).")
    elif batch_size > 0:
        import random
        rng = random.Random(batch_seed)
        k = min(batch_size, len(ds))
        indices = rng.sample(range(len(ds)), k)
        ds = ds.select(indices)
    batch = [row for row in ds]
    print(f"Selected {len(batch)} queries for evaluation.")
    return batch
