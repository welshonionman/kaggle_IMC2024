from pathlib import Path
from copy import deepcopy
import numpy as np


def find_optimal_reconstruction(maps: dict) -> int:
    images_registered = 0
    best_idx = None

    print("\nFinding the best reconstruction")

    if isinstance(maps, dict):
        for idx1, rec in maps.items():
            print(idx1, rec.summary())
            try:
                if len(rec.images) > images_registered:
                    images_registered = len(rec.images)
                    best_idx = idx1
            except Exception:
                continue

    return best_idx


def parse_reconstructed_object(
    results: dict,
    dataset: str,
    scene: str,
    maps: dict,
    best_idx: int,
    train_test: str,
    base_path: Path,
) -> dict:
    if best_idx is not None:
        for k, im in maps[best_idx].images.items():
            key = base_path / train_test / scene / "images" / im.name
            results[dataset][scene][key] = {}
            results[dataset][scene][key]["R"] = deepcopy(im.cam_from_world.rotation.matrix())
            results[dataset][scene][key]["t"] = deepcopy(np.array(im.cam_from_world.translation))
    return results
