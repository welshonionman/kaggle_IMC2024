import os
import time
import numpy as np
import random
import pandas as pd
from collections import defaultdict
from pathlib import Path
import torch
import kornia as K
from src.colmapdb.database import COLMAPDatabase
from src.colmapdb.h5_to_db import add_keypoints, add_matches
from src.dataclass import Config


def arr_to_str(a):
    """Returns ;-separated string representing the input"""
    return ";".join([str(x) for x in a.reshape(-1)])


def load_torch_image(file_name: Path | str, load_type="RGB32", device=torch.device("cpu")):
    if load_type == "RGB32":
        img = K.io.load_image(file_name, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    elif load_type == "GRAY32":
        img = K.io.load_image(file_name, K.io.ImageLoadType.GRAY32, device=device)[None, ...]
    else:
        raise ValueError(f"Unknown load type: {load_type}")
    return img


def cat2scenes(cat_csv_path: Path) -> dict[str, list[str]]:
    cat2scenes_dict = defaultdict(list)
    cat_df = pd.read_csv(cat_csv_path)
    for row in cat_df.itertuples():
        cats = row.categories.split(";")
        for cat in cats:
            cat2scenes_dict[cat].append(row.scene)
    return dict(cat2scenes_dict)


def preparation(
    target_dict: dict,
    results: dict,
    config: Config,
) -> tuple[dict, dict]:
    data_dict = target_dict["data_dict"]
    dataset = target_dict["dataset"]
    scene = target_dict["scene"]

    images_dir = data_dict[dataset][scene][0].parent
    image_paths = data_dict[dataset][scene]
    if not config.is_kaggle_notebook:
        image_paths = image_paths[: config.valid_image_num]

    results[dataset][scene] = {}

    feature_dir = config.feature_dir / f"{dataset}_{scene}"
    feature_dir.mkdir(parents=True, exist_ok=True)

    database_path = feature_dir / "colmap.db"
    if database_path.exists():
        database_path.unlink()

    path_dict = {"images_dir": images_dir, "image_paths": image_paths, "feature_dir": feature_dir, "database_path": database_path}
    return path_dict, results


def set_seed(seed=42, cudnn_deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True


def timer(func):
    def wrapper(*args, **kwargs):
        config = kwargs["config"]
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"\nElapsed time: {elapsed_time:.2f} seconds",
            file=open(config.log_path, "a"),
        )
        return result

    return wrapper


def import_into_colmap(
    images_dir: Path,
    feature_dir: Path,
    database_path: Path,
) -> None:
    """Adds keypoints into colmap"""
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(db, feature_dir, images_dir, "", "simple-pinhole", single_camera)
    add_matches(db, feature_dir, fname_to_id)
    db.commit()
