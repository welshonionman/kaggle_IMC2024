import os
import time
import numpy as np
import random
from pathlib import Path
import torch
import kornia as K
from src.colmapdb.database import COLMAPDatabase
from src.colmapdb.h5_to_db import add_keypoints, add_matches


def arr_to_str(a):
    """Returns ;-separated string representing the input"""
    return ";".join([str(x) for x in a.reshape(-1)])


def load_torch_image(file_name: Path | str, device=torch.device("cpu")):
    """Loads an image and adds batch dimension"""
    img = K.io.load_image(file_name, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img


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
    path: Path,
    feature_dir: Path,
    database_path: Path,
) -> None:
    """Adds keypoints into colmap"""
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(db, feature_dir, path, "", "simple-pinhole", single_camera)
    add_matches(db, feature_dir, fname_to_id)
    db.commit()
