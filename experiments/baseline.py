import torch
import shutil
from pathlib import Path
import kornia as K
from src.pipeline import run_from_config


class Config:
    base_path: Path = Path("/kaggle/input/image-matching-challenge-2024")
    feature_dir: Path = Path("/kaggle/.sample/")
    target_scene: list[str] = [
        # "church",
        "dioscuri",
        # "lizard",
        "multi-temporal-temple-baalshamin",
        # "pond",
        # "transp_obj_glass_cup",
        # "transp_obj_glass_cylinder",
    ]
    device: torch.device = K.utils.get_cuda_device_if_available(0)

    # get_image_pairs function's arguments
    pair_matching_args = {
        "model_name": "/kaggle/input/dinov2/pytorch/base/1",
        "similarity_threshold": 0.3,
        "tolerance": 500,
        "min_matches": 50,
        "exhaustive_if_less": 50,
        "p": 2.0,
    }

    # detect_keypoints function's arguments
    keypoint_detection_args = {
        "num_features": 4096,
        "resize_to": 1024,
    }

    # keypoint_distances function's arguments
    keypoint_distances_args = {
        "min_matches": 15,
        "verbose": False,
    }

    # import_into_colmap function's arguments
    colmap_mapper_options = {
        "min_model_size": 3,  # By default colmap does not generate a reconstruction if less than 10 images are registered. Lower it to 3.
        "max_num_models": 2,
        "num_threads": 1,
    }


if __name__ == "__main__":
    shutil.rmtree(Config.feature_dir, ignore_errors=True)
    run_from_config(Config)
