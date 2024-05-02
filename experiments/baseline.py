import torch
import shutil
from pathlib import Path
import kornia as K
import os
import warnings
from src.pipeline import run_from_config
from src.utils.utils import cat2scenes

warnings.filterwarnings("ignore")


class Config:
    exp_name: str = __file__.split("/")[-1].replace(".py", "")
    is_kaggle_notebook: bool = any("KAGGLE" in item for item in dict(os.environ).keys())
    train_test = "test" if is_kaggle_notebook else "train"

    valid_image_num: int = 9999  # validationに使用する画像数（デバッグ用）
    log_path = Path(f"/kaggle/log/{exp_name}.log")
    gt_csv_path = Path("/kaggle/src/valid_gt.csv")

    base_path: Path = Path("/kaggle/input/image-matching-challenge-2024")
    feature_dir: Path = Path("/kaggle/.sample/")
    cat2scenes_dict = cat2scenes(base_path / train_test / "categories.csv")

    target_scene: list[str] = [
        "church",
        "dioscuri",
        "lizard",
        "multi-temporal-temple-baalshamin",
        "pond",
        "transp_obj_glass_cup",
        "transp_obj_glass_cylinder",
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
    aliked_config = {
        "max_num_keypoints": 4096,
        "resize_to": 1024,
        "detection_threshold": 0.01,
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
    
    rotate: bool = True


if __name__ == "__main__":
    shutil.rmtree(Config.feature_dir, ignore_errors=True)
    run_from_config(config=Config)
