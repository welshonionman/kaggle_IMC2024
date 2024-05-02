from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class ALIKEDConfig:
    max_num_keypoints: int
    detection_threshold: float
    resize: int


@dataclass
class Config:
    exp_name: str
    is_kaggle_notebook: bool
    train_test: str

    valid_image_num: int
    log_path: Path
    gt_csv_path: Path

    base_path: Path
    feature_dir: Path
    cat2scenes_dict: dict
    target_scene: list[str]
    device: torch.device

    pair_matching_args: dict

    aliked_config: ALIKEDConfig

    keypoint_distances_args: dict

    colmap_mapper_options: dict

    rotate: bool
