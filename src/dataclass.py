from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class Config:
    exp_name: str
    is_kaggle_notebook: bool
    valid_image_num: int
    log_path: Path
    gt_csv_path: Path

    base_path: Path
    feature_dir: Path
    target_scene: list[str]
    device: torch.device

    pair_matching_args: dict

    keypoint_detection_args: dict

    keypoint_distances_args: dict

    colmap_mapper_options: dict
