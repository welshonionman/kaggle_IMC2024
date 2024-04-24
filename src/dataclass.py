from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class Config:
    base_path: Path
    feature_dir: Path
    target_scene: list[str]
    device: torch.device

    pair_matching_args: dict

    keypoint_detection_args: dict

    keypoint_distances_args: dict

    colmap_mapper_options: dict
