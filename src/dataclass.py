from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class Config:
    validation: bool
    base_path: Path
    feature_dir: Path

    device: torch.device

    pair_matching_args: dict

    keypoint_detection_args: dict

    keypoint_distances_args: dict

    colmap_mapper_options: dict
