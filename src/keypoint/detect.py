from tqdm import tqdm
from pathlib import Path
import torch
import h5py
import kornia.feature as KF
from lightglue import ALIKED
from src.utils import load_torch_image
from src.dataclass import Config
from src.keypoint.rotate import apply_rotate


def feature_ALIKED(
    path_dict: dict[str, Path | list[Path]],
    scene: str,
    config: Config,
) -> None:
    aliked_config = config.aliked_config
    image_paths = path_dict["image_paths"]
    feature_dir = path_dict["feature_dir"]
    is_rotate = getattr(config, "rotate", False)

    detector = (
        ALIKED(
            max_num_keypoints=aliked_config["max_num_keypoints"],
            detection_threshold=aliked_config["detection_threshold"],
            resize=aliked_config["resize_to"],
        )
        .eval()
        .to(config.device, torch.float32)
    ).extract

    with h5py.File(feature_dir / "keypoints.h5", mode="w") as f_keypoints, h5py.File(feature_dir / "descriptors.h5", mode="w") as f_descriptors:
        for path in tqdm(image_paths, desc="Detecting keypoints / ALIKED"):
            key = path.name

            with torch.inference_mode():
                image = load_torch_image(path, device=config.device).to(torch.float32)
                if is_rotate and ("air-to-ground" in config.cat2scenes_dict) and (scene in config.cat2scenes_dict["air-to-ground"]):
                    features = apply_rotate(path, image, detector, config)
                else:
                    features = detector(image)
                f_keypoints[key] = features["keypoints"].squeeze().detach().cpu().numpy()
                f_descriptors[key] = features["descriptors"].squeeze().detach().cpu().numpy()


def detect_keypoints(
    path_dict: dict[str, Path | list[Path]],
    scene: str,
    config: Config,
) -> None:
    detected = False

    if "ALIKED" in config.detector:
        feature_ALIKED(path_dict, scene, config)
        detected = True
    if not detected:
        raise ValueError("No detector specified")
