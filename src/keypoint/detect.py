from tqdm import tqdm
from pathlib import Path
import torch
import h5py
from lightglue import ALIKED
import kornia.feature as KF
from src.utils import load_torch_image
from src.dataclass import ALIKEDConfig, Config


def feature_ALIKED(
    path_dict: dict[str, Path | list[Path]],
    config: Config,
    aliked_config: ALIKEDConfig,
) -> None:
    image_paths = path_dict["image_paths"]
    feature_dir = path_dict["feature_dir"]
    device = config.device
    num_features = aliked_config["max_num_keypoints"]
    detection_threshold = aliked_config["detection_threshold"]
    resize_to = aliked_config["resize_to"]

    dtype = torch.float32  # ALIKED has issues with float16
    extractor = ALIKED(max_num_keypoints=num_features, detection_threshold=detection_threshold, resize=resize_to).eval().to(device, dtype)

    with h5py.File(feature_dir / "keypoints.h5", mode="w") as f_keypoints, h5py.File(feature_dir / "descriptors.h5", mode="w") as f_descriptors:
        for path in tqdm(image_paths, desc="Computing keypoints / ALIKED"):
            key = path.name

            with torch.inference_mode():
                image = load_torch_image(path, device=device).to(dtype)
                features = extractor.extract(image)

                f_keypoints[key] = features["keypoints"].squeeze().detach().cpu().numpy()
                f_descriptors[key] = features["descriptors"].squeeze().detach().cpu().numpy()


def detect_keypoints(
    path_dict: dict[str, Path | list[Path]],
    config: Config,
) -> None:
    feature_ALIKED(path_dict, config, config.aliked_config)

