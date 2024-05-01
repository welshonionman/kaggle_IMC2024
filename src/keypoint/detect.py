from tqdm import tqdm
from pathlib import Path
import torch
import h5py
from lightglue import ALIKED
import kornia.feature as KF
from src.utils import load_torch_image
from src.dataclass import ALIKEDConfig, Config
from src.keypoint.rotate import detect_rot, rotate_kpts


def save_features(
    extractor,
    path_dict: dict[str, Path | list[Path]],
    config: Config,
) -> None:
    image_paths = path_dict["image_paths"]
    feature_dir = path_dict["feature_dir"]
    is_rotate = getattr(config, "rotate", False)

    with h5py.File(feature_dir / "keypoints.h5", mode="w") as f_keypoints, h5py.File(feature_dir / "descriptors.h5", mode="w") as f_descriptors:
        for path in tqdm(image_paths, desc="Detecting keypoints / ALIKED"):
            key = path.name

            with torch.inference_mode():
                image = load_torch_image(path, device=config.device).to(torch.float32)
                if is_rotate:
                    correct_rot = detect_rot(path)
                    image = torch.rot90(image, correct_rot, dims=(2, 3))

                features = extractor.extract(image)

                if is_rotate:
                    tmp_np = features["keypoints"][0].cpu().numpy()
                    rot_kpts = rotate_kpts(tmp_np, (image.shape[3], image.shape[2]), correct_rot)
                    keypoints_rot = torch.from_numpy(rot_kpts).unsqueeze(0).to(config.device)
                    features["keypoints"] = keypoints_rot
                f_keypoints[key] = features["keypoints"].squeeze().detach().cpu().numpy()
                f_descriptors[key] = features["descriptors"].squeeze().detach().cpu().numpy()


def feature_ALIKED(
    path_dict: dict[str, Path | list[Path]],
    config: Config,
    aliked_config: ALIKEDConfig,
) -> None:
    extractor = (
        ALIKED(
            max_num_keypoints=aliked_config["max_num_keypoints"],
            detection_threshold=aliked_config["detection_threshold"],
            resize=aliked_config["resize_to"],
        )
        .eval()
        .to(config.device, torch.float32)
    )

    save_features(extractor, path_dict, config)


def detect_keypoints(
    path_dict: dict[str, Path | list[Path]],
    config: Config,
) -> None:
    feature_ALIKED(path_dict, config, config.aliked_config)
