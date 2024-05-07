from tqdm import tqdm
from pathlib import Path
import torch
import h5py
import kornia.feature as KF
from lightglue import ALIKED
from src.utils import load_torch_image
from src.dataclass import Config, ALIKEDConfig
from src.keypoint.rotate import apply_rotate


def detector_ALIKED(
    aliked_config: ALIKEDConfig,
    device: torch.device,
) -> torch.nn.Module:
    detector = (
        ALIKED(
            max_num_keypoints=aliked_config["max_num_keypoints"],
            detection_threshold=aliked_config["detection_threshold"],
            resize=aliked_config["resize_to"],
        )
        .eval()
        .to(device, torch.float32)
    ).extract

    return detector


def detector_by_scene_ALIKED(
    scene: str,
    config: Config,
) -> torch.nn.Module:
    detector_transp = getattr(config, "detector_transp", False)

    if (detector_transp) and ("transparent" in config.cat2scenes_dict) and (scene in config.cat2scenes_dict["transparent"]):
        detector = detector_ALIKED(config.aliked_config_transparent, config.device)
    else:
        detector = detector_ALIKED(config.aliked_config, config.device)

    return detector


def feature_ALIKED(
    path_dict: dict[str, Path | list[Path]],
    scene: str,
    config: Config,
) -> None:
    image_paths = path_dict["image_paths"]
    feature_dir = path_dict["feature_dir"]
    is_rotate = getattr(config, "rotate", False)

    with (
        h5py.File(feature_dir / "keypoints.h5", mode="w") as f_keypoints,
        h5py.File(feature_dir / "descriptors.h5", mode="w") as f_descriptors,
    ):
        for path in tqdm(image_paths, desc="Detecting keypoints / ALIKED"):
            key = path.name

            with torch.inference_mode():
                image = load_torch_image(path, device=config.device).to(torch.float32)

                detector = detector_by_scene_ALIKED(scene, config)

                if is_rotate and ("air-to-ground" in config.cat2scenes_dict) and (scene in config.cat2scenes_dict["air-to-ground"]):
                    features = apply_rotate(path, image, detector, config)
                else:
                    features = detector(image)

                f_keypoints[key] = features["keypoints"].squeeze().detach().cpu().numpy()
                f_descriptors[key] = features["descriptors"].squeeze().detach().cpu().numpy()


def feature_DeDoDe(
    path_dict: dict[str, Path | list[Path]],
    scene: str,
    config: Config,
) -> None:
    image_paths = path_dict["image_paths"]
    feature_dir = path_dict["feature_dir"]
    is_rotate = getattr(config, "rotate", False)
    detector = (
        KF.DeDoDe()
        .from_pretrained(
            detector_weights="L-upright",
            descriptor_weights="B-upright",
        )
        .eval()
        .to(config.device, torch.float16)
    )

    with (
        h5py.File(feature_dir / "keypoints.h5", mode="w") as f_keypoints,
        h5py.File(feature_dir / "descriptors.h5", mode="w") as f_descriptors,
    ):
        for path in tqdm(image_paths, desc="Detecting keypoints / DeDoDe"):
            key = path.name
            with torch.inference_mode():
                image = load_torch_image(path, device=config.device).to(torch.float16)
                if is_rotate and ("air-to-ground" in config.cat2scenes_dict) and (scene in config.cat2scenes_dict["air-to-ground"]):
                    features = apply_rotate(path, image, detector, config)
                else:
                    keypoints, scores, features = detector(image)
                f_keypoints[key] = keypoints.squeeze().detach().cpu().numpy()
                f_descriptors[key] = features.squeeze().detach().cpu().numpy()


def detect_keypoints(
    path_dict: dict[str, Path | list[Path]],
    scene: str,
    config: Config,
) -> None:
    detected = False

    if "ALIKED" in config.detector:
        feature_ALIKED(path_dict, scene, config)
        detected = True
    if "DeDoDe" in config.detector:
        feature_DeDoDe(path_dict, scene, config)
        detected = True
    if not detected:
        raise ValueError("No detector specified")
