from pathlib import Path
import torch
from lightglue import ALIKED, SuperPoint, DoGHardNet
from lightglue.utils import Extractor
import h5py
from tqdm import tqdm
import numpy as np
import gc
import kornia.feature as KF
from src.utils import load_torch_image
from src.dataclass import Config, LightGlueConfig
from src.keypoint.rotate import apply_rotate


def get_detector(
    model_name: str,
    detector_config: LightGlueConfig,
    device: torch.device,
) -> Extractor:
    dict_model = {
        "aliked": ALIKED,
        "superpoint": SuperPoint,
        "doghardnet": DoGHardNet,
    }

    extractor_class = dict_model[model_name]

    detector = (
        extractor_class(
            model_name=detector_config["model_name"],
            max_num_keypoints=detector_config["max_num_keypoints"],
            detection_threshold=detector_config["detection_threshold"],
            resize=detector_config["resize_to"],
        )
        .eval()
        .to(device, torch.float32)
        .extract
    )

    return detector


def get_detector_by_scene(
    model_name: str,
    scene: str,
    config: Config,
) -> torch.nn.Module:
    detector_transp = getattr(config, "detector_transp", False)

    if (detector_transp) and ("transparent" in config.cat2scenes_dict) and (scene in config.cat2scenes_dict["transparent"]):
        detector = get_detector(model_name, config.aliked_config_transparent, config.device)
    else:
        detector = get_detector(model_name, config.aliked_config, config.device)

    return detector


def detect_common(
    model_name: str,
    image_paths,
    feature_dir: Path,
    scene: str,
    config: Config,
):
    feature_dir.mkdir(parents=True, exist_ok=True)
    is_rotate = getattr(config, "rotate", False)

    with (
        h5py.File(f"{feature_dir}/keypoints_{model_name}.h5", mode="w") as f_keypoints,
        h5py.File(f"{feature_dir}/descriptors_{model_name}.h5", mode="w") as f_descriptors,
    ):
        for path in tqdm(image_paths, desc=f"Detecting keypoints / {model_name}"):
            key = path.name

            with torch.inference_mode():
                image = load_torch_image(path, device=config.device).to(torch.float32)

                detector = get_detector_by_scene(model_name, scene, config)

                if is_rotate and ("air-to-ground" in config.cat2scenes_dict) and (scene in config.cat2scenes_dict["air-to-ground"]):
                    features = apply_rotate(model_name, path, image, detector, config)
                else:
                    features = detector(image)

                kpts = features["keypoints"].reshape(-1, 2).detach().cpu().numpy()
                descs = features["descriptors"].reshape(len(kpts), -1).detach().cpu().numpy()
                f_keypoints[key] = kpts
                f_descriptors[key] = descs


def get_matcher(
    model_name: str,
    config: Config,
) -> KF.LightGlueMatcher:
    matcher = (
        KF.LightGlueMatcher(
            model_name,
            {
                "width_confidence": -1,
                "depth_confidence": -1,
                "mp": True if "cuda" in str(config.device) else False,
            },
        )
        .eval()
        .to(config.device)
    )

    return matcher


def match_with_lightglue_common(
    model_name,
    image_paths,
    index_pairs,
    feature_dir,
    config: Config,
):
    matcher = get_matcher(model_name, config)

    cnt_pairs = 0

    with (
        h5py.File(f"{feature_dir}/keypoints_{model_name}.h5", mode="r") as f_keypoints,
        h5py.File(f"{feature_dir}/descriptors_{model_name}.h5", mode="r") as f_descriptors,
        h5py.File(f"{feature_dir}/matches_{model_name}.h5", mode="w") as f_match,
    ):
        for idx1, idx2 in tqdm(index_pairs, desc="Matching keypoing / LightGlue"):
            key1, key2 = image_paths[idx1].name, image_paths[idx2].name

            keypoints1 = torch.from_numpy(f_keypoints[key1][...]).to(config.device)
            keypoints2 = torch.from_numpy(f_keypoints[key2][...]).to(config.device)
            descriptors1 = torch.from_numpy(f_descriptors[key1][...]).to(config.device)
            descriptors2 = torch.from_numpy(f_descriptors[key2][...]).to(config.device)

            with torch.inference_mode():
                dists, idxs = matcher(
                    descriptors1,
                    descriptors2,
                    KF.laf_from_center_scale_ori(keypoints1[None]),
                    KF.laf_from_center_scale_ori(keypoints2[None]),
                )

            if len(idxs) == 0:
                continue

            n_matches = len(idxs)
            keypoints1 = keypoints1[idxs[:, 0], :].cpu().numpy().reshape(-1, 2).astype(np.float32)
            keypoints2 = keypoints2[idxs[:, 1], :].cpu().numpy().reshape(-1, 2).astype(np.float32)
            group = f_match.require_group(key1)

            if n_matches >= config.matching_config["min_matches"]:
                group.create_dataset(key2, data=np.concatenate([keypoints1, keypoints2], axis=1))
                cnt_pairs += 1
                if config.matching_config["verbose"]:
                    print(f"{key1}-{key2}: {n_matches} matches @ {cnt_pairs}th pair({model_name}+lightglue)")
            else:
                if config.matching_config["verbose"]:
                    print(f"{key1}-{key2}: {n_matches} matches --> skipped")


def feature_lightglue_common(
    model_name: str,
    path_dict: dict[str, Path | list[Path]],
    index_pairs: list[tuple[int, int]],
    scene: str,
    config: Config,
):
    image_paths = path_dict["image_paths"]
    feature_dir = path_dict["feature_dir"]

    detect_common(
        model_name,
        image_paths,
        feature_dir,
        scene,
        config,
    )
    gc.collect()

    match_with_lightglue_common(
        model_name,
        image_paths,
        index_pairs,
        feature_dir,
        config,
    )
    gc.collect()

    print(f"Features matched ({model_name}+LightGlue)")
