from pathlib import Path
from src.dataclass import Config
from src.keypoint.detector import feature_lightglue_common, feature_kornia_common, feature_loftr


def detect_keypoints(
    path_dict: dict[str, Path | list[Path]],
    index_pairs: list[tuple[int, int]],
    scene: str,
    config: Config,
) -> None:
    detected = False

    files_matches = []
    detectors = [detector.lower() for detector in config.detector]

    if "aliked" in detectors:
        model_name = "aliked"
        feature_lightglue_common(model_name, path_dict, index_pairs, scene, config)
        files_matches.append(f"{path_dict['feature_dir']}/matches_{model_name}.h5")
        detected = True

    if "dedode" in detectors:
        model_name = "dedodeg"
        feature_kornia_common(model_name, path_dict, index_pairs, scene, config)
        files_matches.append(f"{path_dict['feature_dir']}/matches_{model_name}.h5")
        detected = True

    if "disk" in detectors:
        model_name = "disk"
        feature_kornia_common(model_name, path_dict, index_pairs, scene, config)
        files_matches.append(f"{path_dict['feature_dir']}/matches_{model_name}.h5")
        detected = True

    if "sift" in detectors:
        model_name = "sift"
        feature_kornia_common(model_name, path_dict, index_pairs, scene, config)
        files_matches.append(f"{path_dict['feature_dir']}/matches_{model_name}.h5")
        detected = True

    if "loftr" in detectors:
        model_name = "loftr"
        feature_loftr(path_dict, index_pairs, scene, config)
        files_matches.append(f"{path_dict['feature_dir']}/matches_{model_name}.h5")
        detected = True

    if not detected:
        raise ValueError("No detector specified")

    return files_matches
