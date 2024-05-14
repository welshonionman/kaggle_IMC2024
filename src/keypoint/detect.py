from pathlib import Path
from src.dataclass import Config
from src.keypoint.detector import detect_lightglue_common, detect_kornia_common, detect_loftr


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
        detect_lightglue_common(model_name, path_dict, index_pairs, scene, config)
        files_matches.append(f"{path_dict['feature_dir']}/matches_{model_name}.h5")
        detected = True

    if "dedode" in detectors:
        model_name = "dedodeg"
        detect_kornia_common(model_name, path_dict, index_pairs, scene, config)
        files_matches.append(f"{path_dict['feature_dir']}/matches_{model_name}.h5")
        detected = True

    if "disk" in detectors:
        model_name = "disk"
        detect_kornia_common(model_name, path_dict, index_pairs, scene, config)
        files_matches.append(f"{path_dict['feature_dir']}/matches_{model_name}.h5")
        detected = True

    if "sift" in detectors:
        model_name = "sift"
        detect_kornia_common(model_name, path_dict, index_pairs, scene, config)
        files_matches.append(f"{path_dict['feature_dir']}/matches_{model_name}.h5")
        detected = True

    if "loftr" in detectors:
        model_name = "loftr"
        detect_loftr(path_dict, index_pairs, scene, config)
        files_matches.append(f"{path_dict['feature_dir']}/matches_{model_name}.h5")
        detected = True

    if not detected:
        raise ValueError("No detector specified")

    return files_matches
