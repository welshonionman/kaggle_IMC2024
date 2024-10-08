from pathlib import Path
from src.dataclass import Config
from src.keypoint.detector import (
    feature_loftr,
    # feature_eloftr,
    feature_aliked,
    feature_superpoint,
    feature_doghardnet,
    feature_dedode,
    feature_disk,
    feature_sift,
)


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
        feature_aliked(model_name, path_dict, index_pairs, scene, config)
        files_matches.append(f"{path_dict['feature_dir']}/matches_{model_name}.h5")
        detected = True

    if "superpoint" in detectors:
        model_name = "superpoint"
        feature_superpoint(model_name, path_dict, index_pairs, scene, config)
        files_matches.append(f"{path_dict['feature_dir']}/matches_{model_name}.h5")
        detected = True

    if "doghardnet" in detectors:
        model_name = "doghardnet"
        feature_doghardnet(model_name, path_dict, index_pairs, scene, config)
        files_matches.append(f"{path_dict['feature_dir']}/matches_{model_name}.h5")
        detected = True

    if "dedode" in detectors:
        model_name = "dedodeg"
        feature_dedode(model_name, path_dict, index_pairs, scene, config)
        files_matches.append(f"{path_dict['feature_dir']}/matches_{model_name}.h5")
        detected = True

    if "disk" in detectors:
        model_name = "disk"
        feature_disk(model_name, path_dict, index_pairs, scene, config)
        files_matches.append(f"{path_dict['feature_dir']}/matches_{model_name}.h5")
        detected = True

    if "sift" in detectors:
        model_name = "sift"
        feature_sift(model_name, path_dict, index_pairs, scene, config)
        files_matches.append(f"{path_dict['feature_dir']}/matches_{model_name}.h5")
        detected = True

    if "loftr" in detectors:
        model_name = "loftr"
        feature_loftr(path_dict, index_pairs, scene, config)
        files_matches.append(f"{path_dict['feature_dir']}/matches_{model_name}.h5")
        detected = True

    if "eloftr" in detectors:
        model_name = "eloftr"
        # feature_eloftr(path_dict, index_pairs, scene, config)
        files_matches.append(f"{path_dict['feature_dir']}/matches_{model_name}.h5")
        detected = True

    if not detected:
        raise ValueError("No detector specified")

    return files_matches
