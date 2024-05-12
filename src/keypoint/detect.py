from pathlib import Path
from src.dataclass import Config
from src.keypoint.detector import detect_lightglue_common
from src.keypoint.detector.loftr import detect_loftr


def detect_keypoints(
    path_dict: dict[str, Path | list[Path]],
    index_pairs: list[tuple[int, int]],
    scene: str,
    config: Config,
) -> None:
    detected = False

    files_matches = []

    if "ALIKED" in config.detector:
        model_name = "aliked"
        detect_lightglue_common(
            model_name,
            path_dict,
            index_pairs,
            scene,
            config,
        )
        files_matches.append(f"{path_dict['feature_dir']}/matches_{model_name}.h5")
        detected = True

    if "LoFTR" in config.detector:
        detect_loftr(
            path_dict,
            index_pairs,
            scene,
            config,
        )
        files_matches.append(f"{path_dict['feature_dir']}/matches_LoFTR.h5")
        detected = True

    if not detected:
        raise ValueError("No detector specified")

    return files_matches
