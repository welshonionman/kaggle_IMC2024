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


def keypoint_distances(
    path_dict: dict[str, Path | list[Path]],
    index_pairs: list[tuple[int, int]],
    min_matches: int = 15,
    verbose: bool = True,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Computes distances between keypoints of images.

    Stores output at feature_dir/matches.h5
    """
    image_paths = path_dict["image_paths"]
    feature_dir = path_dict["feature_dir"]
    matcher_params = {
        "width_confidence": -1,
        "depth_confidence": -1,
        "mp": True if "cuda" in str(device) else False,
    }
    matcher = KF.LightGlueMatcher("aliked", matcher_params).eval().to(device)

    with (
        h5py.File(feature_dir / "keypoints.h5", mode="r") as f_keypoints,
        h5py.File(feature_dir / "descriptors.h5", mode="r") as f_descriptors,
        h5py.File(feature_dir / "matches.h5", mode="w") as f_matches,
    ):
        for idx1, idx2 in tqdm(index_pairs, desc="Computing keypoing distances"):
            key1, key2 = image_paths[idx1].name, image_paths[idx2].name

            keypoints1 = torch.from_numpy(f_keypoints[key1][...]).to(device)
            keypoints2 = torch.from_numpy(f_keypoints[key2][...]).to(device)
            descriptors1 = torch.from_numpy(f_descriptors[key1][...]).to(device)
            descriptors2 = torch.from_numpy(f_descriptors[key2][...]).to(device)

            with torch.inference_mode():
                distances, indices = matcher(
                    descriptors1,
                    descriptors2,
                    KF.laf_from_center_scale_ori(keypoints1[None]),
                    KF.laf_from_center_scale_ori(keypoints2[None]),
                )

            # We have matches to consider
            n_matches = len(indices)
            if n_matches:
                if verbose:
                    print(f"{key1}-{key2}: {n_matches} matches")
                # Store the matches in the group of one image
                if n_matches >= min_matches:
                    group = f_matches.require_group(key1)
                    group.create_dataset(key2, data=indices.detach().cpu().numpy().reshape(-1, 2))
