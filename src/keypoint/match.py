from tqdm import tqdm
from pathlib import Path
import torch
import h5py
import kornia.feature as KF
from src.dataclass import Config


model_dict = {
    "ALIKED": "aliked",
    "DeDoDe": "dedodeg",
}


def match_keypoints(
    path_dict: dict[str, Path | list[Path]],
    index_pairs: list[tuple[int, int]],
    config: Config,
) -> None:
    image_paths = path_dict["image_paths"]
    feature_dir = path_dict["feature_dir"]
    min_matches = config.matching_config["min_matches"]
    verbose = config.matching_config["verbose"]

    matcher_params = {
        "width_confidence": -1,
        "depth_confidence": -1,
        "mp": True if "cuda" in str(config.device) else False,
    }
    model = model_dict[config.detector[0]]
    matcher = (
        KF.LightGlueMatcher(
            model,
            matcher_params,
        )
        .eval()
        .to(config.device)
    )

    with (
        h5py.File(feature_dir / "keypoints.h5", mode="r") as f_keypoints,
        h5py.File(feature_dir / "descriptors.h5", mode="r") as f_descriptors,
        h5py.File(feature_dir / "matches.h5", mode="w") as f_matches,
    ):
        for idx1, idx2 in tqdm(index_pairs, desc="Matching keypoing"):
            key1, key2 = image_paths[idx1].name, image_paths[idx2].name

            keypoints1 = torch.from_numpy(f_keypoints[key1][...]).to(config.device)
            keypoints2 = torch.from_numpy(f_keypoints[key2][...]).to(config.device)
            descriptors1 = torch.from_numpy(f_descriptors[key1][...]).to(config.device)
            descriptors2 = torch.from_numpy(f_descriptors[key2][...]).to(config.device)

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
