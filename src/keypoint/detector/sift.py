from tqdm import tqdm
from pathlib import Path
import torch
import h5py
import numpy as np
import gc
import torch.nn.functional as F
import kornia.feature as KF
from src.utils import load_torch_image
from src.dataclass import Config
from src.keypoint.rotate import apply_rotate


def get_detector(
    model_name: str,
    detector_config: dict,
    device: torch.device,
) -> KF.DeDoDe:

    detector = KF.SIFTFeatureScaleSpace(upright=True).to(device)

    return detector


def get_detector_by_scene(
    model_name: str,
    scene: str,
    config: Config,
) -> torch.nn.Module:
    # detector_transp = getattr(config, "detector_transp", False)

    # if (detector_transp) and ("transparent" in config.cat2scenes_dict) and (scene in config.cat2scenes_dict["transparent"]):
    #     detector = get_detector(model_name, config.aliked_config_transparent, config.device)
    # else:
    #     detector = get_detector(model_name, config.aliked_config, config.device)

    detector = get_detector(model_name, config.aliked_config, config.device)

    return detector


def pad_to_multiple_of_16(tensor):
    _, _, h, w = tensor.size()  # テンソルのサイズを取得
    pad_h = (16 - h % 16) % 16  # 高さ方向のパディング量を計算
    pad_w = (16 - w % 16) % 16  # 幅方向のパディング量を計算

    padding = (0, pad_w, 0, pad_h)  # パディングの順序は (left, right, top, bottom)
    padded_tensor = F.pad(tensor, padding, mode="constant", value=0)  # ゼロパディングを適用

    return padded_tensor


def load_resize_image(model_name: str, path: Path, resize_to: int, config: Config):
    image = load_torch_image(path, load_type="GRAY32", device=config.device).to(torch.float32)
    original_shape = image.shape  # shape=(B, C, H, W)
    ratio = resize_to / min([image.shape[2], image.shape[3]])
    h, w = int(image.shape[2] * ratio), int(image.shape[3] * ratio)
    image = F.interpolate(image, size=(h, w), mode="bilinear", align_corners=False)

    return image, original_shape


def adjust_keypoints_scale(image, ori_shape, mkpts, rot):
    if rot in [0, 2]:
        mkpts[:, 0] *= float(ori_shape[3]) / float(image.shape[3])
        mkpts[:, 1] *= float(ori_shape[2]) / float(image.shape[2])
    else:
        mkpts[:, 0] *= float(ori_shape[2]) / float(image.shape[3])
        mkpts[:, 1] *= float(ori_shape[3]) / float(image.shape[2])

    return mkpts


def detect_common(
    model_name: str,
    image_paths,
    feature_dir: Path,
    scene: str,
    config: Config,
) -> None:
    feature_dir.mkdir(parents=True, exist_ok=True)
    is_rotate = getattr(config, "rotate", False)

    with (
        h5py.File(f"{feature_dir}/keypoints_{model_name}.h5", mode="w") as f_keypoints,
        h5py.File(f"{feature_dir}/descriptors_{model_name}.h5", mode="w") as f_descriptors,
    ):
        for path in tqdm(image_paths, desc=f"Detecting keypoints / {model_name}"):
            key = path.name
            rot = 0
            with torch.inference_mode():
                image, ori_shape = load_resize_image(model_name, path, 1024, config)

                detector = get_detector_by_scene(model_name, scene=scene, config=config)

                if is_rotate and ("air-to-ground" in config.cat2scenes_dict) and (scene in config.cat2scenes_dict["air-to-ground"]):
                    outputs = apply_rotate(model_name, path, image, detector, config)
                else:
                    outputs = detector(image)

                keypoints = outputs[0][..., 2][0]
                scores = outputs[1][0]  # noqa
                descriptions = outputs[2][0]

                keypoints = adjust_keypoints_scale(image, ori_shape, keypoints, rot)
                kpts = keypoints.squeeze().detach().cpu().numpy()
                descs = descriptions.squeeze().detach().cpu().numpy()
                f_keypoints[key] = kpts
                f_descriptors[key] = descs


def get_matcher(
    model_name: str,
    config: Config,
) -> KF.DescriptorMatcher:
    if model_name in ["dedodeg", "disk", "sift"]:
        matcher_name = "LighGlue"
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

    return matcher_name, matcher


def match_with_kornia_common(
    model_name,
    image_paths,
    index_pairs,
    feature_dir,
    config: Config,
):
    matcher_name, matcher = get_matcher(model_name, config)
    cnt_pairs = 0

    with (
        h5py.File(f"{feature_dir}/keypoints_{model_name}.h5", mode="r") as f_keypoints,
        h5py.File(f"{feature_dir}/descriptors_{model_name}.h5", mode="r") as f_descriptors,
        h5py.File(f"{feature_dir}/matches_{model_name}.h5", mode="w") as f_match,
    ):
        for idx1, idx2 in tqdm(index_pairs, desc=f"Matching keypoing / {matcher_name}"):
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
                    print(f"{key1}-{key2}: {n_matches} matches @ {cnt_pairs}th pair({model_name}+kornia)")
            else:
                if config.matching_config["verbose"]:
                    print(f"{key1}-{key2}: {n_matches} matches --> skipped")


def feature_sift(
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

    match_with_kornia_common(
        model_name,
        image_paths,
        index_pairs,
        feature_dir,
        config,
    )
    gc.collect()

    print(f"Features matched ({model_name}+kornia)")
