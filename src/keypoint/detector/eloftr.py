import gc
from tqdm import tqdm
import cv2
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import kornia as K
from copy import deepcopy
from ext_repos.EfficientLoFTR.src.loftr import LoFTR, full_default_cfg, reparameter
import h5py
from check_orientation.pre_trained_models import create_model
from src.dataclass import Config
from src.keypoint.rotate import detect_rot


class LoFTRDataset(Dataset):
    def __init__(
        self,
        fnames1: list[Path],
        fnames2: list[Path],
        idxs1: list[int],
        idxs2: list[int],
        resize_small_edge_to: int,
        scene: str,
        config: Config,
    ):
        self.fnames1 = fnames1
        self.fnames2 = fnames2
        self.keys1 = [fname.name for fname in fnames1]
        self.keys2 = [fname.name for fname in fnames2]
        self.idxs1 = idxs1
        self.idxs2 = idxs2
        self.resize_small_edge_to = resize_small_edge_to
        self.scene = scene
        self.config = config
        self.round_unit = 16

    def __len__(self):
        return len(self.fnames1)

    def fill_holes(self, image, fill_value=1):
        """
        二値画像中の一定の面積より小さい穴を埋める
        """
        if image.sum() == 0:
            return image
        result_image = image.copy()
        contours, hierarchies = cv2.findContours(result_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour, hierarchy in zip(contours, hierarchies[0]):
            cv2.drawContours(result_image, [contour], -1, fill_value, thickness=cv2.FILLED)
        return result_image

    def masked_image(self, path):
        img = cv2.imread(str(path))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 5)
        _, img_thre = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
        img_thre = img_thre // 255
        img_thre = cv2.dilate(img_thre, np.ones((5, 5)), iterations=30)
        img_thre = self.fill_holes(img_thre, 1)
        masked_img = img * img_thre[:, :, np.newaxis]
        return masked_img

    def load_torch_image(
        self,
        fname: Path,
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        if ("transparent" in self.config.cat2scenes_dict) and (self.scene in self.config.cat2scenes_dict["transparent"]):
            img = self.masked_image(str(fname))
        else:
            img = cv2.imread(str(fname))
        original_shape = img.shape
        ratio = self.resize_small_edge_to / min([img.shape[0], img.shape[1]])
        w = int(img.shape[1] * ratio)  # int( (img.shape[1] * ratio) // self.round_unit * self.round_unit )
        h = int(img.shape[0] * ratio)  # int( (img.shape[0] * ratio) // self.round_unit * self.round_unit )

        img_resized = cv2.resize(img, (w, h))
        img_resized = cv2.resize(
            img_resized, (img_resized.shape[1] // 32 * 32, img_resized.shape[0] // 32 * 32)
        )  # input size shuold be divisible by 32

        img_resized = K.image_to_tensor(img_resized, False).float() / 255.0
        img_resized = K.color.bgr_to_rgb(img_resized)
        img_resized = K.color.rgb_to_grayscale(img_resized)
        return img_resized.to(self.config.device), original_shape

    def __getitem__(
        self, idx: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        str,
        str,
        int,
        int,
        tuple[int, int, int],
        tuple[int, int, int],
    ]:
        key1 = self.keys1[idx]
        key2 = self.keys2[idx]
        idx1 = self.idxs1[idx]
        idx2 = self.idxs2[idx]
        fname1 = self.fnames1[idx]
        fname2 = self.fnames2[idx]
        image1, ori_shape_1 = self.load_torch_image(fname1)
        image2, ori_shape_2 = self.load_torch_image(fname2)

        return image1, image2, key1, key2, idx1, idx2, ori_shape_1, ori_shape_2


def rotate_images(image_paths: list[Path], config: Config) -> dict:
    rot_dict = {}
    check_orientation_model = create_model("swsl_resnext50_32x4d").to(config.device).eval()

    for image_path in tqdm(image_paths, desc="Detecting orientation"):
        rot_dict[image_path] = detect_rot(image_path)

    del check_orientation_model
    torch.cuda.empty_cache()
    return rot_dict


def adjust_keypoints_scale(image, ori_shape, mkpts, rot):
    if rot in [0, 2]:
        mkpts[:, 0] *= float(ori_shape[1]) / float(image.shape[3])
        mkpts[:, 1] *= float(ori_shape[0]) / float(image.shape[2])
    else:
        mkpts[:, 0] *= float(ori_shape[0]) / float(image.shape[3])
        mkpts[:, 1] *= float(ori_shape[1]) / float(image.shape[2])

    return mkpts


def feature_eloftr(
    path_dict: dict,
    index_pairs: list[tuple[int, int]],
    scene: str,
    config: Config,
    resize_small_edge_to=750,
    min_matches=15,
) -> None:
    image_paths = path_dict["image_paths"]
    feature_dir = path_dict["feature_dir"]

    feature_dir.mkdir(parents=True, exist_ok=True)

    _default_cfg = deepcopy(full_default_cfg)
    matcher = LoFTR(config=_default_cfg)
    matcher.load_state_dict(torch.load("/kaggle/ext_repos/EfficientLoFTR/eloftr_outdoor.ckpt")["state_dict"])
    matcher = reparameter(matcher)
    matcher = matcher.to(config.device).eval()

    fnames1, fnames2, idxs1, idxs2 = [], [], [], []
    for idx1, idx2 in tqdm(index_pairs):
        fname1, fname2 = image_paths[idx1], image_paths[idx2]
        fnames1.append(fname1)
        fnames2.append(fname2)
        idxs1.append(idx1)
        idxs2.append(idx2)

    dataset = LoFTRDataset(
        fnames1,
        fnames2,
        idxs1,
        idxs2,
        resize_small_edge_to,
        scene,
        config,
    )

    cnt_pairs = 0

    is_rotate = (
        getattr(config, "rotate", False) and ("air-to-ground" in config.cat2scenes_dict) and (scene in config.cat2scenes_dict["air-to-ground"])
    )

    with torch.inference_mode():
        if is_rotate:
            rot_dict = rotate_images(image_paths, config)

    with h5py.File(f"{feature_dir}/matches_eloftr.h5", mode="w") as f_match:
        for data in tqdm(dataset, desc="Matching keypoints / LoFTR"):
            image1, image2, key1, key2, idx1, idx2, ori_shape_1, ori_shape_2 = data
            fname1, fname2 = image_paths[idx1], image_paths[idx2]
            rot1, rot2 = 0, 0

            with torch.inference_mode():
                if is_rotate:
                    rot1 = rot_dict[image_paths[idx1]]
                    rot2 = rot_dict[image_paths[idx2]]
                    image1 = torch.rot90(torch.tensor(image1), rot1, [2, 3])
                    image2 = torch.rot90(torch.tensor(image2), rot2, [2, 3])

                batch = {
                    "image0": image1.to(config.device),
                    "image1": image2.to(config.device),
                }
                matcher(batch)

                mkpts1 = batch["mkpts0_f"].cpu().numpy()  # shape: (n, 2)
                mkpts2 = batch["mkpts1_f"].cpu().numpy()  # shape: (n, 2)
                mconf = batch["mconf"].cpu().numpy()  # shape: (n,)

            mkpts1 = adjust_keypoints_scale(image1, ori_shape_1, mkpts1, rot1)
            mkpts2 = adjust_keypoints_scale(image2, ori_shape_2, mkpts2, rot2)

            n_matches = mconf.shape[0]

            group = f_match.require_group(key1)
            if n_matches >= min_matches:
                group.create_dataset(key2, data=np.concatenate([mkpts1, mkpts2], axis=1).astype(np.float32))
                cnt_pairs += 1
                if config.matching_config["verbose"]:
                    print(f"{key1}-{key2}: {n_matches} matches @ {cnt_pairs}th pair(loftr)")
            else:
                if config.matching_config["verbose"]:
                    print(f"{key1}-{key2}: {n_matches} matches --> skipped")
    del dataset
    torch.cuda.empty_cache()
    gc.collect()
