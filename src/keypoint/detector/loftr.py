import gc
from tqdm import tqdm
import cv2
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import kornia as K
from kornia.feature import LoFTR
import h5py
from src.dataclass import Config


class LoFTRDataset(Dataset):
    def __init__(
        self,
        fnames1: list[Path],
        fnames2: list[Path],
        idxs1: list[int],
        idxs2: list[int],
        resize_small_edge_to: int,
        device: torch.device,
    ):
        self.fnames1 = fnames1
        self.fnames2 = fnames2
        self.keys1 = [fname.name for fname in fnames1]
        self.keys2 = [fname.name for fname in fnames2]
        self.idxs1 = idxs1
        self.idxs2 = idxs2
        self.resize_small_edge_to = resize_small_edge_to
        self.device = device
        self.round_unit = 16

    def __len__(self):
        return len(self.fnames1)

    def load_torch_image(
        self,
        fname: Path,
        device: torch.device,
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        img = cv2.imread(str(fname))
        original_shape = img.shape
        ratio = self.resize_small_edge_to / min([img.shape[0], img.shape[1]])
        w = int(img.shape[1] * ratio)  # int( (img.shape[1] * ratio) // self.round_unit * self.round_unit )
        h = int(img.shape[0] * ratio)  # int( (img.shape[0] * ratio) // self.round_unit * self.round_unit )
        img_resized = cv2.resize(img, (w, h))
        img_resized = K.image_to_tensor(img_resized, False).float() / 255.0
        img_resized = K.color.bgr_to_rgb(img_resized)
        img_resized = K.color.rgb_to_grayscale(img_resized)
        return img_resized.to(device), original_shape

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
        image1, ori_shape_1 = self.load_torch_image(fname1, self.device)
        image2, ori_shape_2 = self.load_torch_image(fname2, self.device)

        return image1, image2, key1, key2, idx1, idx2, ori_shape_1, ori_shape_2


def feature_loftr(
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

    matcher = LoFTR()
    # matcher.load_state_dict(torch.load("/kaggle/EfficientLoFTR/eloftr_outdoor.ckpt")["state_dict"])
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
        config.device,
    )

    cnt_pairs = 0

    with h5py.File(f"{feature_dir}/matches_loftr.h5", mode="w") as f_match:
        for data in tqdm(dataset, desc="Matching keypoints / LoFTR"):
            image1, image2, key1, key2, idx1, idx2, ori_shape_1, ori_shape_2 = data
            fname1, fname2 = image_paths[idx1], image_paths[idx2]

            with torch.no_grad():
                correspondences = matcher(
                    {
                        "image0": image1.to(config.device),
                        "image1": image2.to(config.device),
                    }
                )
                mkpts1 = correspondences["keypoints0"].cpu().numpy()  # shape: (n, 2)
                mkpts2 = correspondences["keypoints1"].cpu().numpy()  # shape: (n, 2)
                mconf = correspondences["confidence"].cpu().numpy()  # shape: (n,)

            mkpts1[:, 0] *= float(ori_shape_1[1]) / float(image1.shape[3])
            mkpts1[:, 1] *= float(ori_shape_1[0]) / float(image1.shape[2])

            mkpts2[:, 0] *= float(ori_shape_2[1]) / float(image2.shape[3])
            mkpts2[:, 1] *= float(ori_shape_2[0]) / float(image2.shape[2])

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
    gc.collect()
