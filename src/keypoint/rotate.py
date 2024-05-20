import cv2
from pathlib import Path
import numpy as np
import torch
import albumentations as albu
from check_orientation.pre_trained_models import create_model
from src.dataclass import Config
import kornia as K
import kornia.feature as KF


def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    return torch.from_numpy(image)


def detect_rot(img_path: Path):
    img = cv2.cvtColor(cv2.imread(str(img_path), -1), cv2.COLOR_BGR2RGB)
    device = K.utils.get_cuda_device_if_available(0)

    model = create_model("swsl_resnext50_32x4d").to(device)
    model.eval()

    transform = albu.Compose([albu.Resize(height=224, width=224), albu.Normalize(p=1)], p=1)
    img_ = transform(image=img)["image"]
    pred = model(tensor_from_rgb_image(img_).unsqueeze(0).to(device))

    correct_rot = np.argmax(pred.detach().cpu().numpy())
    mapping = {0: 0, 1: 3, 2: 2, 3: 1}  # rot90のパラメータとの対応
    rot_value = mapping[correct_rot]
    return rot_value


def rotate_kpts(kpts, im_shape, k):
    kpts = kpts.copy()
    width, height = im_shape

    rot_kpts = np.zeros_like(kpts)
    if k == 1:
        rot_kpts[:, 0], rot_kpts[:, 1] = height - kpts[:, 1], kpts[:, 0]
    elif k == 2:
        rot_kpts[:, 0], rot_kpts[:, 1] = width - kpts[:, 0], height - kpts[:, 1]
    elif k == 3:
        rot_kpts[:, 0], rot_kpts[:, 1] = kpts[:, 1], width - kpts[:, 0]
        # rot_kpts[:,0], rot_kpts[:,1] = kpts[:,1], kpts[:,0] + width

    elif k == 0:
        rot_kpts = kpts
    else:
        raise ValueError(f"Unknown rotation {k}")

    return rot_kpts


def apply_rotate(
    model_name: str,
    path: Path,
    image: torch.Tensor,
    detector: torch.nn.Module,
    config: Config,
) -> dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    correct_rot = detect_rot(path)
    image = torch.rot90(image, correct_rot, dims=(2, 3))

    if model_name in ["aliked", "superpoint", "doghardnet"]:
        features = detector(image)
        tmp_np = features["keypoints"][0].cpu().numpy()
        rot_kpts = rotate_kpts(tmp_np, (image.shape[3], image.shape[2]), correct_rot)
        keypoints_rot = torch.from_numpy(rot_kpts).unsqueeze(0).to(config.device)
        features["keypoints"] = keypoints_rot
        return features

    if model_name == "dedodeg":
        keypoints, scores, descriptions = detector(image)
        tmp_np = keypoints[0].cpu().numpy()
        rot_kpts = rotate_kpts(tmp_np, (image.shape[3], image.shape[2]), correct_rot)
        keypoints_rot = torch.from_numpy(rot_kpts).unsqueeze(0).to(config.device)
        return keypoints_rot, scores, descriptions

    if model_name == "disk":
        features = detector(image, n=4096, score_threshold=0.01)
        keypoints = features[0].keypoints
        tmp_np = keypoints.cpu().numpy()
        rot_kpts = rotate_kpts(tmp_np, (image.shape[3], image.shape[2]), correct_rot)
        keypoints_rot = torch.from_numpy(rot_kpts).unsqueeze(0).to(config.device)
        features[0].keypoints = keypoints_rot
        return features

    if model_name == "sift":
        features = detector(image)  # TODO: 回転に対応
        return features
