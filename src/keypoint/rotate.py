from pathlib import Path
import numpy as np
import torch
import albumentations as albu
from PIL import Image
from check_orientation.pre_trained_models import create_model


def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    return torch.from_numpy(image)


def detect_rot(img_path: Path):
    img = np.array(Image.open(img_path))
    model = create_model("swsl_resnext50_32x4d")
    model.eval()
    transform = albu.Compose([albu.Resize(height=224, width=224), albu.Normalize(p=1)], p=1)
    img_ = transform(image=img)["image"]
    pred = model(tensor_from_rgb_image(img_).unsqueeze(0))

    correct_rot = np.argmax(pred.detach().numpy())
    mapping = {0: 0, 1: 3, 2: 2, 3: 1}  # rot90のパラメータとの対応
    # m = ["orgin: ", "1/2pi: ", "pi: ", "3/2pi: "]
    # print("pred : ", pred)
    # print("rotation : ", m[correct_rot])

    return mapping[correct_rot]


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
