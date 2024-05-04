import torch
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from lightglue import ALIKED
from src.utils import load_torch_image, preparation
from src.dataclass import Config
from src.utils.submission import parse_train_labels


def keypoint_viz_save(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    save_path: Path,
):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    dpi = 100
    b, c, h, w = image.shape

    image = image[0, ...].permute(1, 2, 0).cpu()
    keypoints = keypoints.cpu()

    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    plt.imshow(image)
    plt.scatter(keypoints[0, :, 0], keypoints[0, :, 1], s=0.5, c="red")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=100)


def viz_ALIKED(
    path_dict: dict[str, Path | list[Path]],
    scene: str,
    config: Config,
) -> None:
    aliked_config = config.aliked_config
    image_paths = path_dict["image_paths"]

    detector = (
        ALIKED(
            max_num_keypoints=aliked_config["max_num_keypoints"],
            detection_threshold=aliked_config["detection_threshold"],
            resize=aliked_config["resize_to"],
        )
        .eval()
        .to(config.device, torch.float32)
    ).extract

    for path in tqdm(image_paths, desc="Detecting keypoints / ALIKED"):
        save_path = config.keypoint_viz_dir / scene / path.name
        if not path.exists() or save_path.exists():
            continue
        with torch.inference_mode():
            image = load_torch_image(path, device=config.device).to(torch.float32)
            features = detector(image)
            keypoint_viz_save(image, features["keypoints"], save_path)


def keypoint_viz(config: Config) -> None:
    config.keypoint_viz_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    data_dict = parse_train_labels(config.base_path, config)
    datasets = sorted(data_dict, key=lambda x: len(data_dict[x][x]))

    for dataset in datasets:
        if dataset not in results:
            results[dataset] = {}
        for scene in data_dict[dataset]:
            print(scene)
            target_dict = {"data_dict": data_dict, "dataset": dataset, "scene": scene}
            path_dict, results = preparation(target_dict, results, config)
            if "ALIKED" in config.detector:
                viz_ALIKED(path_dict, scene, config)
            else:
                raise NotImplementedError
