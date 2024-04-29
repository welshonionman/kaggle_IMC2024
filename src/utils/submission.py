from pathlib import Path
import numpy as np
import pandas as pd
from src.utils import arr_to_str
from src.dataclass import Config


def parse_train_labels(base_path: Path, config: Config):
    train_labels = pd.read_csv(config.gt_csv_path)


    data_dict = {}
    for i, row in enumerate(train_labels.itertuples()):
        image_path = row.image_path
        dataset = row.dataset
        scene = row.scene
        if dataset not in data_dict:
            data_dict[dataset] = {}
        if scene not in data_dict[dataset]:
            data_dict[dataset][scene] = []
        data_dict[dataset][scene].append(Path(base_path / image_path))

    for dataset in sorted(data_dict):
        for scene in sorted(data_dict[dataset]):
            print(
                f"{dataset} / {scene} -> {len(data_dict[dataset][scene])} images",
                file=open(config.log_path, "a"),
            )
    return data_dict


def parse_sample_submission(base_path: Path, config: Config) -> dict[dict[str, list[Path]]]:
    """Construct a dict describing the test data as

    {"dataset": {"scene": [<image paths>]}}
    """
    data_dict = {}
    with open(base_path / "sample_submission.csv", "r") as csv_f:
        for i, row in enumerate(csv_f):
            # Skip header
            if i == 0:
                print("header:", row)

            if row and i > 0:
                image_path, dataset, scene, _, _ = row.strip().split(",")
                if dataset not in data_dict:
                    data_dict[dataset] = {}
                if scene not in data_dict[dataset]:
                    data_dict[dataset][scene] = []
                data_dict[dataset][scene].append(Path(base_path / image_path))

    for dataset in sorted(data_dict):
        for scene in sorted(data_dict[dataset]):
            print(
                f"{dataset} / {scene} -> {len(data_dict[dataset][scene])} images",
                file=open(config.log_path, "a"),
            )

    return data_dict


def create_submission(
    results: dict,
    data_dict: dict[dict[str, list[Path]]],
    base_path: Path,
) -> None:
    """Prepares a submission file."""

    with open("/kaggle/working/submission.csv", "w") as f:
        f.write("image_path,dataset,scene,rotation_matrix,translation_vector\n")

        for dataset in data_dict:
            # Only write results for datasets with images that have results
            if dataset in results:
                res = results[dataset]
            else:
                res = {}

            # Same for scenes
            for scene in data_dict[dataset]:
                if scene in res:
                    scene_res = res[scene]
                else:
                    scene_res = {"R": {}, "t": {}}

                # Write the row with rotation and translation matrices
                for image in data_dict[dataset][scene]:
                    if image in scene_res:
                        # print(image)
                        R = scene_res[image]["R"].reshape(-1)
                        T = scene_res[image]["t"].reshape(-1)
                    else:
                        R = np.eye(3).reshape(-1)
                        T = np.zeros((3))
                    image_path = str(image.relative_to(base_path))
                    f.write(f"{image_path},{dataset},{scene},{arr_to_str(R)},{arr_to_str(T)}\n")
