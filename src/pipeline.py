from time import sleep
from pathlib import Path
import gc
import numpy as np
import kornia as K
import torch
import pycolmap
import sys
import pandas as pd

from src.pair_match import get_image_pairs
from src.keypoint import detect_keypoints, keypoint_distances
from src.dataclass import Config
from src.utils.submission import parse_sample_submission, create_submission, parse_train_labels
from src.utils import import_into_colmap
from src.utils.metrics import score
from src.reconstruction import find_optimal_reconstruction, parse_reconstructed_object

pycolmap.logging.minloglevel = 3


def preparation(
    data_dict: dict[dict[str, list[Path]]],
    dataset: str,
    scene: str,
    results: dict,
    config: Config,
) -> tuple[dict, Path, list[Path], Path, Path]:
    images_dir = data_dict[dataset][scene][0].parent
    image_paths = data_dict[dataset][scene][:4]
    results[dataset][scene] = {}

    feature_dir = config.feature_dir / f"{dataset}_{scene}"
    feature_dir.mkdir(parents=True, exist_ok=True)

    database_path = feature_dir / "colmap.db"
    if database_path.exists():
        database_path.unlink()

    return results, images_dir, image_paths, feature_dir, database_path


def gpu_process(
    image_paths: list[Path],
    feature_dir: Path,
    config: Config,
    device: torch.device,
) -> None:
    # 1. 似ていると思われる画像のペアを取得する
    distances, index_pairs = get_image_pairs(image_paths, **config.pair_matching_args, device=config.device)
    gc.collect()

    # 2. すべての画像の特徴点を検出する
    detect_keypoints(image_paths, feature_dir, **config.keypoint_detection_args, device=device)
    gc.collect()

    # 3. 似ている画像のペアの特徴点をマッチングする
    keypoint_distances(image_paths, index_pairs, feature_dir, **config.keypoint_distances_args, device=device)
    gc.collect()
    return


def cpu_process(
    images_dir: Path,
    feature_dir: Path,
    database_path: Path,
    config: Config,
    results: dict,
    dataset: str,
    scene: str,
    train_test: str,
) -> dict:
    # 4.1. マッチングした特徴点の距離をcolmapにインポートする
    import_into_colmap(images_dir, feature_dir, database_path)

    output_path = feature_dir / "colmap_rec_aliked"
    output_path.mkdir(parents=True, exist_ok=True)

    # 4.2. RANSACを実行する（マッチングの外れ値を検出する）
    pycolmap.match_exhaustive(database_path, sift_options={"num_threads": 1})

    mapper_options = pycolmap.IncrementalPipelineOptions(**config.colmap_mapper_options)

    # 5.1 シーンの再構築を開始する（スパースな再構築）
    maps = pycolmap.incremental_mapping(database_path=database_path, image_path=images_dir, output_path=output_path, options=mapper_options)

    # 5.2. 最適な再構築を探す：pycolmapが提供するインクリメンタルマッピングでは、複数のモデルを再構築しようとしますが、最良のものを選ぶ必要があります
    best_idx = find_optimal_reconstruction(maps)

    # 再構築オブジェクトを解析して、再構築における各画像の回転行列と並進ベクトルを取得する
    results = parse_reconstructed_object(results, dataset, scene, maps, best_idx, train_test, config.base_path)
    return results


def run_from_config(config: Config) -> None:
    device = K.utils.get_cuda_device_if_available(0)
    results = {}

    is_kaggle_notebook = "kaggle_web_client" in sys.modules
    if is_kaggle_notebook:
        train_test = "test"
        data_dict = parse_sample_submission(config.base_path)
    else:
        train_test = "train"
        data_dict = parse_train_labels(config.base_path)

    datasets = sorted(list(data_dict.keys()))

    for dataset in datasets:
        if (not is_kaggle_notebook) and (dataset not in config.target_scene):
            continue
        if dataset not in results:
            results[dataset] = {}
        for scene in data_dict[dataset]:
            print(f"\n****** {dataset} ******")
            results, images_dir, image_paths, feature_dir, database_path = preparation(data_dict, dataset, scene, results, config)

            gpu_process(image_paths, feature_dir, config, device)
            sleep(1)

            results = cpu_process(images_dir, feature_dir, database_path, config, results, dataset, scene, train_test)
            print(f"\n登録済み: {dataset} / {scene} -> {len(results[dataset][scene])} / {len(data_dict[dataset][scene])}")

            create_submission(results, data_dict, config.base_path)
            gc.collect()

    if not is_kaggle_notebook:
        print()
        gt_csv = "/kaggle/input/image-matching-challenge-2024/train/train_labels.csv"
        user_csv = "/kaggle/working/submission.csv"
        gt_df = pd.read_csv(gt_csv).rename(columns={"image_name": "image_path"})
        sub_df = pd.read_csv(user_csv)
        sub_df["image_path"] = sub_df["image_path"].str.split("/").str[-1]
        score(gt_df, sub_df, config.target_scene)
