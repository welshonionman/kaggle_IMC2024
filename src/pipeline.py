from pathlib import Path
import gc
import kornia as K
import torch
import pycolmap
import sys
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures

from src.pair_match import get_image_pairs
from src.keypoint import detect_keypoints, keypoint_distances
from src.dataclass import Config
from src.utils.submission import parse_sample_submission, create_submission, parse_train_labels
from src.utils import import_into_colmap
from src.utils.evaluate import evaluate
from src.reconstruction import find_optimal_reconstruction, parse_reconstructed_object

pycolmap.logging.minloglevel = 3


def preparation(
    target_dict: dict,
    results: dict,
    config: Config,
) -> tuple[dict, dict]:
    data_dict = target_dict["data_dict"]
    dataset = target_dict["dataset"]
    scene = target_dict["scene"]

    images_dir = data_dict[dataset][scene][0].parent
    image_paths = data_dict[dataset][scene][:30]
    results[dataset][scene] = {}

    feature_dir = config.feature_dir / f"{dataset}_{scene}"
    feature_dir.mkdir(parents=True, exist_ok=True)

    database_path = feature_dir / "colmap.db"
    if database_path.exists():
        database_path.unlink()

    path_dict = {"images_dir": images_dir, "image_paths": image_paths, "feature_dir": feature_dir, "database_path": database_path}
    return path_dict, results


def gpu_process(
    path_dict: dict,
    config: Config,
    device: torch.device,
) -> None:
    image_paths = path_dict["image_paths"]
    feature_dir = path_dict["feature_dir"]

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
    results: dict,
    prev_target_dict: dict,
    path_dict: dict,
    category: str,
    config: Config,
) -> dict:
    images_dir = path_dict["images_dir"]
    feature_dir = path_dict["feature_dir"]
    database_path = path_dict["database_path"]

    data_dict = prev_target_dict["data_dict"]
    dataset = prev_target_dict["dataset"]
    scene = prev_target_dict["scene"]

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
    results = parse_reconstructed_object(results, dataset, scene, maps, best_idx, category, config.base_path)
    print(f"\n登録済み: {dataset} / {scene} -> {len(results[dataset][scene])} / {len(data_dict[dataset][scene])}")

    create_submission(results, data_dict, config.base_path)
    gc.collect()


def run_from_config(config: Config) -> None:
    device = K.utils.get_cuda_device_if_available(0)
    results = {}
    is_kaggle_notebook = "kaggle_web_client" in sys.modules
    is_remain_cpu_process = False

    if is_kaggle_notebook:
        category = "test"
        data_dict = parse_sample_submission(config.base_path)
    else:
        category = "train"
        data_dict = parse_train_labels(config.base_path)

    datasets = sorted(list(data_dict.keys()))

    for dataset in datasets:
        if (not is_kaggle_notebook) and (dataset not in config.target_scene):
            continue

        if dataset not in results:
            results[dataset] = {}

        for scene in data_dict[dataset]:
            print(f"\n****** {dataset} ******")
            target_dict = {"data_dict": data_dict, "dataset": dataset, "scene": scene}
            path_dict, results = preparation(target_dict, results, config)

            if not is_remain_cpu_process:
                gpu_process(path_dict, config, device)
                is_remain_cpu_process = True
                prev_paths_dict = path_dict
                prev_target_dict = target_dict

            else:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future1 = executor.submit(gpu_process, path_dict, config, device)
                    future2 = executor.submit(cpu_process, results, prev_target_dict, prev_paths_dict, category, config)

                    future_list = [future1, future2]
                    finished, pending = futures.wait(future_list, return_when=futures.ALL_COMPLETED)
                    is_remain_cpu_process = True

                prev_paths_dict = path_dict
                prev_target_dict = target_dict

    if is_remain_cpu_process:
        cpu_process(results, prev_target_dict, prev_paths_dict, category, config)

    if not is_kaggle_notebook:
        print()
        evaluate(config)
