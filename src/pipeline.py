from time import sleep
import gc
import numpy as np
import kornia as K
import pycolmap
from copy import deepcopy
from IPython.display import clear_output
from src.pair_match import get_image_pairs
from src.keypoint import detect_keypoints, keypoint_distances
from src.dataclass import Config
from src.utils.submission import parse_sample_submission, create_submission
from src.utils import import_into_colmap


def run_from_config(config: Config) -> None:
    # def run_from_config(config) -> None:
    device = K.utils.get_cuda_device_if_available(0)
    results = {}

    data_dict = parse_sample_submission(config.base_path)
    datasets = list(data_dict.keys())

    for dataset in datasets:
        if dataset not in results:
            results[dataset] = {}

        for scene in data_dict[dataset]:
            images_dir = data_dict[dataset][scene][0].parent
            results[dataset][scene] = {}
            image_paths = data_dict[dataset][scene]
            print(f"Got {len(image_paths)} images")

            try:
                feature_dir = config.feature_dir / f"{dataset}_{scene}"
                feature_dir.mkdir(parents=True, exist_ok=True)
                database_path = feature_dir / "colmap.db"
                if database_path.exists():
                    database_path.unlink()

                # 1. Get the pairs of images that are somewhat similar
                index_pairs = get_image_pairs(
                    image_paths,
                    **config.pair_matching_args,
                    device=config.device,
                )
                gc.collect()

                # 2. Detect keypoints of all images
                detect_keypoints(
                    image_paths,
                    feature_dir,
                    **config.keypoint_detection_args,
                    device=device,
                )
                gc.collect()

                # 3. Match  keypoints of pairs of similar images
                keypoint_distances(
                    image_paths,
                    index_pairs,
                    feature_dir,
                    **config.keypoint_distances_args,
                    device=device,
                )
                gc.collect()

                sleep(1)

                # 4.1. Import keypoint distances of matches into colmap for RANSAC
                import_into_colmap(
                    images_dir,
                    feature_dir,
                    database_path,
                )

                output_path = feature_dir / "colmap_rec_aliked"
                output_path.mkdir(parents=True, exist_ok=True)

                # 4.2. Compute RANSAC (detect match outliers)
                # By doing it exhaustively we guarantee we will find the best possible configuration
                pycolmap.match_exhaustive(database_path)

                mapper_options = pycolmap.IncrementalPipelineOptions(**config.colmap_mapper_options)

                # 5.1 Incrementally start reconstructing the scene (sparse reconstruction)
                # The process starts from a random pair of images and is incrementally extended by
                # registering new images and triangulating new points.
                maps = pycolmap.incremental_mapping(
                    database_path=database_path,
                    image_path=images_dir,
                    output_path=output_path,
                    options=mapper_options,
                )

                print(maps)
                clear_output(wait=False)

                # 5.2. Look for the best reconstruction: The incremental mapping offered by
                # pycolmap attempts to reconstruct multiple models, we must pick the best one
                images_registered = 0
                best_idx = None

                print("Looking for the best reconstruction")

                if isinstance(maps, dict):
                    for idx1, rec in maps.items():
                        print(idx1, rec.summary())
                        try:
                            if len(rec.images) > images_registered:
                                images_registered = len(rec.images)
                                best_idx = idx1
                        except Exception:
                            continue

                # Parse the reconstruction object to get the rotation matrix and translation vector
                # obtained for each image in the reconstruction
                if best_idx is not None:
                    for k, im in maps[best_idx].images.items():
                        key = config.base_path / "test" / scene / "images" / im.name
                        results[dataset][scene][key] = {}
                        results[dataset][scene][key]["R"] = deepcopy(im.cam_from_world.rotation.matrix())
                        results[dataset][scene][key]["t"] = deepcopy(np.array(im.cam_from_world.translation))

                print(f"Registered: {dataset} / {scene} -> {len(results[dataset][scene])} images")
                print(f"Total: {dataset} / {scene} -> {len(data_dict[dataset][scene])} images")
                create_submission(results, data_dict, config.base_path)
                gc.collect()

            except Exception as e:
                print(e)
