import torch
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
import itertools
from typing import Any
import numpy as np
from src.utils import load_torch_image


def embed_images(
    image_paths: list[Path],  # 処理する画像ファイルのパスリスト
    model_name: str,  # 画像埋め込みに使用するモデル名
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """画像の埋め込みベクトルを計算します。

    返り値は shape が [len(filenames), output_dim] のテンソルです。
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)

    embeddings = []

    for i, path in enumerate(tqdm(image_paths, desc="Global descriptors")):
        image = load_torch_image(path, device=device)

        with torch.inference_mode():
            inputs = processor(images=image, return_tensors="pt", do_rescale=False).to(device)
            outputs = model(**inputs)  # last_hidden_state and pooled

            # 最後の隠れ層の全てのステートに対して最大プーリングを行いますが、fastprogress は除外します
            # 距離をより良い方法で計算するために正規化を行います
            embedding = F.normalize(outputs.last_hidden_state[:, 1:].max(dim=1)[0], dim=-1, p=2)

        embeddings.append(embedding.detach().cpu())
    return torch.cat(embeddings, dim=0)


def get_pairs_exhaustive(lst: list[Any]) -> list[tuple[int, int]]:
    """リストのすべての可能なインデックスペアを取得します"""
    return list(itertools.combinations(range(len(lst)), 2))


def get_image_pairs(
    path_dict: dict[str, Path | list[Path]],
    model_name: str,
    similarity_threshold: float = 0.6,
    tolerance: int = 1000,
    min_matches: int = 20,
    exhaustive_if_less: int = 20,
    p: float = 2.0,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """類似した画像のペアを取得します"""
    image_paths = path_dict["image_paths"]
    # if len(image_paths) <= exhaustive_if_less:
    #     return get_pairs_exhaustive(image_paths)

    matches = []

    # 画像を埋め込み、フィルタリングのための距離を計算する
    embeddings = embed_images(image_paths, model_name, device)  # shape: [len(filenames), output_dim]
    distances = torch.cdist(embeddings, embeddings, p=p)  # shape: [len(filenames), len(filenames)]

    # 類似度閾値を超えるペアを削除する（十分な数がある場合）
    mask = distances <= similarity_threshold
    image_indices = np.arange(len(image_paths))

    for current_image_index in range(len(image_paths)):
        mask_row = mask[current_image_index]
        indices_to_match = image_indices[mask_row]

        # 閾値以下のマッチが十分にない場合、最も類似したものを選ぶ
        if len(indices_to_match) < min_matches:
            indices_to_match = np.argsort(distances[current_image_index])[:min_matches]

        for other_image_index in indices_to_match:
            # 自分自身とのマッチングをスキップする
            if other_image_index == current_image_index:
                continue

            # 特定の距離許容値以下である必要がある
            # 十分なマッチがない画像については、最も類似したものを選んだため、
            # すべての画像が分析対象の画像と非常に異なる可能性がある
            if distances[current_image_index, other_image_index] < tolerance:
                # 冗長性を避けるために、ソートされた形式でペアを追加する
                matches.append(tuple(sorted((current_image_index, other_image_index.item()))))

    index_pairs = sorted(list(set(matches)))

    return distances, index_pairs
