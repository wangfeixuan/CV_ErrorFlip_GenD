from __future__ import annotations

from pathlib import Path

import numpy as np


def _load(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_mixed_features(
    blur_path: str | Path,
    jpeg_path: str | Path,
    resize_path: str | Path,
) -> dict:
    """Load three aligned caches and use the blur cache's clean features."""
    paths = {
        "blur": Path(blur_path).expanduser().resolve(),
        "jpeg": Path(jpeg_path).expanduser().resolve(),
        "resize": Path(resize_path).expanduser().resolve(),
    }
    caches = {name: _load(path) for name, path in paths.items()}
    canonical = caches["blur"]
    required = {
        "features_original",
        "features_blur",
        "base_logits",
        "labels",
        "label_index",
        "video_ids",
        "group_ids",
        "head_weight",
        "head_bias",
    }
    for name, cache in caches.items():
        missing = required - set(cache)
        if missing:
            raise ValueError(f"{name} cache is missing {sorted(missing)}")
        for key in ("labels", "label_index", "video_ids", "group_ids"):
            if not np.array_equal(cache[key], canonical[key]):
                raise ValueError(f"{name} cache is not aligned on {key}")
        max_clean_difference = float(
            np.max(np.abs(cache["features_original"] - canonical["features_original"]))
        )
        if max_clean_difference > 1e-4:
            raise ValueError(
                f"{name} clean features differ from canonical blur cache by "
                f"{max_clean_difference:.6f}"
            )
    return {
        "clean": canonical["features_original"],
        "degradations": {
            "blur": canonical["features_blur"],
            "jpeg": caches["jpeg"]["features_blur"],
            "resize": caches["resize"]["features_blur"],
        },
        "labels": canonical["labels"],
        "label_index": canonical["label_index"],
        "video_ids": canonical["video_ids"],
        "group_ids": canonical["group_ids"],
        "head_weight": canonical["head_weight"],
        "head_bias": canonical["head_bias"],
        "base_logits": canonical["base_logits"],
        "feature_paths": {name: str(path) for name, path in paths.items()},
        "clean_max_difference": max(
            float(
                np.max(
                    np.abs(
                        cache["features_original"]
                        - canonical["features_original"]
                    )
                )
            )
            for cache in caches.values()
        ),
    }
