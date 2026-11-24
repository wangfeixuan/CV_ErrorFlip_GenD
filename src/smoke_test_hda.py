from __future__ import annotations

import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np

from .train_hda import train


def build_feature_cache(path: Path, videos: int = 60, dimension: int = 32) -> None:
    rng = np.random.default_rng(2026)
    labels = np.asarray(["real", "fake"] * (videos // 2))
    label_index = (labels == "fake").astype(np.int8)
    video_ids = np.asarray([f"Celeb-real__id{i:03d}" for i in range(videos)])
    group_ids = video_ids.copy()
    clean = rng.normal(size=(videos, 4, dimension)).astype(np.float32)
    clean += (label_index[:, None, None] * 2 - 1) * 0.7
    degraded = (clean * 0.55 + rng.normal(scale=0.65, size=clean.shape)).astype(
        np.float32
    )
    head_weight = rng.normal(scale=0.06, size=(2, dimension)).astype(np.float32)
    head_weight[1] -= head_weight[0]
    head_bias = np.asarray([0.05, -0.05], dtype=np.float32)
    np.savez_compressed(
        path,
        features_original=clean,
        features_blur=degraded,
        base_logits=clean @ head_weight.T + head_bias,
        base_logits_blur=degraded @ head_weight.T + head_bias,
        labels=labels,
        label_index=label_index,
        video_ids=video_ids,
        group_ids=group_ids,
        head_weight=head_weight,
        head_bias=head_bias,
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        paths = {
            name: root / f"{name}.npz"
            for name in ("blur", "jpeg", "resize")
        }
        for path in paths.values():
            build_feature_cache(path)
        checkpoint = train(
            Namespace(
                blur_features=str(paths["blur"]),
                jpeg_features=str(paths["jpeg"]),
                resize_features=str(paths["resize"]),
                weighting="learned",
                seed=2026,
                output_dir=str(root / "run"),
                device="cpu",
                config=str(
                    Path(__file__).resolve().parents[1] / "configs" / "hda.yaml"
                ),
                epochs=1,
                patience=1,
                batch_size=16,
                learning_rate=0.0005,
                bottleneck=8,
                overwrite=False,
            )
        )
        assert checkpoint.is_file()
        assert (root / "run" / "eval_seen_blur_jpeg.json").is_file()
        assert (root / "run" / "eval_unseen_blur_resize.json").is_file()
        assert (root / "run" / "eval_unseen_jpeg_resize.json").is_file()
        print(checkpoint)


if __name__ == "__main__":
    main()
