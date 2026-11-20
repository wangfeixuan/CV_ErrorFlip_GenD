from __future__ import annotations

import json
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np

from .data import assign_group_splits
from .evaluate import evaluate
from .train import train
from .train_mixed import train as train_mixed


def build_synthetic_features(path: Path, videos: int = 60, dimension: int = 32) -> None:
    rng = np.random.default_rng(2026)
    labels = np.asarray(["real", "fake"] * (videos // 2))
    label_index = (labels == "fake").astype(np.int8)
    video_ids = np.asarray([f"{label}__{index:04d}" for index, label in enumerate(labels)])
    group_ids = video_ids.copy()
    clean = rng.normal(size=(videos, 4, dimension)).astype(np.float32)
    signal = (label_index[:, None, None] * 2 - 1) * 0.8
    clean += signal
    blur = (clean * 0.55 + rng.normal(scale=0.9, size=clean.shape)).astype(
        np.float32
    )
    head_weight = rng.normal(
        scale=0.08, size=(2, dimension)
    ).astype(np.float32)
    head_weight[1] -= head_weight[0]
    head_bias = np.asarray([0.1, -0.1], dtype=np.float32)
    logits = clean @ head_weight.T + head_bias
    np.savez_compressed(
        path,
        features_original=clean,
        features_blur=blur,
        base_logits=logits,
        base_logits_blur=blur @ head_weight.T + head_bias,
        labels=labels,
        label_index=label_index,
        video_ids=video_ids,
        group_ids=group_ids,
        head_weight=head_weight,
        head_bias=head_bias,
    )
    records = assign_group_splits(
        [
            {
                "video_id": str(video_id),
                "label": str(label),
                "label_index": int(index),
                "group_id": str(group_id),
            }
            for video_id, label, index, group_id in zip(
                video_ids, labels, label_index, group_ids
            )
        ],
        {"train": 0.7, "validation": 0.15, "test": 0.15},
        2026,
    )
    from .data import write_manifest

    write_manifest(records, path.with_suffix(".manifest.csv"))


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        features = root / "features.npz"
        build_synthetic_features(features)
        for method in (
            "clean_only",
            "uniform",
            "consistency",
            "jpeg_only",
            "resize_only",
            "uniform_random",
            "flip_hard",
            "flip_weighted",
        ):
            run_dir = root / method
            checkpoint = train(
                Namespace(
                    features=str(features),
                    method=method,
                    seed=2026,
                    output_dir=str(run_dir),
                    device="cpu",
                    config=str(
                        Path(__file__).resolve().parents[1]
                        / "configs"
                        / "base.yaml"
                    ),
                    epochs=2,
                    patience=2,
                    batch_size=16,
                    learning_rate=0.001,
                    bottleneck=8,
                    blurred_frames=2,
                    overwrite=False,
                )
            )
            result = evaluate(
                Namespace(
                    features=str(features),
                    checkpoint=str(checkpoint),
                    split="test",
                    output=str(run_dir / "eval.json"),
                    device="cpu",
                    threshold=0.5,
                )
            )
            assert result["videos"] > 0
            assert result["adapter_clean"]["auc"] is not None
            print(method, json.dumps(result["inversion"]))

        blur_features = root / "blur.npz"
        jpeg_features = root / "jpeg.npz"
        resize_features = root / "resize.npz"
        for path in (blur_features, jpeg_features, resize_features):
            build_synthetic_features(path)
        mixed_checkpoint = train_mixed(
            Namespace(
                blur_features=str(blur_features),
                jpeg_features=str(jpeg_features),
                resize_features=str(resize_features),
                type_strategy="random_one",
                seed=2026,
                output_dir=str(root / "mixed"),
                device="cpu",
                config=str(
                    Path(__file__).resolve().parents[1]
                    / "configs"
                    / "base.yaml"
                ),
                epochs=2,
                patience=2,
                batch_size=16,
                learning_rate=0.001,
                bottleneck=8,
                overwrite=False,
            )
        )
        mixed_result = evaluate(
            Namespace(
                features=str(blur_features),
                checkpoint=str(mixed_checkpoint),
                split="test",
                output=str(root / "mixed" / "eval.json"),
                device="cpu",
                threshold=0.5,
            )
        )
        print("mixed", json.dumps(mixed_result["inversion"]))
        mixed_full_checkpoint = train_mixed(
            Namespace(
                blur_features=str(blur_features),
                jpeg_features=str(jpeg_features),
                resize_features=str(resize_features),
                type_strategy="all_types",
                seed=2026,
                output_dir=str(root / "mixed_full"),
                device="cpu",
                config=str(
                    Path(__file__).resolve().parents[1]
                    / "configs"
                    / "base.yaml"
                ),
                epochs=1,
                patience=1,
                batch_size=16,
                learning_rate=0.001,
                bottleneck=8,
                overwrite=False,
            )
        )
        assert mixed_full_checkpoint.is_file()


if __name__ == "__main__":
    main()
