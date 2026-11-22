from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .adapter import GenDAdapter
from .gend_model import choose_device
from .masks import all_masks, compose_frame_probabilities, compose_masks
from .metrics import binary_metrics, finite_mean, roc_auc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "external"),
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--condition")
    return parser.parse_args()


def load_npz(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@torch.inference_mode()
def predictions_for_masks(
    model: GenDAdapter,
    clean: torch.Tensor,
    blurred: torch.Tensor,
    masks: torch.Tensor,
) -> np.ndarray:
    probabilities = []
    for mask in masks:
        composed = compose_masks(clean, blurred, mask[None])
        _, video_probability = model(composed[0])
        probabilities.append(video_probability.cpu().numpy())
    return np.stack(probabilities)


@torch.inference_mode()
def feature_restoration_metrics(
    model: GenDAdapter,
    clean: torch.Tensor,
    blurred: torch.Tensor,
    masks: torch.Tensor,
) -> dict:
    raw_cosine = torch.nn.functional.cosine_similarity(
        blurred, clean, dim=-1
    ).mean()
    adapted_cosines = []
    for mask in masks:
        composed = compose_masks(clean, blurred, mask[None])[0]
        _, _, adapted = model(composed, return_features=True)
        adapted_cosines.append(
            torch.nn.functional.cosine_similarity(
                adapted, clean, dim=-1
            ).mean()
        )
    adapted_cosine = torch.stack(adapted_cosines).mean()
    return {
        "raw_blur_cosine": float(raw_cosine),
        "adapted_masked_cosine": float(adapted_cosine),
        "cosine_improvement": float(adapted_cosine - raw_cosine),
    }


def condition_summary(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    metrics = [
        binary_metrics(labels, probabilities[index], threshold=threshold)
        for index in range(len(probabilities))
    ]
    predictions = probabilities >= threshold
    worst_correct = np.min(predictions == labels[None, :], axis=0)
    return {
        "masks": int(len(probabilities)),
        "mean_accuracy": finite_mean(metric["accuracy"] for metric in metrics),
        "mean_auc": finite_mean(metric["auc"] for metric in metrics),
        "mean_balanced_accuracy": finite_mean(
            metric["balanced_accuracy"] for metric in metrics
        ),
        "mean_f1": finite_mean(metric["f1"] for metric in metrics),
        "mean_false_positive_rate": finite_mean(
            metric["false_positive_rate"] for metric in metrics
        ),
        "mean_false_negative_rate": finite_mean(
            metric["false_negative_rate"] for metric in metrics
        ),
        "worst_mask_accuracy": float(worst_correct.mean()),
        "flip_rate": float(np.mean(np.any(predictions != predictions[0:1], axis=0))),
        "mean_position_variance": float(np.mean(np.var(probabilities, axis=0))),
        "per_mask": metrics,
    }


def evaluate(args: argparse.Namespace) -> dict:
    device = choose_device(args.device)
    data = load_npz(args.features)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if args.split == "external":
        selected_index = np.arange(len(data["video_ids"]), dtype=np.int64)
    else:
        split_by_video = checkpoint["split_by_video"]
        selected_index = np.asarray(
            [
                index
                for index, video_id in enumerate(data["video_ids"])
                if split_by_video[str(video_id)] == args.split
            ],
            dtype=np.int64,
        )
    if not len(selected_index):
        raise ValueError(f"No videos found for split={args.split}")

    model = GenDAdapter.from_pretrained_head(
        torch.from_numpy(checkpoint["head_weight"]),
        torch.from_numpy(checkpoint["head_bias"]),
        bottleneck=int(checkpoint["config"]["model"]["bottleneck"]),
        dropout=float(checkpoint["config"]["model"]["dropout"]),
        scale_init=float(checkpoint["config"]["model"]["scale_init"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    clean = torch.from_numpy(data["features_original"][selected_index]).to(device)
    blurred = torch.from_numpy(data["features_blur"][selected_index]).to(device)
    labels = data["label_index"][selected_index].astype(bool)
    video_ids = data["video_ids"][selected_index]
    original_logits = torch.from_numpy(data["base_logits"][selected_index])
    base_original_probability = (
        torch.softmax(original_logits, dim=-1)[..., 1].numpy()
    )
    original_blur_logits = torch.from_numpy(
        data["base_logits_blur"][selected_index]
    )
    base_blur_probability = (
        torch.softmax(original_blur_logits, dim=-1)[..., 1].numpy()
    )
    base_probabilities = base_original_probability.mean(axis=1)

    masks = all_masks(4, device=device)
    adapter_probabilities = predictions_for_masks(model, clean, blurred, masks)
    base_mask_probability = compose_frame_probabilities(
        torch.from_numpy(base_original_probability),
        torch.from_numpy(base_blur_probability),
        masks.cpu(),
    ).numpy()

    pair_mask_index = torch.where(
        masks.sum(dim=1) == 2
    )[0].cpu().numpy()
    base_pair = base_mask_probability[pair_mask_index]
    adapter_pair = adapter_probabilities[pair_mask_index]
    base_pair_correct = (base_pair >= args.threshold) == labels[None, :]
    adapter_pair_correct = (adapter_pair >= args.threshold) == labels[None, :]
    recovered_mask_events = int(np.sum(~base_pair_correct & adapter_pair_correct))
    harmed_mask_events = int(np.sum(base_pair_correct & ~adapter_pair_correct))
    base_error_mask_events = int(np.sum(~base_pair_correct))
    base_correct_events = int(np.sum(base_pair_correct))
    base_flipped_video = base_pair_correct.mean(axis=0) < 1
    any_recovered_video = np.any(~base_pair_correct & adapter_pair_correct, axis=0)

    clean_metrics = binary_metrics(labels, base_probabilities, args.threshold)
    result = {
        "split": args.split,
        "condition": getattr(args, "condition", None) or Path(args.features).stem,
        "feature_file": str(Path(args.features).resolve()),
        "method": checkpoint["method"],
        "seed": int(checkpoint["seed"]),
        "training_blurred_frames": int(checkpoint.get("blurred_frames", 2)),
        "videos": int(len(selected_index)),
        "threshold": args.threshold,
        "base_clean": clean_metrics,
        "adapter_clean": binary_metrics(
            labels,
            adapter_probabilities[0],
            args.threshold,
        ),
        "base_all_masks": condition_summary(
            labels, base_mask_probability, args.threshold
        ),
        "adapter_all_masks": condition_summary(
            labels, adapter_probabilities, args.threshold
        ),
        "base_two_frame_masks": condition_summary(
            labels, base_pair, args.threshold
        ),
        "adapter_two_frame_masks": condition_summary(
            labels, adapter_pair, args.threshold
        ),
        "adapter_by_mask_count": {
            str(count): condition_summary(
                labels,
                adapter_probabilities[
                    (masks.sum(dim=1) == count).cpu().numpy()
                ],
                args.threshold,
            )
            for count in range(5)
        },
        "inversion": {
            "base_error_mask_events": base_error_mask_events,
            "recovered_mask_events": recovered_mask_events,
            "harm_mask_events": harmed_mask_events,
            "recovery_rate": (
                recovered_mask_events / base_error_mask_events
                if base_error_mask_events
                else None
            ),
            "harm_rate": (
                harmed_mask_events / base_correct_events
                if base_correct_events
                else None
            ),
            "net_recovered_mask_events": recovered_mask_events - harmed_mask_events,
            "base_flipped_videos": int(base_flipped_video.sum()),
            "base_flipped_videos_with_any_recovery": int(
                np.sum(base_flipped_video & any_recovered_video)
            ),
        },
        "feature_restoration": feature_restoration_metrics(
            model, clean, blurred, masks
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with output.with_suffix(".per_video.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["video_id", "label", "mask_id", "mask_bits", "base_probability", "adapter_probability"]
        )
        for mask_index, mask in enumerate(masks.cpu().numpy().astype(int)):
            for index, video_id in enumerate(video_ids):
                writer.writerow(
                    [
                        video_id,
                        "fake" if labels[index] else "real",
                        mask_index,
                        "".join(str(value) for value in mask),
                        float(base_mask_probability[mask_index, index]),
                        float(adapter_probabilities[mask_index, index]),
                    ]
                )
    return result


def main() -> None:
    result = evaluate(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
