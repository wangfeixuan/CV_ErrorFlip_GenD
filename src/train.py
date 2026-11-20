from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from .adapter import GenDAdapter
from .constants import DEFAULT_CONFIG
from .data import assign_group_splits, split_summary, write_manifest
from .gend_model import choose_device
from .masks import compose_frame_probabilities, compose_masks, masks_with_count
from .metrics import binary_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument(
        "--method",
        choices=(
            "clean_only",
            "uniform",
            "jpeg_only",
            "resize_only",
            "uniform_random",
            "flip_hard",
            "flip_weighted",
        ),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--bottleneck", type=int)
    parser.add_argument("--blurred-frames", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_features(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "features_original",
            "features_blur",
            "labels",
            "label_index",
            "video_ids",
            "group_ids",
            "head_weight",
            "head_bias",
            "base_logits",
            "base_logits_blur",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Feature cache is missing arrays: {sorted(missing)}")
        return {key: data[key] for key in data.files}


def make_records(data: dict, split_seed: int, ratios: dict) -> list[dict]:
    return assign_group_splits(
        [
            {
                "video_id": str(video_id),
                "label": str(label),
                "label_index": int(label_index),
                "group_id": str(group_id),
            }
            for video_id, label, label_index, group_id in zip(
                data["video_ids"],
                data["labels"],
                data["label_index"],
                data["group_ids"],
            )
        ],
        ratios,
        split_seed,
    )


def mask_losses(
    model: GenDAdapter,
    clean: torch.Tensor,
    blurred: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    composed = compose_masks(clean, blurred, masks)
    flat = composed.reshape(-1, composed.shape[-2], composed.shape[-1])
    _, probabilities = model(flat)
    losses = torch.nn.functional.binary_cross_entropy(
        probabilities.clamp(1e-6, 1 - 1e-6),
        labels.repeat(len(masks)),
        reduction="none",
    ).reshape(len(masks), len(labels))
    return losses


def select_flip_hard_loss(
    losses: torch.Tensor,
    base_clean_probability: torch.Tensor,
    base_mask_probability: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the hardest currently wrong mask that was also a base error."""
    clean_correct = (base_clean_probability >= threshold) == labels.bool()
    mask_wrong = (base_mask_probability >= threshold) != labels[:, None].bool()
    eligible = clean_correct[:, None] & mask_wrong
    all_candidates = torch.ones_like(eligible)
    eligible_or_fallback = torch.where(
        eligible.any(dim=1, keepdim=True), eligible, all_candidates
    )
    scores = losses.T.masked_fill(~eligible_or_fallback, -torch.inf)
    selected = scores.argmax(dim=1)
    selected_loss = losses[selected, torch.arange(len(labels), device=losses.device)]
    return selected_loss.mean(), selected


def select_flip_weighted_loss(
    losses: torch.Tensor,
    base_clean_probability: torch.Tensor,
    base_mask_probability: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
    flip_weight: float,
) -> torch.Tensor:
    """Emphasize base flip samples while retaining every mask in the loss."""
    clean_correct = (base_clean_probability >= threshold) == labels.bool()
    mask_wrong = (base_mask_probability >= threshold) != labels[:, None].bool()
    eligible = clean_correct[:, None] & mask_wrong
    weights = torch.where(
        eligible,
        torch.full_like(eligible, flip_weight, dtype=losses.dtype),
        torch.ones_like(losses.T, dtype=losses.dtype),
    )
    weighted_loss = (losses.T * weights).sum(dim=1) / weights.sum(dim=1)
    return weighted_loss.mean()


@torch.inference_mode()
def evaluate_loss_and_metrics(
    model: GenDAdapter,
    clean: torch.Tensor,
    blurred: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    threshold: float,
) -> dict:
    model.eval()
    _, clean_probability = model(clean)
    losses = mask_losses(model, clean, blurred, labels, masks)
    probabilities = []
    for mask in masks:
        composed = compose_masks(clean, blurred, mask[None])
        _, probability = model(composed[0])
        probabilities.append(probability)
    probabilities = torch.stack(probabilities)
    return {
        "loss": float(
            torch.nn.functional.binary_cross_entropy(
                clean_probability.clamp(1e-6, 1 - 1e-6), labels
            )
            + losses.mean()
        ),
        "clean": binary_metrics(
            labels.cpu().numpy(),
            clean_probability.cpu().numpy(),
            threshold=threshold,
        ),
        "degraded_mean_accuracy": float(
            np.mean(
                [
                    binary_metrics(
                        labels.cpu().numpy(),
                        probability.cpu().numpy(),
                        threshold=threshold,
                    )["accuracy"]
                    for probability in probabilities
                ]
            )
        ),
    }


def train(args: argparse.Namespace) -> Path:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    training = dict(config["training"])
    model_config = dict(config["model"])
    overrides = {
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }
    training.update({key: value for key, value in overrides.items() if value is not None})
    if args.bottleneck is not None:
        model_config["bottleneck"] = args.bottleneck

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output} is not empty; pass --overwrite to reuse it")
    output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)
    data = load_features(args.features)
    split_seed = int(config["seed"])
    records = make_records(data, split_seed, config["data"]["split_ratios"])
    write_manifest(records, output / "split_manifest.csv")
    (output / "split_summary.json").write_text(
        json.dumps(split_summary(records), indent=2), encoding="utf-8"
    )
    split_by_video = {record["video_id"]: record["split"] for record in records}
    splits = np.asarray([split_by_video[str(video_id)] for video_id in data["video_ids"]])
    train_index = np.flatnonzero(splits == "train")
    validation_index = np.flatnonzero(splits == "validation")

    clean = torch.from_numpy(data["features_original"]).to(device)
    blurred = torch.from_numpy(data["features_blur"]).to(device)
    label_index = torch.from_numpy(data["label_index"].astype(np.int64)).to(device)
    base_clean_frame_probability = torch.softmax(
        torch.from_numpy(data["base_logits"]), dim=-1
    )[..., 1].to(device)
    base_blur_frame_probability = torch.softmax(
        torch.from_numpy(data["base_logits_blur"]), dim=-1
    )[..., 1].to(device)
    base_clean_probability = base_clean_frame_probability.mean(dim=1)
    if not 0 <= args.blurred_frames <= 4:
        raise ValueError("--blurred-frames must be between 0 and 4")
    train_masks = masks_with_count(4, args.blurred_frames, device=device)
    evaluation_masks = masks_with_count(4, 2, device=device)
    base_mask_probability = compose_frame_probabilities(
        base_clean_frame_probability,
        base_blur_frame_probability,
        train_masks,
    ).T

    model = GenDAdapter.from_pretrained_head(
        torch.from_numpy(data["head_weight"]),
        torch.from_numpy(data["head_bias"]),
        bottleneck=int(model_config["bottleneck"]),
        dropout=float(model_config["dropout"]),
        scale_init=float(model_config["scale_init"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    best_loss = math.inf
    best_epoch = -1
    stale_epochs = 0
    history = []
    train_tensor = torch.from_numpy(train_index.astype(np.int64)).to(device)
    validation_tensor = torch.from_numpy(validation_index.astype(np.int64)).to(device)

    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        permutation = torch.randperm(len(train_tensor), device=device)
        train_indices = train_tensor[permutation]
        epoch_loss = 0.0
        used = 0
        for start in range(0, len(train_indices), int(training["batch_size"])):
            index = train_indices[start : start + int(training["batch_size"])]
            labels = label_index[index].float()
            if args.method == "clean_only":
                degraded_loss = None
            elif args.method == "uniform_random":
                selected = torch.randint(
                    0,
                    len(train_masks),
                    (len(index),),
                    device=device,
                )
                mask = train_masks[selected].to(clean.dtype)
                composed = (
                    clean[index] * (1.0 - mask[:, :, None])
                    + blurred[index] * mask[:, :, None]
                )
                _, degraded_probability = model(composed)
                degraded_loss = torch.nn.functional.binary_cross_entropy(
                    degraded_probability.clamp(1e-6, 1 - 1e-6),
                    labels,
                )
            else:
                losses = mask_losses(
                    model, clean[index], blurred[index], labels, train_masks
                )
                if args.method in ("uniform", "jpeg_only", "resize_only"):
                    degraded_loss = losses.mean()
                elif args.method == "flip_hard":
                    degraded_loss, _ = select_flip_hard_loss(
                        losses,
                        base_clean_probability[index],
                        base_mask_probability[index],
                        labels,
                        float(training["threshold"]),
                    )
                else:
                    degraded_loss = select_flip_weighted_loss(
                        losses,
                        base_clean_probability[index],
                        base_mask_probability[index],
                        labels,
                        float(training["threshold"]),
                        float(training["flip_weight"]),
                    )
            _, clean_probability = model(clean[index])
            clean_loss = torch.nn.functional.binary_cross_entropy(
                clean_probability.clamp(1e-6, 1 - 1e-6), labels
            )
            if degraded_loss is None:
                # Match the total loss weight used by mask-trained adapters.
                total_weight = (
                    float(training["clean_loss_weight"])
                    + float(training["degraded_loss_weight"])
                )
                loss = total_weight * clean_loss
            else:
                loss = (
                    float(training["clean_loss_weight"]) * clean_loss
                    + float(training["degraded_loss_weight"]) * degraded_loss
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
            optimizer.step()
            epoch_loss += float(loss.item()) * len(index)
            used += len(index)

        validation = evaluate_loss_and_metrics(
            model,
            clean[validation_tensor],
            blurred[validation_tensor],
            label_index[validation_tensor].float(),
            evaluation_masks,
            float(training["threshold"]),
        )
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / used,
            "validation_loss": validation["loss"],
            "validation_clean_accuracy": validation["clean"]["accuracy"],
            "validation_clean_auc": validation["clean"]["auc"],
            "validation_mask_accuracy": validation["degraded_mean_accuracy"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "method": args.method,
                    "seed": args.seed,
                    "split_seed": split_seed,
                    "blurred_frames": args.blurred_frames,
                    "epoch": epoch,
                    "validation": validation,
                    "feature_file": str(Path(args.features).resolve()),
                    "split_by_video": split_by_video,
                    "config": {
                        **config,
                        "training": training,
                        "model": model_config,
                        "method": args.method,
                        "seed": args.seed,
                    },
                    "head_weight": data["head_weight"],
                    "head_bias": data["head_bias"],
                },
                output / "best.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= int(training["patience"]):
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (output / "run.json").write_text(
        json.dumps(
            {
                "method": args.method,
                "seed": args.seed,
                "split_seed": split_seed,
                "blurred_frames": args.blurred_frames,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "feature_file": str(Path(args.features).resolve()),
                "split_summary": split_summary(records),
                "split_seed": args.seed,
                "device": str(device),
                "config": {**config, "training": training, "model": model_config},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output / "best.pt"


def main() -> None:
    path = train(parse_args())
    print(f"Best checkpoint: {path}")


if __name__ == "__main__":
    main()
