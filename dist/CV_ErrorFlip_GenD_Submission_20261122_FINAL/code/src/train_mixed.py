from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from .adapter import GenDAdapter
from .constants import DEFAULT_CONFIG
from .data import assign_group_splits, split_summary, write_manifest
from .gend_model import choose_device
from .masks import compose_masks, masks_with_count
from .metrics import binary_metrics
from .mixed_features import load_mixed_features
from .train import binary_cross_entropy, class_balance_weight, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blur-features", required=True)
    parser.add_argument("--jpeg-features", required=True)
    parser.add_argument("--resize-features", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--type-strategy",
        choices=("random_one", "all_types"),
        default="random_one",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--bottleneck", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


@torch.inference_mode()
def evaluate_validation(
    model: GenDAdapter,
    clean: torch.Tensor,
    degradations: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    threshold: float,
    fake_class_weight: float = 1.0,
) -> dict:
    model.eval()
    _, clean_probability = model(clean)
    clean_loss = binary_cross_entropy(
        clean_probability, labels, fake_class_weight
    )
    details = {}
    losses = []
    for type_index, type_name in enumerate(("blur", "jpeg", "resize")):
        composed = compose_masks(clean, degradations[type_index], masks)
        flat = composed.reshape(-1, composed.shape[-2], composed.shape[-1])
        _, probabilities = model(flat)
        losses.append(
            binary_cross_entropy(
                probabilities,
                labels.repeat(len(masks)),
                fake_class_weight,
            )
        )
        per_mask = probabilities.reshape(len(masks), len(labels))
        details[type_name] = {
            "accuracy": float(
                np.mean(
                    [
                        binary_metrics(
                            labels.cpu().numpy(),
                            probability.cpu().numpy(),
                            threshold=threshold,
                        )["accuracy"]
                        for probability in per_mask
                    ]
                )
            )
        }
    degraded_loss = torch.stack(losses).mean()
    return {
        "loss": float(clean_loss + degraded_loss),
        "clean": binary_metrics(
            labels.cpu().numpy(), clean_probability.cpu().numpy(), threshold
        ),
        "degradations": details,
        "degraded_mean_accuracy": float(
            np.mean([item["accuracy"] for item in details.values()])
        ),
    }


def train(args: argparse.Namespace) -> Path:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    training = dict(config["training"])
    model_config = dict(config["model"])
    method_name = "mixed" if args.type_strategy == "random_one" else "mixed_full"
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
    data = load_mixed_features(
        args.blur_features, args.jpeg_features, args.resize_features
    )
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
    fake_class_weight = class_balance_weight(
        data["label_index"],
        train_index,
        bool(training.get("class_balance_weighting", False)),
    )

    clean = torch.from_numpy(data["clean"]).to(device)
    degradation_list = [
        torch.from_numpy(data["degradations"][name]).to(device)
        for name in ("blur", "jpeg", "resize")
    ]
    degradations = torch.stack(degradation_list)
    labels = torch.from_numpy(data["label_index"].astype(np.int64)).to(device)
    masks = masks_with_count(4, 2, device=device)
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
    train_tensor = torch.from_numpy(train_index.astype(np.int64)).to(device)
    validation_tensor = torch.from_numpy(validation_index.astype(np.int64)).to(device)
    best_loss = math.inf
    best_epoch = -1
    stale_epochs = 0
    history = []

    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        permutation = torch.randperm(len(train_tensor), device=device)
        train_indices = train_tensor[permutation]
        epoch_loss = 0.0
        used = 0
        type_counts = torch.zeros(3, dtype=torch.long)
        for start in range(0, len(train_indices), int(training["batch_size"])):
            index = train_indices[start : start + int(training["batch_size"])]
            batch_labels = labels[index].float()
            mask_ids = torch.randint(0, len(masks), (len(index),), device=device)
            selected_masks = masks[mask_ids].to(clean.dtype)
            if args.type_strategy == "random_one":
                type_ids = torch.randint(0, 3, (len(index),), device=device)
                type_counts += torch.bincount(type_ids, minlength=3).cpu()
                selected_degraded = degradations[
                    type_ids, torch.arange(len(index), device=device)
                ]
                degraded_input = (
                    clean[index] * (1.0 - selected_masks[:, :, None])
                    + selected_degraded * selected_masks[:, :, None]
                )
                _, degraded_probability = model(degraded_input)
                degraded_loss = binary_cross_entropy(
                    degraded_probability, batch_labels, fake_class_weight
                )
            else:
                type_counts += len(index)
                type_losses = []
                for type_index in range(3):
                    degraded_input = (
                        clean[index] * (1.0 - selected_masks[:, :, None])
                        + degradations[type_index, index]
                        * selected_masks[:, :, None]
                    )
                    _, degraded_probability = model(degraded_input)
                    type_losses.append(
                        binary_cross_entropy(
                            degraded_probability,
                            batch_labels,
                            fake_class_weight,
                        )
                    )
                degraded_loss = torch.stack(type_losses).mean()
            _, clean_probability = model(clean[index])
            clean_loss = binary_cross_entropy(
                clean_probability, batch_labels, fake_class_weight
            )
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

        validation = evaluate_validation(
            model,
            clean[validation_tensor],
            degradations[:, validation_tensor],
            labels[validation_tensor].float(),
            masks,
            float(training["threshold"]),
            fake_class_weight,
        )
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / used,
            "validation_loss": validation["loss"],
            "validation_clean_accuracy": validation["clean"]["accuracy"],
            "validation_clean_auc": validation["clean"]["auc"],
            "validation_blur_accuracy": validation["degradations"]["blur"]["accuracy"],
            "validation_jpeg_accuracy": validation["degradations"]["jpeg"]["accuracy"],
            "validation_resize_accuracy": validation["degradations"]["resize"]["accuracy"],
            "train_blur_samples": int(type_counts[0]),
            "train_jpeg_samples": int(type_counts[1]),
            "train_resize_samples": int(type_counts[2]),
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
                    "method": method_name,
                    "seed": args.seed,
                    "split_seed": split_seed,
                    "blurred_frames": 2,
                    "fake_class_weight": fake_class_weight,
                    "epoch": epoch,
                    "validation": validation,
                    "split_by_video": split_by_video,
                    "feature_files": data["feature_paths"],
                    "config": {
                        **config,
                        "training": training,
                        "model": model_config,
                        "method": method_name,
                        "type_strategy": args.type_strategy,
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
                "method": method_name,
                "seed": args.seed,
                "split_seed": split_seed,
                "blurred_frames": 2,
                "fake_class_weight": fake_class_weight,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "split_summary": split_summary(records),
                "feature_paths": data["feature_paths"],
                "clean_max_difference": data["clean_max_difference"],
                "type_strategy": args.type_strategy,
                "device": str(device),
                "config": {**config, "training": training, "model": model_config},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output / "best.pt"


def main() -> None:
    print(train(parse_args()))


if __name__ == "__main__":
    main()
