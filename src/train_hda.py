from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from .adapter import GenDAdapter
from .data import assign_group_splits, split_summary, write_manifest
from .gend_model import choose_device
from .metrics import binary_metrics
from .train import binary_cross_entropy, set_seed


TYPE_CLEAN = 0
TYPE_BLUR = 1
TYPE_JPEG = 2
TYPE_RESIZE = 3
TYPE_NAMES = ("clean", "blur", "jpeg", "resize")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blur-features", required=True)
    parser.add_argument("--jpeg-features", required=True)
    parser.add_argument("--resize-features", required=True)
    parser.add_argument(
        "--weighting",
        choices=("uniform", "confidence", "learned"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config", default="configs/hda.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--bottleneck", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_hda_features(
    blur_path: str | Path,
    jpeg_path: str | Path,
    resize_path: str | Path,
) -> dict:
    paths = {
        "blur": Path(blur_path).expanduser().resolve(),
        "jpeg": Path(jpeg_path).expanduser().resolve(),
        "resize": Path(resize_path).expanduser().resolve(),
    }
    caches = {name: load_npz(path) for name, path in paths.items()}
    canonical = caches["blur"]
    required = {
        "features_original",
        "features_blur",
        "base_logits",
        "base_logits_blur",
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
        delta = np.max(
            np.abs(cache["features_original"] - canonical["features_original"])
        )
        if delta > 1e-4:
            raise ValueError(f"Clean features differ for {name}: {delta}")
    return {
        "clean": canonical["features_original"],
        "degradations": {
            "blur": canonical["features_blur"],
            "jpeg": caches["jpeg"]["features_blur"],
            "resize": caches["resize"]["features_blur"],
        },
        "base_clean_logits": canonical["base_logits"],
        "base_degraded_logits": {
            "blur": canonical["base_logits_blur"],
            "jpeg": caches["jpeg"]["base_logits_blur"],
            "resize": caches["resize"]["base_logits_blur"],
        },
        "labels": canonical["labels"],
        "label_index": canonical["label_index"],
        "video_ids": canonical["video_ids"],
        "group_ids": canonical["group_ids"],
        "head_weight": canonical["head_weight"],
        "head_bias": canonical["head_bias"],
        "feature_paths": {name: str(path) for name, path in paths.items()},
    }


def heterogeneous_patterns(type_a: int, type_b: int) -> torch.Tensor:
    if type_a == type_b:
        raise ValueError("Degradation types must differ")
    patterns = []
    for position_a in range(4):
        for position_b in range(4):
            if position_a == position_b:
                continue
            pattern = [TYPE_CLEAN] * 4
            pattern[position_a] = type_a
            pattern[position_b] = type_b
            patterns.append(pattern)
    return torch.tensor(patterns, dtype=torch.long)


def compose_pattern_batch(
    clean: torch.Tensor,
    degradations: dict[int, torch.Tensor],
    pattern: torch.Tensor,
) -> torch.Tensor:
    result = clean.clone()
    for type_id in (TYPE_BLUR, TYPE_JPEG, TYPE_RESIZE):
        selector = (pattern == type_id)[:, :, None]
        result = torch.where(selector, degradations[type_id], result)
    return result


def compose_probability_batch(
    clean_probability: torch.Tensor,
    degraded_probability: dict[int, torch.Tensor],
    pattern: torch.Tensor,
) -> torch.Tensor:
    result = clean_probability.clone()
    for type_id in (TYPE_BLUR, TYPE_JPEG, TYPE_RESIZE):
        selector = pattern == type_id
        result = torch.where(selector, degraded_probability[type_id], result)
    return result


class ReliabilityAdapter(GenDAdapter):
    def __init__(
        self,
        feature_dim: int,
        bottleneck: int,
        dropout: float,
        scale_init: float,
        reliability_hidden: int,
    ):
        super().__init__(
            feature_dim=feature_dim,
            num_classes=2,
            bottleneck=bottleneck,
            dropout=dropout,
            scale_init=scale_init,
        )
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, reliability_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(reliability_hidden, 1),
        )

    def forward_aggregated(
        self,
        features: torch.Tensor,
        weighting: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        adapted = self.adapter(features)
        logits = self.head(adapted)
        probabilities = logits.softmax(dim=-1)[..., 1]
        scores = None
        if weighting == "uniform":
            weights = torch.full_like(probabilities, 1.0 / probabilities.shape[-1])
        elif weighting == "confidence":
            entropy = -(
                logits.softmax(dim=-1)
                * logits.log_softmax(dim=-1)
            ).sum(dim=-1)
            inverse = 1.0 / (entropy + 0.05)
            weights = inverse / inverse.sum(dim=-1, keepdim=True)
        else:
            scores = self.scorer(adapted).squeeze(-1)
            weights = scores.softmax(dim=-1)
        video_probability = (weights * probabilities).sum(dim=-1)
        return video_probability, weights, scores, adapted


def sample_patterns(
    patterns: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    indices = torch.randint(
        len(patterns),
        (batch_size,),
        generator=generator,
        device=patterns.device,
    )
    return patterns[indices]


def reliability_rank_loss(
    scores: torch.Tensor,
    patterns: torch.Tensor,
    margin: float = 0.25,
) -> torch.Tensor:
    clean_mask = patterns == TYPE_CLEAN
    clean_scores = (scores * clean_mask).sum(dim=-1) / clean_mask.sum(
        dim=-1
    ).clamp_min(1)
    degraded_scores = (
        scores * (~clean_mask)
    ).sum(dim=-1) / (~clean_mask).sum(dim=-1).clamp_min(1)
    return torch.nn.functional.softplus(
        margin - (clean_scores - degraded_scores)
    ).mean()


@torch.inference_mode()
def evaluate_patterns(
    model: ReliabilityAdapter,
    clean: torch.Tensor,
    degradations: dict[int, torch.Tensor],
    base_clean_probability: torch.Tensor,
    base_degraded_probability: dict[int, torch.Tensor],
    labels: torch.Tensor,
    patterns: torch.Tensor,
    weighting: str,
    threshold: float,
) -> dict:
    model.eval()
    base_rows = []
    adapter_rows = []
    oracle_rows = []
    weight_rows = []
    for pattern in patterns:
        repeated = pattern.unsqueeze(0).expand(len(labels), -1)
        mixed = compose_pattern_batch(clean, degradations, repeated)
        adapter_probability, weights, _, _ = model.forward_aggregated(
            mixed, weighting
        )
        base_frames = compose_probability_batch(
            base_clean_probability, base_degraded_probability, repeated
        )
        base_probability = base_frames.mean(dim=-1)
        clean_mask = (repeated == TYPE_CLEAN).float()
        oracle_weights = clean_mask / clean_mask.sum(dim=-1, keepdim=True)
        oracle_probability = (oracle_weights * base_frames).sum(dim=-1)
        base_rows.append(
            binary_metrics(
                labels.cpu().numpy(),
                base_probability.cpu().numpy(),
                threshold,
            )
        )
        adapter_rows.append(
            binary_metrics(
                labels.cpu().numpy(),
                adapter_probability.cpu().numpy(),
                threshold,
            )
        )
        oracle_rows.append(
            binary_metrics(
                labels.cpu().numpy(),
                oracle_probability.cpu().numpy(),
                threshold,
            )
        )
        weights_np = weights.cpu().numpy()
        weight_rows.append(
            {
                "clean": float(weights_np[clean_mask.cpu().numpy().astype(bool)].mean()),
                "degraded": float(
                    weights_np[(~clean_mask.cpu().numpy().astype(bool))].mean()
                ),
            }
        )

    def mean_accuracy(rows):
        return float(np.mean([row["accuracy"] for row in rows]))

    def worst_accuracy(rows):
        return float(
            np.mean(
                [
                    np.min(
                        [
                            row["accuracy"]
                            for row in rows
                        ]
                    )
                ]
            )
        )

    adapter_video = np.asarray(
        [
            metric["accuracy"]
            for metric in adapter_rows
        ]
    )
    base_video = np.asarray(
        [
            metric["accuracy"]
            for metric in base_rows
        ]
    )
    return {
        "patterns": len(patterns),
        "base_mean_accuracy": mean_accuracy(base_rows),
        "base_worst_accuracy": float(np.min(base_video)),
        "adapter_mean_accuracy": mean_accuracy(adapter_rows),
        "adapter_worst_accuracy": float(np.min(adapter_video)),
        "oracle_mean_accuracy": mean_accuracy(oracle_rows),
        "oracle_worst_accuracy": float(np.min([row["accuracy"] for row in oracle_rows])),
        "mean_clean_weight": float(np.mean([row["clean"] for row in weight_rows])),
        "mean_degraded_weight": float(
            np.mean([row["degraded"] for row in weight_rows])
        ),
        "weights": weight_rows,
    }


def evaluation_suites(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "seen_blur_jpeg": heterogeneous_patterns(TYPE_BLUR, TYPE_JPEG).to(device),
        "unseen_blur_resize": heterogeneous_patterns(TYPE_BLUR, TYPE_RESIZE).to(device),
        "unseen_jpeg_resize": heterogeneous_patterns(TYPE_JPEG, TYPE_RESIZE).to(device),
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
    training.update(
        {key: value for key, value in overrides.items() if value is not None}
    )
    if args.bottleneck is not None:
        model_config["bottleneck"] = args.bottleneck

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output} is not empty; pass --overwrite to reuse it")
    output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)
    data = load_hda_features(
        args.blur_features, args.jpeg_features, args.resize_features
    )
    records = assign_group_splits(
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
        config["data"]["split_ratios"],
        int(config["seed"]),
    )
    split_by_video = {
        record["video_id"]: record["split"] for record in records
    }
    write_manifest(records, output / "split_manifest.csv")
    (output / "split_summary.json").write_text(
        json.dumps(split_summary(records), indent=2), encoding="utf-8"
    )
    splits = np.asarray(
        [split_by_video[str(video_id)] for video_id in data["video_ids"]]
    )
    train_index = np.flatnonzero(splits == "train")
    validation_index = np.flatnonzero(splits == "validation")
    test_index = np.flatnonzero(splits == "test")

    clean = torch.from_numpy(data["clean"]).to(device)
    degradations = {
        TYPE_BLUR: torch.from_numpy(data["degradations"]["blur"]).to(device),
        TYPE_JPEG: torch.from_numpy(data["degradations"]["jpeg"]).to(device),
        TYPE_RESIZE: torch.from_numpy(data["degradations"]["resize"]).to(device),
    }
    labels = torch.from_numpy(data["label_index"].astype(np.int64)).to(device)
    base_clean_frames = (
        torch.from_numpy(data["base_clean_logits"]).to(device).softmax(dim=-1)[..., 1]
    )
    base_degraded_frames = {
        type_id: torch.from_numpy(
            data["base_degraded_logits"][TYPE_NAMES[type_id]]
        ).to(device).softmax(dim=-1)[..., 1]
        for type_id in (TYPE_BLUR, TYPE_JPEG, TYPE_RESIZE)
    }

    model = ReliabilityAdapter(
        feature_dim=clean.shape[-1],
        bottleneck=int(model_config["bottleneck"]),
        dropout=float(model_config["dropout"]),
        scale_init=float(model_config["scale_init"]),
        reliability_hidden=int(model_config["reliability_hidden"]),
    )
    with torch.no_grad():
        model.head.weight.copy_(
            torch.from_numpy(data["head_weight"])
        )
        model.head.bias.copy_(torch.from_numpy(data["head_bias"]))
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    train_patterns = heterogeneous_patterns(TYPE_BLUR, TYPE_JPEG).to(device)
    validation_patterns = train_patterns
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    train_tensor = torch.from_numpy(train_index.astype(np.int64)).to(device)
    validation_tensor = torch.from_numpy(validation_index.astype(np.int64)).to(device)
    best_score = math.inf
    best_epoch = -1
    stale_epochs = 0
    history = []

    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        permutation = torch.randperm(len(train_tensor), device=device)
        train_indices = train_tensor[permutation]
        epoch_loss = 0.0
        used = 0
        for start in range(0, len(train_indices), int(training["batch_size"])):
            index = train_indices[start : start + int(training["batch_size"])]
            batch_labels = labels[index].float()
            pattern = sample_patterns(train_patterns, len(index), generator)
            mixed = compose_pattern_batch(
                clean[index],
                {
                    type_id: value[index]
                    for type_id, value in degradations.items()
                },
                pattern,
            )
            mixed_probability, weights, scores, _ = model.forward_aggregated(
                mixed, args.weighting
            )
            mixed_loss = binary_cross_entropy(mixed_probability, batch_labels)
            clean_probability, _, _, _ = model.forward_aggregated(
                clean[index], "uniform"
            )
            clean_loss = binary_cross_entropy(clean_probability, batch_labels)
            loss = (
                float(training["clean_loss_weight"]) * clean_loss
                + float(training["mixed_loss_weight"]) * mixed_loss
            )
            if args.weighting == "learned":
                assert scores is not None
                rank_loss = reliability_rank_loss(scores, pattern)
                uniform_prior = (
                    weights * (weights.clamp_min(1e-6).log() - math.log(0.25))
                ).sum(dim=-1).mean()
                loss = (
                    loss
                    + float(training["reliability_rank_weight"]) * rank_loss
                    + float(training["uniform_prior_weight"]) * uniform_prior
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
            optimizer.step()
            epoch_loss += float(loss.item()) * len(index)
            used += len(index)

        validation = evaluate_patterns(
            model,
            clean[validation_tensor],
            {
                type_id: value[validation_tensor]
                for type_id, value in degradations.items()
            },
            base_clean_frames[validation_tensor],
            {
                type_id: value[validation_tensor]
                for type_id, value in base_degraded_frames.items()
            },
            labels[validation_tensor],
            validation_patterns,
            args.weighting,
            float(training["threshold"]),
        )
        score = 1.0 - validation["adapter_mean_accuracy"]
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / used,
            "validation_loss": score,
            "validation_mean_accuracy": validation["adapter_mean_accuracy"],
            "validation_worst_accuracy": validation["adapter_worst_accuracy"],
            "validation_oracle_accuracy": validation["oracle_mean_accuracy"],
            "clean_weight": validation["mean_clean_weight"],
            "degraded_weight": validation["mean_degraded_weight"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "method": f"hda_{args.weighting}",
                    "seed": args.seed,
                    "split_by_video": split_by_video,
                    "feature_files": data["feature_paths"],
                    "head_weight": data["head_weight"],
                    "head_bias": data["head_bias"],
                    "config": {
                        **config,
                        "training": training,
                        "model": model_config,
                        "weighting": args.weighting,
                    },
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

    checkout = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkout["state_dict"])
    model.eval()
    test_tensor = torch.from_numpy(test_index.astype(np.int64)).to(device)
    with torch.inference_mode():
        adapter_clean_probability, _, _, _ = model.forward_aggregated(
            clean[test_tensor], "uniform"
        )
    base_clean_probability = base_clean_frames[test_tensor].mean(dim=-1)
    clean_metrics = {
        "base_clean": binary_metrics(
            labels[test_tensor].cpu().numpy(),
            base_clean_probability.cpu().numpy(),
            float(training["threshold"]),
        ),
        "adapter_clean": binary_metrics(
            labels[test_tensor].cpu().numpy(),
            adapter_clean_probability.cpu().numpy(),
            float(training["threshold"]),
        ),
    }
    results = {}
    for suite_name, patterns in evaluation_suites(device).items():
        result = evaluate_patterns(
            model,
            clean[test_tensor],
            {
                type_id: value[test_tensor]
                for type_id, value in degradations.items()
            },
            base_clean_frames[test_tensor],
            {
                type_id: value[test_tensor]
                for type_id, value in base_degraded_frames.items()
            },
            labels[test_tensor],
            patterns,
            args.weighting,
            float(training["threshold"]),
        )
        result.update(
            {
                "method": f"hda_{args.weighting}",
                "seed": args.seed,
                "suite": suite_name,
                "videos": int(len(test_index)),
                **clean_metrics,
            }
        )
        results[suite_name] = result
        (output / f"eval_{suite_name}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    (output / "run.json").write_text(
        json.dumps(
            {
                "method": f"hda_{args.weighting}",
                "seed": args.seed,
                "best_epoch": best_epoch,
                "split_summary": split_summary(records),
                "feature_files": data["feature_paths"],
                "device": str(device),
                "config": {**config, "training": training, "model": model_config},
                "test": results,
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
