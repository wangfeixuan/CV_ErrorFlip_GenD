from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from .adapter import GenDAdapter
from .gend_model import choose_device
from .metrics import binary_metrics
from .residual_diagnostics import (
    CONDITIONS,
    correct_class_margin,
    correlation,
    jsonable,
    load_npz,
    probabilities,
    split_indices,
)


RESIDUALS = ("R1", "R2", "R3")
RATIOS = (0.01, 0.05, 0.10, 0.20, 0.50)
ALPHAS = tuple(round(index * 0.1, 1) for index in range(11))
SINGLE_SELECTORS = (
    "random",
    "magnitude",
    "classifier",
    "signed_discriminative",
    "discriminative",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blur-features", required=True)
    parser.add_argument("--jpeg-features", required=True)
    parser.add_argument("--resize-features", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--primary-seed", type=int, default=2026)
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[GenDAdapter, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GenDAdapter.from_pretrained_head(
        torch.from_numpy(checkpoint["head_weight"]),
        torch.from_numpy(checkpoint["head_bias"]),
        bottleneck=int(checkpoint["config"]["model"]["bottleneck"]),
        dropout=float(checkpoint["config"]["model"]["dropout"]),
        scale_init=float(checkpoint["config"]["model"]["scale_init"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def head_original(checkpoint: dict, device: torch.device) -> torch.nn.Linear:
    head = torch.nn.Linear(
        checkpoint["head_weight"].shape[1],
        checkpoint["head_weight"].shape[0],
    ).to(device)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(checkpoint["head_weight"]))
        head.bias.copy_(torch.from_numpy(checkpoint["head_bias"]))
    return head.eval()


@torch.inference_mode()
def feature_bundle(
    model: GenDAdapter,
    data: dict,
    indices: np.ndarray,
    device: torch.device,
) -> dict:
    z0 = torch.from_numpy(data["features_original"][indices]).to(device).float()
    zd = torch.from_numpy(data["features_blur"][indices]).to(device).float()
    za = model.adapter(zd)
    zc = model.adapter(z0)
    return {
        "z0": z0,
        "zd": zd,
        "za": za,
        "zc": zc,
        "R1": z0 - zd,
        "R2": z0 - za,
        "R3": zc - za,
        "labels": torch.from_numpy(
            data["label_index"][indices].astype(np.int64)
        ).to(device),
        "video_ids": data["video_ids"][indices],
        "group_ids": data["group_ids"][indices],
    }


def balanced_score(metrics: dict) -> float:
    return float(metrics["balanced_accuracy"] or 0.0)


def video_scores(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    probability = probabilities(logits)
    correct_probability = torch.where(labels.bool(), probability, 1 - probability)
    frame_margin = correct_class_margin(logits, labels)
    frame_rule = (frame_margin >= 0).bool()
    probability_rule = (probability >= 0.5).bool()
    confidence = torch.maximum(probability, 1 - probability)
    top_logits = logits.topk(dim=-1, k=1).values.squeeze(-1)
    return {
        "probability": probability,
        "correct_probability": correct_probability,
        "confidence": confidence,
        "frame_margin": frame_margin,
        "video_probability_margin": correct_probability - 0.5,
        "frame_rule_disagreement": float((frame_rule != probability_rule).float().mean()),
        "logit_scale": float(top_logits.abs().mean()),
    }


def classification_report(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
) -> dict:
    scores = video_scores(logits, labels)
    probability = scores["probability"]
    metrics = binary_metrics(
        labels.cpu().numpy(),
        probability.cpu().numpy(),
        threshold,
    )
    nll = torch.nn.functional.binary_cross_entropy(
        probability.clamp(1e-6, 1 - 1e-6),
        labels.float(),
    )
    return {
        **metrics,
        "cross_entropy": float(nll),
        "correct_probability": float(scores["correct_probability"].mean()),
        "mean_confidence": float(scores["confidence"].mean()),
        "mean_frame_logit_margin": float(scores["frame_margin"].mean()),
        "frame_video_rule_disagreement": scores[
            "frame_rule_disagreement"
        ],
        "mean_abs_logit": scores["logit_scale"],
        "probability": probability,
        "correct_probability_vector": scores["correct_probability"],
        "confidence_vector": scores["confidence"],
    }


def transition_counts(
    base_probability: torch.Tensor,
    new_probability: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
) -> dict:
    base_correct = (base_probability >= threshold) == labels.bool()
    new_correct = (new_probability >= threshold) == labels.bool()
    wrong_to_correct = ~base_correct & new_correct
    correct_to_wrong = base_correct & ~new_correct
    return {
        "wrong_to_correct": int(wrong_to_correct.sum()),
        "correct_to_wrong": int(correct_to_wrong.sum()),
        "net_correction_gain": float(
            (wrong_to_correct.sum() - correct_to_wrong.sum()) / len(labels)
        ),
        "real_wrong_to_correct": int(
            (wrong_to_correct & ~labels.bool()).sum()
        ),
        "real_correct_to_wrong": int(
            (correct_to_wrong & ~labels.bool()).sum()
        ),
        "fake_wrong_to_correct": int(
            (wrong_to_correct & labels.bool()).sum()
        ),
        "fake_correct_to_wrong": int(
            (correct_to_wrong & labels.bool()).sum()
        ),
    }


def interpolate_features(start: torch.Tensor, residual: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.nn.functional.normalize(
        start + float(alpha) * residual,
        p=2,
        dim=-1,
    )


def reverse_intervention_start(
    bundle: dict,
    residual_name: str,
) -> torch.Tensor:
    """Return the clean endpoint in the same coordinate system as residual."""
    if residual_name == "R1":
        return bundle["z0"]
    if residual_name == "R2":
        return bundle["z0"]
    if residual_name == "R3":
        return bundle["zc"]
    raise ValueError(f"Unknown residual: {residual_name}")


def cross_head_diagnosis(
    bundles: dict[str, dict[str, dict]],
    checkpoints: dict[str, dict],
    threshold: float,
) -> list[dict]:
    rows = []
    for condition, splits in bundles.items():
        for split, bundle in splits.items():
            labels = bundle["labels"]
            for checkpoint_name, checkpoint_data in checkpoints.items():
                model = checkpoint_data["model"]
                for head_name, head in (
                    ("original", checkpoint_data["original_head"]),
                    ("adapter", model.head),
                ):
                    for feature_name in ("z0", "zd", "za", "zc"):
                        metrics = classification_report(
                            head(bundle[feature_name]),
                            labels,
                            threshold,
                        )
                        rows.append(
                            {
                                "condition": condition,
                                "split": split,
                                "checkpoint": checkpoint_name,
                                "head": head_name,
                                "features": feature_name,
                                **{
                                    key: value
                                    for key, value in metrics.items()
                                    if key
                                    not in {
                                        "probability",
                                        "correct_probability_vector",
                                        "confidence_vector",
                                    }
                                },
                            }
                        )
    return rows


def residual_direction_metrics(
    model: GenDAdapter,
    bundle: dict,
    head: torch.nn.Linear,
) -> dict:
    labels = bundle["labels"]
    weight_difference = (head.weight[1] - head.weight[0]).detach()
    direction = weight_difference / weight_difference.norm().clamp_min(1e-8)
    signed_direction = torch.where(labels.bool(), 1.0, -1.0)[:, None, None] * direction
    result = {}
    for residual_name in RESIDUALS:
        residual = bundle[residual_name]
        norm = residual.norm(dim=-1)
        projection = (residual * signed_direction).sum(dim=-1)
        cosine = projection / norm.clamp_min(1e-8)
        result[residual_name] = {
            "norm_mean": float(norm.mean()),
            "norm_std": float(norm.std(unbiased=False)),
            "projection_mean": float(projection.mean()),
            "projection_std": float(projection.std(unbiased=False)),
            "direction_cosine_mean": float(cosine.mean()),
            "direction_cosine_std": float(cosine.std(unbiased=False)),
        }
        for group_name, mask in (
            ("real", ~labels.bool()),
            ("fake", labels.bool()),
        ):
            result[residual_name][group_name] = {
                "norm_mean": float(norm[mask].mean()),
                "projection_mean": float(projection[mask].mean()),
                "direction_cosine_mean": float(cosine[mask].mean()),
            }
    return result


def residual_projection_ratio(
    model: GenDAdapter,
    bundle: dict,
    head: torch.nn.Linear,
) -> dict:
    weight_difference = (head.weight[1] - head.weight[0]).detach()
    direction = weight_difference / weight_difference.norm().clamp_min(1e-8)
    labels = bundle["labels"]
    signed_direction = torch.where(labels.bool(), 1.0, -1.0)[:, None, None] * direction
    result = {}
    for residual_name in RESIDUALS:
        residual = bundle[residual_name]
        parallel = (residual * signed_direction).sum(dim=-1, keepdim=True) * signed_direction
        orthogonal = residual - parallel
        result[residual_name] = {
            "parallel_norm_mean": float(parallel.norm(dim=-1).mean()),
            "orthogonal_norm_mean": float(orthogonal.norm(dim=-1).mean()),
            "parallel_fraction": float(
                parallel.square().sum()
                / residual.square().sum().clamp_min(1e-8)
            ),
        }
    return result


def interpolate_residual(
    start: torch.Tensor,
    residual: torch.Tensor,
    alphas: tuple[float, ...],
    head: torch.nn.Linear,
    labels: torch.Tensor,
    threshold: float,
) -> list[dict]:
    rows = []
    for alpha in alphas:
        logits = head(interpolate_features(start, residual, alpha))
        metrics = classification_report(logits, labels, threshold)
        rows.append(
            {
                "alpha": alpha,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "auc": metrics["auc"],
                "cross_entropy": metrics["cross_entropy"],
                "mean_frame_logit_margin": metrics[
                    "mean_frame_logit_margin"
                ],
                "correct_probability": metrics["correct_probability"],
                "false_positive_rate": metrics["false_positive_rate"],
                "false_negative_rate": metrics["false_negative_rate"],
            }
        )
    return rows


def validation_choice(
    rows: list[dict],
    parameter: str,
) -> dict:
    return max(
        rows,
        key=lambda row: (
            (
                row.get("balanced_accuracy")
                or row.get("add_balanced_accuracy")
                or 0.0
            ),
            row.get("mean_frame_logit_margin", row.get("add_margin", 0.0)),
        ),
    )


def selector_scores(
    model: GenDAdapter,
    bundle: dict,
    residual_name: str,
    head: torch.nn.Linear,
) -> dict[str, torch.Tensor]:
    residual = bundle[residual_name]
    labels = bundle["labels"]
    weight_difference = (head.weight[1] - head.weight[0]).detach()
    sign = torch.where(labels.bool(), 1.0, -1.0)
    gradient = sign[:, None, None] * weight_difference[None, None, :]
    reference = bundle["za"] if residual_name in ("R2", "R3") else bundle["zd"]
    effective = gradient - (
        gradient * reference
    ).sum(dim=-1, keepdim=True) * reference
    directional = effective * residual
    return {
        "magnitude": residual.abs().mean(dim=(0, 1)),
        "classifier": effective.abs().mean(dim=(0, 1)),
        "signed_discriminative": directional.mean(dim=(0, 1)),
        "discriminative": torch.relu(directional).mean(dim=(0, 1)),
        "energy": residual.square().mean(dim=(0, 1)),
    }


def mask_from_indices(indices: torch.Tensor, dimension: int) -> torch.Tensor:
    result = torch.zeros(dimension, dtype=torch.bool, device=indices.device)
    result[indices] = True
    return result


def random_mask(
    dimension: int,
    count: int,
    energy: torch.Tensor,
    target_energy: float | None,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=energy.device)
    generator.manual_seed(seed)
    candidates = torch.argsort(
        torch.rand(
            2000,
            dimension,
            generator=generator,
            device=energy.device,
        ),
        dim=1,
    )[:, :count]
    if target_energy is None:
        chosen = candidates[0]
    else:
        candidate_energy = energy[candidates].sum(dim=1)
        chosen = candidates[int((candidate_energy - target_energy).abs().argmin())]
    return mask_from_indices(chosen, dimension)


def selected_masks(
    scores: dict[str, torch.Tensor],
    ratio: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    dimension = len(scores["magnitude"])
    count = max(1, min(dimension, int(round(dimension * ratio))))
    masks = {}
    for selector in (
        "magnitude",
        "classifier",
        "signed_discriminative",
        "discriminative",
    ):
        indices = torch.topk(scores[selector], count).indices
        masks[selector] = mask_from_indices(indices, dimension)
    masks["random"] = random_mask(
        dimension,
        count,
        scores["energy"],
        None,
        seed + count,
    )
    for selector in ("magnitude", "classifier", "signed_discriminative"):
        target = scores["energy"][masks[selector]].sum()
        masks[f"random_energy_{selector}"] = random_mask(
            dimension,
            count,
            scores["energy"],
            float(target),
            seed + count * 17,
        )
    return masks


def intervention(
    start: torch.Tensor,
    residual: torch.Tensor,
    mask: torch.Tensor,
    head: torch.nn.Linear,
    labels: torch.Tensor,
    direction: str,
    threshold: float,
) -> dict:
    mask_float = mask.to(start.dtype)
    if direction == "add":
        features = torch.nn.functional.normalize(
            start + mask_float * residual, dim=-1
        )
    elif direction == "reverse":
        features = torch.nn.functional.normalize(
            start - mask_float * residual, dim=-1
        )
    else:
        raise ValueError(direction)
    logits = head(features)
    return {
        **classification_report(logits, labels, threshold),
        "logits": logits,
    }


def single_dimension_experiments(
    model: GenDAdapter,
    train: dict,
    validation: dict,
    test: dict,
    residual_name: str,
    head: torch.nn.Linear,
    threshold: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    scores = selector_scores(model, train, residual_name, head)
    validation_rows = []
    test_rows = []
    masks_by_key = {}
    for ratio in RATIOS:
        masks = selected_masks(scores, ratio, seed)
        for selector, mask in masks.items():
            masks_by_key[(selector, ratio)] = mask
            start = train["za"] if residual_name in ("R2", "R3") else train["zd"]
            reference_start = validation["za"] if residual_name in ("R2", "R3") else validation["zd"]
            valid_add = intervention(
                reference_start,
                validation[residual_name],
                mask,
                head,
                validation["labels"],
                "add",
                threshold,
            )
            clean_reference = reverse_intervention_start(
                validation, residual_name
            )
            valid_reverse = intervention(
                clean_reference,
                validation[residual_name],
                mask,
                head,
                validation["labels"],
                "reverse",
                threshold,
            )
            validation_rows.append(
                {
                    "selector": selector,
                    "ratio": ratio,
                    "selected_dimensions": int(mask.sum()),
                    "add_accuracy": valid_add["accuracy"],
                    "add_balanced_accuracy": valid_add["balanced_accuracy"],
                    "add_auc": valid_add["auc"],
                    "add_margin": valid_add["mean_frame_logit_margin"],
                    "reverse_accuracy": valid_reverse["accuracy"],
                    "reverse_balanced_accuracy": valid_reverse[
                        "balanced_accuracy"
                    ],
                    "reverse_auc": valid_reverse["auc"],
                    "reverse_margin": valid_reverse["mean_frame_logit_margin"],
                }
            )
    chosen = {}
    for selector in sorted({row["selector"] for row in validation_rows}):
        candidate = [row for row in validation_rows if row["selector"] == selector]
        chosen[selector] = validation_choice(candidate, "ratio")
    for selector, choice in chosen.items():
        mask = masks_by_key[(selector, choice["ratio"])]
        test_start = test["za"] if residual_name in ("R2", "R3") else test["zd"]
        add = intervention(
            test_start,
            test[residual_name],
            mask,
            head,
            test["labels"],
            "add",
            threshold,
        )
        base = classification_report(
            head(test_start),
            test["labels"],
            threshold,
        )
        reverse = intervention(
            reverse_intervention_start(test, residual_name),
            test[residual_name],
            mask,
            head,
            test["labels"],
            "reverse",
            threshold,
        )
        reverse_start = reverse_intervention_start(test, residual_name)
        clean_base = classification_report(
            head(reverse_start),
            test["labels"],
            threshold,
        )
        test_rows.append(
            {
                "selector": selector,
                "ratio": choice["ratio"],
                "selected_dimensions": int(mask.sum()),
                "add_accuracy": add["accuracy"],
                "add_balanced_accuracy": add["balanced_accuracy"],
                "add_auc": add["auc"],
                "add_margin_change": (
                    add["mean_frame_logit_margin"]
                    - base["mean_frame_logit_margin"]
                ),
                "reverse_accuracy": reverse["accuracy"],
                "reverse_balanced_accuracy": reverse["balanced_accuracy"],
                "reverse_margin_change": (
                    reverse["mean_frame_logit_margin"]
                    - clean_base["mean_frame_logit_margin"]
                ),
                **{
                    f"add_{key}": value
                    for key, value in transition_counts(
                        base["probability"],
                        add["probability"],
                        test["labels"],
                        threshold,
                    ).items()
                },
                **{
                    f"reverse_{key}": value
                    for key, value in transition_counts(
                        clean_base["probability"],
                        reverse["probability"],
                        test["labels"],
                        threshold,
                    ).items()
                },
            }
        )
    return validation_rows, test_rows


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    def objective(log_temperature: float) -> float:
        temperature = math.exp(float(log_temperature))
        probability = probabilities(logits / temperature)
        return float(
            torch.nn.functional.binary_cross_entropy(
                probability.clamp(1e-6, 1 - 1e-6),
                labels.float(),
            )
        )

    result = minimize_scalar(
        objective,
        bounds=(math.log(0.25), math.log(10.0)),
        method="bounded",
    )
    return math.exp(float(result.x))


def expected_calibration_error(
    probabilities_tensor: torch.Tensor,
    labels: torch.Tensor,
    bins: int = 10,
) -> float:
    probability = probabilities_tensor.detach().cpu().numpy()
    target = labels.detach().cpu().numpy().astype(float)
    confidence = np.maximum(probability, 1 - probability)
    prediction = probability >= 0.5
    bins_edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for start, end in zip(bins_edges[:-1], bins_edges[1:]):
        mask = (confidence > start) & (confidence <= end)
        if mask.any():
            error += mask.mean() * abs(
                prediction[mask].astype(float).mean() - target[mask].mean()
            )
    return float(error)


def calibration_table(
    primary: dict,
    threshold: float,
) -> tuple[list[dict], dict]:
    rows = []
    probabilities_by_condition = {}
    for condition, data in primary.items():
        validation = data["validation"]
        test = data["test"]
        labels = validation["labels"]
        for model_name, valid_logits, test_logits in (
            ("GenD", validation["head0_zd_logits"], test["head0_zd_logits"]),
            ("Adapter", validation["headA_za_logits"], test["headA_za_logits"]),
        ):
            temperature = fit_temperature(valid_logits, labels)
            valid_raw = probabilities(valid_logits)
            valid_cal = probabilities(valid_logits / temperature)
            thresholds = np.linspace(0.05, 0.95, 91)
            valid_labels = labels.cpu().numpy()
            calibrated_threshold = max(
                thresholds,
                key=lambda value: binary_metrics(
                    valid_labels,
                    valid_cal.detach().cpu().numpy(),
                    float(value),
                )["balanced_accuracy"],
            )
            test_raw = classification_report(
                test_logits, test["labels"], threshold
            )
            test_cal = classification_report(
                test_logits / temperature,
                test["labels"],
                float(calibrated_threshold),
            )
            rows.append(
                {
                    "condition": condition,
                    "model": model_name,
                    "temperature": temperature,
                    "raw_threshold": threshold,
                    "calibrated_threshold": float(calibrated_threshold),
                    "raw_test_accuracy": test_raw["accuracy"],
                    "raw_test_balanced_accuracy": test_raw["balanced_accuracy"],
                    "raw_test_auc": test_raw["auc"],
                    "raw_test_nll": test_raw["cross_entropy"],
                    "raw_test_ece": expected_calibration_error(
                        test_logits.softmax(-1)[..., 1].mean(-1),
                        test["labels"],
                    ),
                    "calibrated_test_accuracy": test_cal["accuracy"],
                    "calibrated_test_balanced_accuracy": test_cal[
                        "balanced_accuracy"
                    ],
                    "calibrated_test_auc": test_cal["auc"],
                    "calibrated_test_nll": test_cal["cross_entropy"],
                    "calibrated_test_ece": expected_calibration_error(
                        (test_logits / temperature).softmax(-1)[..., 1].mean(-1),
                        test["labels"],
                    ),
                    "frame_video_rule_disagreement": test_raw[
                        "frame_video_rule_disagreement"
                    ],
                    "mean_abs_logit": test_raw["mean_abs_logit"],
                }
            )
    return rows, probabilities_by_condition


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_seed_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["condition"],
            row["split"],
            row["checkpoint"],
            row["head"],
            row["features"],
        )
        grouped[key].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        result = dict(zip(("condition", "split", "checkpoint", "head", "features"), key))
        result["seeds"] = len(values)
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "auc",
            "f1",
            "false_positive_rate",
            "false_negative_rate",
            "cross_entropy",
            "correct_probability",
            "mean_frame_logit_margin",
        ):
            numeric = [float(row[metric]) for row in values if row[metric] is not None]
            result[f"{metric}_mean"] = float(np.mean(numeric)) if numeric else None
            result[f"{metric}_std"] = float(np.std(numeric)) if numeric else None
        output.append(result)
    return output


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_root = Path(args.runs_root).expanduser().resolve()
    caches = {
        name: load_npz(path)
        for name, path in (
            ("blur_k9", args.blur_features),
            ("jpeg_q30", args.jpeg_features),
            ("resize_25", args.resize_features),
        )
    }
    seeds = tuple(args.seeds)
    # Cross-head diagnosis across available seeds.
    cross_rows = []
    primary = {}
    for seed in seeds:
        for _, method in (
            ("uniform", "uniform"),
            ("clean_only", "clean_only"),
        ):
            checkpoint_path = runs_root / f"{method}_seed{seed}" / "best.pt"
            if not checkpoint_path.exists():
                continue
            model, checkpoint = load_model(checkpoint_path, device)
            original_head = head_original(checkpoint, device)
            checkpoints = {
                method: {
                    "model": model,
                    "original_head": original_head,
                    "checkpoint": checkpoint,
                }
            }
            condition_bundles = {}
            for condition, data in caches.items():
                indices = {
                    split: split_indices(checkpoint, split, data["video_ids"])
                    for split in ("train", "validation", "test")
                }
                condition_bundles[condition] = {
                    split: feature_bundle(model, data, index, device)
                    for split, index in indices.items()
                }
            rows = cross_head_diagnosis(
                condition_bundles, checkpoints, args.threshold
            )
            for row in rows:
                row["seed"] = seed
            cross_rows.extend(rows)
            if seed == args.primary_seed and method == "uniform":
                primary = {
                    condition: condition_bundles[condition]
                    for condition in condition_bundles
                }
    write_csv(output_dir / "cross_head_seed.csv", cross_rows)
    seed_aggregate = aggregate_seed_rows(cross_rows)
    write_csv(output_dir / "cross_head_summary.csv", seed_aggregate)

    # Corrected reverse/add interventions for R2 and R3.
    selection_validation = []
    selection_test = []
    interpolation_validation = []
    interpolation_test = []
    residual_metrics = []
    calibration_rows = []
    if not primary:
        raise RuntimeError("No primary checkpoint bundles were created")
    for condition, splits in primary.items():
        model, checkpoint = load_model(
            runs_root / f"uniform_seed{args.primary_seed}" / "best.pt", device
        )
        original_head = head_original(checkpoint, device)
        for split, bundle in splits.items():
            for head_name, head in (
                ("original", original_head),
                ("adapter", model.head),
            ):
                metrics = residual_direction_metrics(model, bundle, head)
                for residual_name, values in metrics.items():
                    for group_name in ("all", "real", "fake"):
                        group_values = (
                            values
                            if group_name == "all"
                            else values[group_name]
                        )
                        nested = {
                            key: value
                            for key, value in group_values.items()
                            if not isinstance(value, dict)
                        }
                        residual_metrics.append(
                            {
                                "condition": condition,
                                "split": split,
                                "head": head_name,
                                "residual": residual_name,
                                "group": group_name,
                                **nested,
                            }
                        )
        for residual_name in ("R2", "R3"):
            for head_name, head in (
                ("original", original_head),
                ("adapter", model.head),
            ):
                valid, test_rows = single_dimension_experiments(
                    model,
                    splits["train"],
                    splits["validation"],
                    splits["test"],
                    residual_name,
                    head,
                    args.threshold,
                    args.primary_seed,
                )
                for row in valid:
                    selection_validation.append(
                        {
                            "condition": condition,
                            "residual": residual_name,
                            "head": head_name,
                            **row,
                        }
                    )
                for row in test_rows:
                    selection_test.append(
                        {
                            "condition": condition,
                            "residual": residual_name,
                            "head": head_name,
                            **row,
                        }
                    )
        for residual_name, start_name, head_name, head in (
            ("R1", "zd", "original", original_head),
            ("R1", "za", "adapter", model.head),
            ("R2", "za", "original", original_head),
            ("R2", "za", "adapter", model.head),
            ("R3", "za", "original", original_head),
            ("R3", "za", "adapter", model.head),
        ):
            start_train = splits["train"][start_name]
            start_validation = splits["validation"][start_name]
            start_test = splits["test"][start_name]
            valid = interpolate_residual(
                start_validation,
                splits["validation"][residual_name],
                ALPHAS,
                head,
                splits["validation"]["labels"],
                args.threshold,
            )
            test = interpolate_residual(
                start_test,
                splits["test"][residual_name],
                ALPHAS,
                head,
                splits["test"]["labels"],
                args.threshold,
            )
            for row in valid:
                interpolation_validation.append(
                    {
                        "condition": condition,
                        "residual": residual_name,
                        "start": start_name,
                        "head": head_name,
                        **row,
                    }
                )
            for row in test:
                interpolation_test.append(
                    {
                        "condition": condition,
                        "residual": residual_name,
                        "start": start_name,
                        "head": head_name,
                        **row,
                    }
                )
        # Calibration on raw GenD and Adapter.
        calibration_rows_for_condition = []
        for name, head, feature_name in (
            ("GenD", original_head, "zd"),
            ("Adapter", model.head, "za"),
        ):
            validation_logits = head(splits["validation"][feature_name])
            test_logits = head(splits["test"][feature_name])
            temperature = fit_temperature(
                validation_logits, splits["validation"]["labels"]
            )
            thresholds = np.linspace(0.05, 0.95, 91)
            validation_probability = probabilities(
                validation_logits / temperature
            ).cpu().numpy()
            validation_labels = splits["validation"]["labels"].cpu().numpy()
            calibrated_threshold = max(
                thresholds,
                key=lambda value: binary_metrics(
                    validation_labels,
                    validation_probability,
                    float(value),
                )["balanced_accuracy"],
            )
            test_raw = classification_report(
                test_logits, splits["test"]["labels"], args.threshold
            )
            test_cal = classification_report(
                test_logits / temperature,
                splits["test"]["labels"],
                float(calibrated_threshold),
            )
            calibration_rows_for_condition.append(
                {
                    "condition": condition,
                    "model": name,
                    "temperature": temperature,
                    "calibrated_threshold": float(calibrated_threshold),
                    "raw_accuracy": test_raw["accuracy"],
                    "raw_balanced_accuracy": test_raw["balanced_accuracy"],
                    "raw_auc": test_raw["auc"],
                    "raw_nll": test_raw["cross_entropy"],
                    "raw_ece": expected_calibration_error(
                        test_logits.softmax(-1)[..., 1].mean(-1),
                        splits["test"]["labels"],
                    ),
                    "calibrated_accuracy": test_cal["accuracy"],
                    "calibrated_balanced_accuracy": test_cal[
                        "balanced_accuracy"
                    ],
                    "calibrated_auc": test_cal["auc"],
                    "calibrated_nll": test_cal["cross_entropy"],
                    "calibrated_ece": expected_calibration_error(
                        (test_logits / temperature).softmax(-1)[..., 1].mean(-1),
                        splits["test"]["labels"],
                    ),
                    "frame_video_rule_disagreement": test_raw[
                        "frame_video_rule_disagreement"
                    ],
                    "mean_abs_logit": test_raw["mean_abs_logit"],
                }
            )
        calibration_rows.extend(calibration_rows_for_condition)
        # Record interpolation parameters selected only on validation.
        for key in sorted(
            {(row["residual"], row["start"], row["head"]) for row in interpolation_validation if row["condition"] == condition}
        ):
            candidates = [
                row
                for row in interpolation_validation
                if row["condition"] == condition
                and row["residual"] == key[0]
                and row["start"] == key[1]
                and row["head"] == key[2]
            ]
            chosen = validation_choice(candidates, "alpha")
            selected_test = [
                row
                for row in interpolation_test
                if row["condition"] == condition
                and row["residual"] == key[0]
                and row["start"] == key[1]
                and row["head"] == key[2]
                and row["alpha"] == chosen["alpha"]
            ][0]
            # Save chosen in validation table via extra keys in a compact CSV.
            for row in interpolation_validation:
                if (
                    row["condition"] == condition
                    and row["residual"] == key[0]
                    and row["start"] == key[1]
                    and row["head"] == key[2]
                ):
                    row["selected"] = row["alpha"] == chosen["alpha"]
            for row in interpolation_test:
                if (
                    row["condition"] == condition
                    and row["residual"] == key[0]
                    and row["start"] == key[1]
                    and row["head"] == key[2]
                ):
                    row["selected"] = row["alpha"] == chosen["alpha"]
    write_csv(output_dir / "residual_metrics.csv", residual_metrics)
    write_csv(output_dir / "selection_validation.csv", selection_validation)
    write_csv(output_dir / "selection_test.csv", selection_test)
    write_csv(output_dir / "interpolation_validation.csv", interpolation_validation)
    write_csv(output_dir / "interpolation_test.csv", interpolation_test)
    write_csv(output_dir / "calibration.csv", calibration_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            jsonable(
                {
                    "checkpoint": str(
                        (runs_root / f"uniform_seed{args.primary_seed}" / "best.pt").resolve()
                    ),
                    "threshold": args.threshold,
                    "alphas": ALPHAS,
                    "ratios": RATIOS,
                    "cross_tabulation": "See CSV outputs.",
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
