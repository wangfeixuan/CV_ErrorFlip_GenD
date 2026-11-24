from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from .gend_model import choose_device
from .metrics import binary_metrics
from .residual_diagnostics import load_npz, split_indices
from .residual_stage2 import (
    ALPHAS,
    RATIOS,
    classification_report,
    feature_bundle,
    head_original,
    load_model,
    probabilities,
    reverse_intervention_start,
    selector_scores,
    transition_counts,
)


BLOCK_SIZES = (16, 32)
SINGLE_SELECTORS = (
    "random",
    "magnitude",
    "classifier",
    "signed_discriminative",
    "discriminative",
)
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corrected Stage 2 residual, subspace and predictability audit."
    )
    parser.add_argument("--blur-features", required=True)
    parser.add_argument("--jpeg-features", required=True)
    parser.add_argument("--resize-features", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--runs-root")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--random-repeats", type=int, default=24)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summary_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
) -> dict:
    report = classification_report(logits, labels, threshold)
    return {
        "accuracy": report["accuracy"],
        "balanced_accuracy": report["balanced_accuracy"],
        "auc": report["auc"],
        "f1": report["f1"],
        "false_positive_rate": report["false_positive_rate"],
        "false_negative_rate": report["false_negative_rate"],
        "real_recall": (
            1.0 - report["false_positive_rate"]
            if report["false_positive_rate"] is not None
            else None
        ),
        "fake_recall": (
            1.0 - report["false_negative_rate"]
            if report["false_negative_rate"] is not None
            else None
        ),
        "cross_entropy": report["cross_entropy"],
        "correct_probability": report["correct_probability"],
        "mean_frame_logit_margin": report["mean_frame_logit_margin"],
        "frame_video_rule_disagreement": report[
            "frame_video_rule_disagreement"
        ],
        "mean_abs_logit": report["mean_abs_logit"],
        "probability": report["probability"],
    }


@torch.inference_mode()
def evaluate_mask(
    start: torch.Tensor,
    residual: torch.Tensor,
    mask: torch.Tensor,
    head: torch.nn.Linear,
    labels: torch.Tensor,
    direction: str,
    threshold: float,
) -> tuple[dict, torch.Tensor]:
    coefficient = mask.to(start.dtype)
    delta = coefficient * residual
    if direction == "reverse":
        delta = -delta
    elif direction != "add":
        raise ValueError(direction)
    features = torch.nn.functional.normalize(start + delta, dim=-1)
    logits = head(features)
    return summary_metrics(logits, labels, threshold), probabilities(logits)


def mask_from_indices(indices: torch.Tensor, dimension: int) -> torch.Tensor:
    mask = torch.zeros(dimension, dtype=torch.bool, device=indices.device)
    mask[indices] = True
    return mask


def top_k_mask(score: torch.Tensor, count: int) -> torch.Tensor:
    count = max(1, min(len(score), int(count)))
    return mask_from_indices(torch.topk(score, count).indices, len(score))


def class_balanced_direction_score(
    directional: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    real = directional[~labels.bool()].mean(dim=(0, 1))
    fake = directional[labels.bool()].mean(dim=(0, 1))
    return 0.5 * real + 0.5 * fake


def scores_with_balances(
    head: torch.nn.Linear,
    bundle: dict,
    residual_name: str,
) -> dict[str, torch.Tensor]:
    scores = selector_scores(None, bundle, residual_name, head)
    residual = bundle[residual_name]
    labels = bundle["labels"]
    reference = bundle["zd"] if residual_name == "R1" else bundle["za"]
    direction = (
        head.weight[1] - head.weight[0]
    ).detach()
    sign = torch.where(labels.bool(), 1.0, -1.0)
    gradient = sign[:, None, None] * direction[None, None, :]
    effective = gradient - (
        gradient * reference
    ).sum(dim=-1, keepdim=True) * reference
    directional = effective * residual
    scores["signed_class_balanced"] = class_balanced_direction_score(
        directional, labels
    )
    return scores


def random_mask(
    dimension: int,
    count: int,
    seed: int,
    energy: torch.Tensor | None = None,
    target_energy: float | None = None,
    candidates: int = 96,
) -> tuple[torch.Tensor, float | None]:
    rng = np.random.default_rng(seed)
    if target_energy is None or energy is None:
        indices = torch.from_numpy(
            rng.choice(dimension, size=count, replace=False)
        ).to(torch.long)
        return mask_from_indices(indices, dimension), None
    energy_np = energy.detach().cpu().numpy()
    best_indices = None
    best_error = np.inf
    for _ in range(candidates):
        indices = rng.choice(dimension, size=count, replace=False)
        error = abs(float(energy_np[indices].sum()) - target_energy)
        if error < best_error:
            best_error = error
            best_indices = indices
    mask = mask_from_indices(
        torch.from_numpy(best_indices).to(torch.long),
        dimension,
    )
    actual = float((energy * mask).sum())
    return mask, abs(actual - target_energy) / max(abs(target_energy), 1e-12)


def metrics_mean_std(rows: list[dict], metric: str) -> tuple[float, float]:
    values = np.asarray([float(row[metric]) for row in rows], dtype=float)
    return float(values.mean()), float(values.std(ddof=0))


@torch.inference_mode()
def random_control_summary(
    bundles: dict[str, dict],
    residual_name: str,
    ratios: dict[str, float],
    head: torch.nn.Linear,
    threshold: float,
    repeats: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    validation_rows = []
    test_rows = []
    train = bundles["train"]
    validation = bundles["validation"]
    test = bundles["test"]
    scores = scores_with_balances(head, train, residual_name)
    dimension = len(scores["magnitude"])
    energy = train[residual_name].square().mean(dim=(0, 1))
    for selector in SINGLE_SELECTORS:
        ratio = ratios[selector]
        count = max(1, min(dimension, int(round(dimension * ratio))))
        if selector == "random":
            selector_mask, _ = random_mask(dimension, count, seed + count)
            target_energy = None
        else:
            selector_mask = top_k_mask(scores[selector], count)
            target_energy = float((energy * selector_mask).sum())
        valid_add, _ = evaluate_mask(
            validation["za"],
            validation[residual_name],
            selector_mask,
            head,
            validation["labels"],
            "add",
            threshold,
        )
        test_add, _ = evaluate_mask(
            test["za"],
            test[residual_name],
            selector_mask,
            head,
            test["labels"],
            "add",
            threshold,
        )
        valid_random = []
        test_random = []
        energy_errors = []
        for repeat in range(repeats):
            random_control, relative_error = random_mask(
                dimension,
                count,
                seed + repeat * 104729 + count * 31,
                energy=energy,
                target_energy=target_energy,
            )
            valid_metrics, _ = evaluate_mask(
                validation["za"],
                validation[residual_name],
                random_control,
                head,
                validation["labels"],
                "add",
                threshold,
            )
            test_metrics, _ = evaluate_mask(
                test["za"],
                test[residual_name],
                random_control,
                head,
                test["labels"],
                "add",
                threshold,
            )
            valid_random.append(valid_metrics)
            test_random.append(test_metrics)
            if relative_error is not None:
                energy_errors.append(relative_error)
        for split, selected, controls, output_rows in (
            ("validation", valid_add, valid_random, validation_rows),
            ("test", test_add, test_random, test_rows),
        ):
            random_ba_mean, random_ba_std = metrics_mean_std(
                controls, "balanced_accuracy"
            )
            random_acc_mean, random_acc_std = metrics_mean_std(
                controls, "accuracy"
            )
            random_margin_mean, random_margin_std = metrics_mean_std(
                controls, "mean_frame_logit_margin"
            )
            output_rows.append(
                {
                    "split": split,
                    "residual": residual_name,
                    "ratio": ratio,
                    "selected_dimensions": int(selector_mask.sum()),
                    "selector": selector,
                    "selector_balanced_accuracy": selected[
                        "balanced_accuracy"
                    ],
                    "selector_accuracy": selected["accuracy"],
                    "selector_margin": selected[
                        "mean_frame_logit_margin"
                    ],
                    "random_balanced_accuracy_mean": random_ba_mean,
                    "random_balanced_accuracy_std": random_ba_std,
                    "random_accuracy_mean": random_acc_mean,
                    "random_accuracy_std": random_acc_std,
                    "random_margin_mean": random_margin_mean,
                    "random_margin_std": random_margin_std,
                    "energy_matching_relative_error_mean": (
                        float(np.mean(energy_errors)) if energy_errors else None
                    ),
                    "energy_matching_relative_error_max": (
                        float(np.max(energy_errors)) if energy_errors else None
                    ),
                    "random_repeats": repeats,
                }
            )
    return validation_rows, test_rows


def classification_cross_tab(
    base_probability: torch.Tensor,
    new_probability: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
) -> dict[str, int | float]:
    transitions = transition_counts(
        base_probability, new_probability, labels, threshold
    )
    return {
        "wrong_to_correct": transitions["wrong_to_correct"],
        "correct_to_wrong": transitions["correct_to_wrong"],
        "net_correction_gain": transitions["net_correction_gain"],
        "real_wrong_to_correct": transitions["real_wrong_to_correct"],
        "real_correct_to_wrong": transitions["real_correct_to_wrong"],
        "fake_wrong_to_correct": transitions["fake_wrong_to_correct"],
        "fake_correct_to_wrong": transitions["fake_correct_to_wrong"],
    }


@torch.inference_mode()
def single_selector_experiments(
    bundles: dict[str, dict],
    residual_name: str,
    head: torch.nn.Linear,
    threshold: float,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    train = bundles["train"]
    validation = bundles["validation"]
    test = bundles["test"]
    scores = scores_with_balances(head, train, residual_name)
    dimension = len(scores["magnitude"])
    validation_rows = []
    masks = {}
    for ratio in RATIOS:
        count = max(1, min(dimension, int(round(dimension * ratio))))
        for selector in SINGLE_SELECTORS:
            if selector == "random":
                mask, _ = random_mask(dimension, count, seed + count * 13)
            else:
                mask = top_k_mask(scores[selector], count)
            masks[(selector, ratio)] = mask
            add, _ = evaluate_mask(
                validation["za"],
                validation[residual_name],
                mask,
                head,
                validation["labels"],
                "add",
                threshold,
            )
            reverse, _ = evaluate_mask(
                reverse_intervention_start(validation, residual_name),
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
                    "add_accuracy": add["accuracy"],
                    "add_balanced_accuracy": add["balanced_accuracy"],
                    "add_auc": add["auc"],
                    "add_margin": add["mean_frame_logit_margin"],
                    "reverse_accuracy": reverse["accuracy"],
                    "reverse_balanced_accuracy": reverse["balanced_accuracy"],
                    "reverse_auc": reverse["auc"],
                    "reverse_margin": reverse["mean_frame_logit_margin"],
                }
            )
    selected = {}
    for selector in SINGLE_SELECTORS:
        candidates = [
            row for row in validation_rows if row["selector"] == selector
        ]
        selected[selector] = max(
            candidates,
            key=lambda row: (
                row["add_balanced_accuracy"] or 0.0,
                row["add_margin"],
            ),
        )
    base_probability = probabilities(head(test["za"]))
    reverse_start = reverse_intervention_start(test, residual_name)
    reverse_base_probability = probabilities(head(reverse_start))
    test_rows = []
    for selector, choice in selected.items():
        mask = masks[(selector, choice["ratio"])]
        add, add_probability = evaluate_mask(
            test["za"],
            test[residual_name],
            mask,
            head,
            test["labels"],
            "add",
            threshold,
        )
        reverse, reverse_probability = evaluate_mask(
            reverse_start,
            test[residual_name],
            mask,
            head,
            test["labels"],
            "reverse",
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
                "add_margin": add["mean_frame_logit_margin"],
                "add_false_positive_rate": add["false_positive_rate"],
                "add_false_negative_rate": add["false_negative_rate"],
                "reverse_accuracy": reverse["accuracy"],
                "reverse_balanced_accuracy": reverse["balanced_accuracy"],
                "reverse_auc": reverse["auc"],
                "reverse_margin": reverse["mean_frame_logit_margin"],
                "reverse_false_positive_rate": reverse[
                    "false_positive_rate"
                ],
                "reverse_false_negative_rate": reverse[
                    "false_negative_rate"
                ],
                **{
                    f"add_{key}": value
                    for key, value in classification_cross_tab(
                        base_probability,
                        add_probability,
                        test["labels"],
                        threshold,
                    ).items()
                },
                **{
                    f"reverse_{key}": value
                    for key, value in classification_cross_tab(
                        reverse_base_probability,
                        reverse_probability,
                        test["labels"],
                        threshold,
                    ).items()
                },
            }
        )
    return validation_rows, test_rows, selected


def contiguous_block_scores(score: torch.Tensor, block_size: int) -> torch.Tensor:
    dimension = len(score)
    blocks = dimension // block_size
    return score[: blocks * block_size].reshape(blocks, block_size).sum(dim=1)


def blocks_to_mask(
    block_indices: torch.Tensor,
    block_size: int,
    dimension: int,
) -> torch.Tensor:
    mask = torch.zeros(dimension, dtype=torch.bool, device=block_indices.device)
    for block in block_indices.tolist():
        start = block * block_size
        mask[start : min(start + block_size, dimension)] = True
    return mask


@torch.inference_mode()
def finite_difference_block_rows(
    bundles: dict[str, dict],
    residual_name: str,
    head: torch.nn.Linear,
    threshold: float,
) -> list[dict]:
    """Build candidate subspaces from actual validation interventions.

    Every block is evaluated through the real L2 normalization and head rather
    than through a gradient approximation. The test split is not used to rank
    blocks or choose the number of selected blocks.
    """
    validation = bundles["validation"]
    test = bundles["test"]
    dimension = validation["za"].shape[-1]
    base_validation = summary_metrics(
        head(validation["za"]), validation["labels"], threshold
    )
    base_test = summary_metrics(head(test["za"]), test["labels"], threshold)
    rows = []
    for block_size in BLOCK_SIZES:
        blocks = dimension // block_size
        validation_blocks = []
        for block in range(blocks):
            mask = blocks_to_mask(
                torch.tensor([block]), block_size, dimension
            )
            metrics, _ = evaluate_mask(
                validation["za"],
                validation[residual_name],
                mask,
                head,
                validation["labels"],
                "add",
                threshold,
            )
            validation_blocks.append(
                {
                    "block": block,
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "margin_delta": metrics["mean_frame_logit_margin"]
                    - base_validation["mean_frame_logit_margin"],
                }
            )
        ranked = sorted(
            validation_blocks,
            key=lambda row: (row["margin_delta"], row["balanced_accuracy"]),
            reverse=True,
        )
        candidates = []
        for selected_blocks in (1, 2, 4, min(8, blocks)):
            if selected_blocks <= 0:
                continue
            block_indices = torch.tensor(
                [row["block"] for row in ranked[:selected_blocks]]
            )
            mask = blocks_to_mask(block_indices, block_size, dimension)
            validation_metrics, _ = evaluate_mask(
                validation["za"],
                validation[residual_name],
                mask,
                head,
                validation["labels"],
                "add",
                threshold,
            )
            candidates.append(
                {
                    "selector": "finite_difference_block",
                    "residual": residual_name,
                    "block_size": block_size,
                    "blocks_selected": selected_blocks,
                    "selected_dimensions": int(mask.sum()),
                    "validation_accuracy": validation_metrics["accuracy"],
                    "validation_balanced_accuracy": validation_metrics[
                        "balanced_accuracy"
                    ],
                    "validation_margin_delta": validation_metrics[
                        "mean_frame_logit_margin"
                    ]
                    - base_validation["mean_frame_logit_margin"],
                    "selected_blocks": ",".join(
                        str(block) for block in block_indices.tolist()
                    ),
                    "_mask": mask,
                }
            )
        chosen = max(
            candidates,
            key=lambda row: (
                row["validation_balanced_accuracy"] or 0.0,
                row["validation_margin_delta"],
            ),
        )
        test_metrics, _ = evaluate_mask(
            test["za"],
            test[residual_name],
            chosen["_mask"],
            head,
            test["labels"],
            "add",
            threshold,
        )
        row = {
            key: value
            for key, value in chosen.items()
            if key != "_mask"
        }
        row.update(
            {
                "test_accuracy": test_metrics["accuracy"],
                "test_balanced_accuracy": test_metrics[
                    "balanced_accuracy"
                ],
                "test_auc": test_metrics["auc"],
                "test_margin_delta": test_metrics[
                    "mean_frame_logit_margin"
                ]
                - base_test["mean_frame_logit_margin"],
                "test_false_positive_rate": test_metrics[
                    "false_positive_rate"
                ],
                "test_false_negative_rate": test_metrics[
                    "false_negative_rate"
                ],
            }
        )
        rows.append(row)
    return rows


@torch.inference_mode()
def block_experiments(
    bundles: dict[str, dict],
    residual_name: str,
    head: torch.nn.Linear,
    threshold: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    train = bundles["train"]
    validation = bundles["validation"]
    test = bundles["test"]
    scores = scores_with_balances(head, train, residual_name)
    dimension = len(scores["magnitude"])
    validation_rows = []
    masks = {}
    for block_size in BLOCK_SIZES:
        blocks = dimension // block_size
        block_scores = {
            selector: contiguous_block_scores(scores[selector], block_size)
            for selector in (
                "magnitude",
                "classifier",
                "signed_discriminative",
                "discriminative",
            )
        }
        for ratio in RATIOS:
            chosen_blocks = max(
                1, min(blocks, int(round(dimension * ratio / block_size)))
            )
            masks[("random", block_size, ratio)] = blocks_to_mask(
                torch.from_numpy(
                    np.random.default_rng(
                        seed + chosen_blocks * 37 + block_size
                    ).choice(blocks, size=chosen_blocks, replace=False)
                ),
                block_size,
                dimension,
            )
            for selector, block_score in block_scores.items():
                masks[(selector, block_size, ratio)] = blocks_to_mask(
                    torch.topk(block_score, chosen_blocks).indices,
                    block_size,
                    dimension,
                )
            for selector in (*block_scores, "random"):
                mask = masks[(selector, block_size, ratio)]
                add, _ = evaluate_mask(
                    validation["za"],
                    validation[residual_name],
                    mask,
                    head,
                    validation["labels"],
                    "add",
                    threshold,
                )
                reverse, _ = evaluate_mask(
                    reverse_intervention_start(validation, residual_name),
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
                        "block_size": block_size,
                        "ratio": ratio,
                        "selected_dimensions": int(mask.sum()),
                        "add_accuracy": add["accuracy"],
                        "add_balanced_accuracy": add["balanced_accuracy"],
                        "add_auc": add["auc"],
                        "add_margin": add["mean_frame_logit_margin"],
                        "reverse_accuracy": reverse["accuracy"],
                        "reverse_balanced_accuracy": reverse[
                            "balanced_accuracy"
                        ],
                        "reverse_auc": reverse["auc"],
                        "reverse_margin": reverse["mean_frame_logit_margin"],
                    }
                )
    selected = {}
    for selector in (
        "random",
        "magnitude",
        "classifier",
        "signed_discriminative",
        "discriminative",
    ):
        candidates = [
            row for row in validation_rows if row["selector"] == selector
        ]
        selected[selector] = max(
            candidates,
            key=lambda row: (
                row["add_balanced_accuracy"] or 0.0,
                row["add_margin"],
            ),
        )
    base_probability = probabilities(head(test["za"]))
    reverse_base_probability = probabilities(
        head(reverse_intervention_start(test, residual_name))
    )
    test_rows = []
    for selector, choice in selected.items():
        mask = masks[(selector, choice["block_size"], choice["ratio"])]
        add, add_probability = evaluate_mask(
            test["za"],
            test[residual_name],
            mask,
            head,
            test["labels"],
            "add",
            threshold,
        )
        reverse, reverse_probability = evaluate_mask(
            reverse_intervention_start(test, residual_name),
            test[residual_name],
            mask,
            head,
            test["labels"],
            "reverse",
            threshold,
        )
        test_rows.append(
            {
                "selector": selector,
                "block_size": choice["block_size"],
                "ratio": choice["ratio"],
                "selected_dimensions": int(mask.sum()),
                "add_accuracy": add["accuracy"],
                "add_balanced_accuracy": add["balanced_accuracy"],
                "add_auc": add["auc"],
                "add_margin": add["mean_frame_logit_margin"],
                "add_false_positive_rate": add["false_positive_rate"],
                "add_false_negative_rate": add["false_negative_rate"],
                "reverse_accuracy": reverse["accuracy"],
                "reverse_balanced_accuracy": reverse["balanced_accuracy"],
                "reverse_auc": reverse["auc"],
                "reverse_margin": reverse["mean_frame_logit_margin"],
                **{
                    f"add_{key}": value
                    for key, value in classification_cross_tab(
                        base_probability,
                        add_probability,
                        test["labels"],
                        threshold,
                    ).items()
                },
                **{
                    f"reverse_{key}": value
                    for key, value in classification_cross_tab(
                        reverse_base_probability,
                        reverse_probability,
                        test["labels"],
                        threshold,
                    ).items()
                },
            }
        )
    return validation_rows, test_rows


def pca_basis(
    train_residual: torch.Tensor,
    components: int,
) -> torch.Tensor:
    matrix = (
        train_residual.reshape(-1, train_residual.shape[-1])
        .detach()
        .cpu()
        .numpy()
    )
    pca = PCA(
        n_components=min(components, matrix.shape[0], matrix.shape[1])
    )
    pca.fit(matrix)
    return torch.from_numpy(pca.components_).float()


def discriminant_direction(head: torch.nn.Linear) -> torch.Tensor:
    direction = head.weight[1] - head.weight[0]
    return direction / direction.norm().clamp_min(1e-8)


def project_residual(
    residual: torch.Tensor,
    basis: torch.Tensor,
    component: str,
) -> torch.Tensor:
    if basis.ndim == 1:
        projection = (residual * basis).sum(dim=-1, keepdim=True) * basis
    else:
        projection = (residual @ basis.T) @ basis
    return projection if component == "parallel" else residual - projection


def choose_alpha(rows: list[dict]) -> dict:
    return max(
        rows,
        key=lambda row: (
            row["balanced_accuracy"] or 0.0,
            row["mean_frame_logit_margin"],
        ),
    )


@torch.inference_mode()
def projection_grid(
    start: torch.Tensor,
    projected: torch.Tensor,
    head: torch.nn.Linear,
    labels: torch.Tensor,
    threshold: float,
) -> list[dict]:
    rows = []
    for alpha in ALPHAS:
        features = torch.nn.functional.normalize(
            start + float(alpha) * projected, dim=-1
        )
        metrics = summary_metrics(head(features), labels, threshold)
        metrics.pop("probability")
        rows.append({"alpha": alpha, **metrics})
    return rows


@torch.inference_mode()
def subspace_experiments(
    bundles: dict[str, dict],
    residual_name: str,
    head: torch.nn.Linear,
    threshold: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    train = bundles["train"]
    validation = bundles["validation"]
    test = bundles["test"]
    bases = []
    for components in (16, 32):
        basis = pca_basis(train[residual_name], components)
        bases.append((f"pca{components}_parallel", basis, "parallel"))
        bases.append((f"pca{components}_orthogonal", basis, "orthogonal"))
    direction = discriminant_direction(head)
    bases.extend(
        (
            ("discriminant_parallel", direction, "parallel"),
            ("discriminant_orthogonal", direction, "orthogonal"),
        )
    )
    validation_rows = []
    test_rows = []
    component_rows = []
    reverse_start = reverse_intervention_start(test, residual_name)
    full_mask = torch.ones(
        test["za"].shape[-1], dtype=torch.bool, device=test["za"].device
    )
    reverse_metrics, _ = evaluate_mask(
        reverse_start,
        test[residual_name],
        full_mask,
        head,
        test["labels"],
        "reverse",
        threshold,
    )
    for name, basis, component in bases:
        validation_projection = project_residual(
            validation[residual_name], basis, component
        )
        test_projection = project_residual(
            test[residual_name], basis, component
        )
        validation_grid = projection_grid(
            validation["za"],
            validation_projection,
            head,
            validation["labels"],
            threshold,
        )
        chosen = choose_alpha(validation_grid)
        test_grid = projection_grid(
            test["za"],
            test_projection,
            head,
            test["labels"],
            threshold,
        )
        test_selected = next(
            row for row in test_grid if row["alpha"] == chosen["alpha"]
        )
        for row in validation_grid:
            validation_rows.append(
                {
                    "selector": name,
                    "residual": residual_name,
                    "component": component,
                    **row,
                    "selected": row["alpha"] == chosen["alpha"],
                }
            )
        test_rows.append(
            {
                "selector": name,
                "residual": residual_name,
                "component": component,
                "alpha": chosen["alpha"],
                **{
                    key: value
                    for key, value in test_selected.items()
                    if key != "alpha"
                },
            }
        )
        component_rows.append(
            {
                "selector": name,
                "residual": residual_name,
                "component": component,
                "projected_energy_fraction": float(
                    test_projection.square().sum()
                    / test[residual_name].square().sum().clamp_min(1e-12)
                ),
                "full_reverse_accuracy": reverse_metrics["accuracy"],
                "full_reverse_balanced_accuracy": reverse_metrics[
                    "balanced_accuracy"
                ],
                "full_reverse_margin": reverse_metrics[
                    "mean_frame_logit_margin"
                ],
            }
        )
    return validation_rows, test_rows, component_rows


def flatten_features(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy().reshape(-1, values.shape[-1])


def fit_ridge_with_validation(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
    test_x: torch.Tensor,
) -> tuple[float, np.ndarray]:
    x_train = flatten_features(train_x)
    y_train = flatten_features(train_y)
    x_validation = flatten_features(validation_x)
    y_validation = flatten_features(validation_y)
    x_test = flatten_features(test_x)
    best = None
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha)
        model.fit(x_train, y_train)
        prediction = model.predict(x_validation)
        mse = float(np.mean((prediction - y_validation) ** 2))
        if best is None or mse < best[0]:
            best = (mse, alpha, model)
    _, alpha, model = best
    return alpha, model.predict(x_test)


def residual_predictability(
    bundles: dict[str, dict],
    residual_name: str,
    input_name: str,
    head: torch.nn.Linear,
    threshold: float,
) -> list[dict]:
    train = bundles["train"]
    validation = bundles["validation"]
    test = bundles["test"]
    true_test = flatten_features(test[residual_name])
    training_y = flatten_features(train[residual_name])
    mean_prediction = np.broadcast_to(
        training_y.mean(axis=0), true_test.shape
    )
    zero_prediction = np.zeros_like(true_test)
    full_alpha, full_prediction = fit_ridge_with_validation(
        train[input_name],
        train[residual_name],
        validation[input_name],
        validation[residual_name],
        test[input_name],
    )
    basis = pca_basis(train[residual_name], 32).numpy()
    y_train_projected = flatten_features(train[residual_name]) @ basis.T
    y_validation_projected = (
        flatten_features(validation[residual_name]) @ basis.T
    )
    x_train = flatten_features(train[input_name])
    x_validation = flatten_features(validation[input_name])
    x_test = flatten_features(test[input_name])
    best_subspace = None
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha)
        model.fit(x_train, y_train_projected)
        prediction = model.predict(x_validation)
        mse = float(np.mean((prediction - y_validation_projected) ** 2))
        if best_subspace is None or mse < best_subspace[0]:
            best_subspace = (mse, alpha, model)
    _, subspace_alpha, subspace_model = best_subspace
    subspace_prediction = subspace_model.predict(x_test) @ basis
    predictions = {
        "zero_residual": zero_prediction,
        "training_mean_residual": mean_prediction,
        "ridge_full_residual": full_prediction,
        "ridge_subspace_residual": subspace_prediction,
    }
    direction = discriminant_direction(head).cpu().numpy()
    mean_mse = float(np.mean((mean_prediction - true_test) ** 2))
    rows = []
    for predictor_name, prediction in predictions.items():
        tensor = torch.from_numpy(prediction.astype(np.float32)).reshape_as(
            test["za"]
        )
        metrics, _ = evaluate_mask(
            test["za"],
            tensor,
            torch.ones_like(tensor, dtype=torch.bool),
            head,
            test["labels"],
            "add",
            threshold,
        )
        mse = float(np.mean((prediction - true_test) ** 2))
        prediction_norm = np.linalg.norm(prediction, axis=1)
        true_norm = np.linalg.norm(true_test, axis=1)
        cosine = (prediction * true_test).sum(axis=1) / np.maximum(
            prediction_norm * true_norm, 1e-12
        )
        direction_error = np.abs(
            prediction @ direction - true_test @ direction
        ).mean()
        rows.append(
            {
                "residual": residual_name,
                "input": input_name,
                "predictor": predictor_name,
                "ridge_alpha": (
                    full_alpha
                    if predictor_name == "ridge_full_residual"
                    else subspace_alpha
                    if predictor_name == "ridge_subspace_residual"
                    else None
                ),
                "test_mse": mse,
                "r2_relative_mean": (
                    1.0 - mse / mean_mse if mean_mse > 0 else None
                ),
                "prediction_true_cosine_mean": float(np.nanmean(cosine)),
                "discriminant_direction_absolute_error": float(
                    direction_error
                ),
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "probability"
                },
            }
        )
    oracle_metrics = summary_metrics(
        head(reverse_intervention_start(test, residual_name)),
        test["labels"],
        threshold,
    )
    oracle_metrics.pop("probability")
    rows.append(
        {
            "residual": residual_name,
            "input": input_name,
            "predictor": "oracle_true_clean_endpoint",
            "ridge_alpha": None,
            "test_mse": 0.0,
            "r2_relative_mean": 1.0,
            "prediction_true_cosine_mean": 1.0,
            "discriminant_direction_absolute_error": 0.0,
            **oracle_metrics,
        }
    )
    return rows


def pair_masks(
    scores: dict[str, torch.Tensor],
    selector: str,
    ratio: float,
) -> torch.Tensor:
    return top_k_mask(
        scores[selector],
        max(1, int(round(len(scores["magnitude"]) * ratio))),
    )


def mask_jaccard(first: torch.Tensor, second: torch.Tensor) -> float:
    union = (first | second).sum().item()
    return float((first & second).sum().item() / union) if union else 1.0


@torch.inference_mode()
def selector_transfer(
    bundles: dict[str, dict[str, dict]],
    residual_name: str,
    head: torch.nn.Linear,
    threshold: float,
) -> list[dict]:
    rows = []
    selectors = (
        "magnitude",
        "classifier",
        "signed_discriminative",
        "discriminative",
    )
    for source in bundles:
        source_train = scores_with_balances(
            head, bundles[source]["train"], residual_name
        )
        source_validation = bundles[source]["validation"]
        source_choices = {}
        for selector in selectors:
            candidates = []
            for ratio in RATIOS:
                mask = pair_masks(source_train, selector, ratio)
                metrics, _ = evaluate_mask(
                    source_validation["za"],
                    source_validation[residual_name],
                    mask,
                    head,
                    source_validation["labels"],
                    "add",
                    threshold,
                )
                candidates.append((ratio, metrics))
            source_choices[selector] = max(
                candidates,
                key=lambda item: (
                    item[1]["balanced_accuracy"] or 0.0,
                    item[1]["mean_frame_logit_margin"],
                ),
            )
        for target in bundles:
            target_train = scores_with_balances(
                head, bundles[target]["train"], residual_name
            )
            target_test = bundles[target]["test"]
            base_probability = probabilities(head(target_test["za"]))
            for selector, (ratio, source_choice) in source_choices.items():
                source_mask = pair_masks(source_train, selector, ratio)
                target_mask = pair_masks(target_train, selector, ratio)
                metrics, probability = evaluate_mask(
                    target_test["za"],
                    target_test[residual_name],
                    target_mask,
                    head,
                    target_test["labels"],
                    "add",
                    threshold,
                )
                rows.append(
                    {
                        "source_condition": source,
                        "target_condition": target,
                        "residual": residual_name,
                        "selector": selector,
                        "ratio": ratio,
                        "source_validation_balanced_accuracy": source_choice[
                            "balanced_accuracy"
                        ],
                        "target_test_balanced_accuracy": metrics[
                            "balanced_accuracy"
                        ],
                        "target_test_accuracy": metrics["accuracy"],
                        "target_test_margin": metrics[
                            "mean_frame_logit_margin"
                        ],
                        "mask_jaccard": mask_jaccard(
                            source_mask, target_mask
                        ),
                        **classification_cross_tab(
                            base_probability,
                            probability,
                            target_test["labels"],
                            threshold,
                        ),
                    }
                )
    return rows


def bootstrap_groups(
    groups: np.ndarray,
    base_correct: np.ndarray,
    new_correct: np.ndarray,
    iterations: int,
    seed: int,
) -> dict:
    unique = np.asarray(sorted(set(groups.tolist())))
    members = {group: np.flatnonzero(groups == group) for group in unique}
    differences = new_correct.astype(float) - base_correct.astype(float)
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=float)
    for index in range(iterations):
        chosen = rng.integers(0, len(unique), size=len(unique))
        sample = np.concatenate([members[unique[item]] for item in chosen])
        values[index] = differences[sample].mean()
    return {
        "difference": float(differences.mean()),
        "ci95_lower": float(np.quantile(values, 0.025)),
        "ci95_upper": float(np.quantile(values, 0.975)),
        "groups": int(len(unique)),
        "videos": int(len(groups)),
    }


def cross_condition_bootstrap(
    groups_by_condition: dict[str, np.ndarray],
    base_correct: dict[str, np.ndarray],
    new_correct: dict[str, np.ndarray],
    iterations: int,
    seed: int,
) -> dict:
    names = list(groups_by_condition)
    unique = np.asarray(
        sorted(set(groups_by_condition[names[0]].tolist()))
    )
    members = {
        name: {
            group: np.flatnonzero(groups_by_condition[name] == group)
            for group in unique
        }
        for name in names
    }
    differences = {
        name: new_correct[name].astype(float) - base_correct[name].astype(float)
        for name in names
    }
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=float)
    for index in range(iterations):
        chosen = rng.integers(0, len(unique), size=len(unique))
        per_condition = []
        for name in names:
            sample = np.concatenate(
                [members[name][unique[item]] for item in chosen]
            )
            per_condition.append(differences[name][sample].mean())
        values[index] = float(np.mean(per_condition))
    return {
        "difference": float(
            np.mean([differences[name].mean() for name in names])
        ),
        "ci95_lower": float(np.quantile(values, 0.025)),
        "ci95_upper": float(np.quantile(values, 0.975)),
        "groups": int(len(unique)),
        "videos_per_condition": int(len(groups_by_condition[names[0]])),
        "conditions": len(names),
    }


def validation_interpolation_choice(
    bundle: dict,
    residual_name: str,
    head: torch.nn.Linear,
    threshold: float,
) -> float:
    rows = []
    labels = bundle["labels"].cpu().numpy()
    for alpha in ALPHAS:
        probability = probabilities(
            head(
                torch.nn.functional.normalize(
                    bundle["za"] + alpha * bundle[residual_name],
                    dim=-1,
                )
            )
        )
        metrics = binary_metrics(
            labels, probability.cpu().numpy(), threshold
        )
        rows.append((alpha, metrics["balanced_accuracy"] or 0.0))
    return max(rows, key=lambda item: item[1])[0]


@torch.inference_mode()
def stability_rows_for_checkpoint(
    model,
    caches: dict,
    checkpoint: dict,
    device: torch.device,
    threshold: float,
    seed: int,
) -> tuple[list[dict], dict]:
    rows = []
    masks = {}
    for condition, data in caches.items():
        indices = {
            split: split_indices(checkpoint, split, data["video_ids"])
            for split in ("train", "validation", "test")
        }
        bundles = {
            split: feature_bundle(model, data, index, device)
            for split, index in indices.items()
        }
        for residual_name in ("R2", "R3"):
            alpha = validation_interpolation_choice(
                bundles["validation"],
                residual_name,
                model.head,
                threshold,
            )
            metrics = summary_metrics(
                model.head(
                    torch.nn.functional.normalize(
                        bundles["test"]["za"]
                        + alpha * bundles["test"][residual_name],
                        dim=-1,
                    )
                ),
                bundles["test"]["labels"],
                threshold,
            )
            rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "residual": residual_name,
                    "selected_alpha": alpha,
                    "test_accuracy": metrics["accuracy"],
                    "test_balanced_accuracy": metrics[
                        "balanced_accuracy"
                    ],
                    "test_auc": metrics["auc"],
                    "test_margin": metrics["mean_frame_logit_margin"],
                }
            )
        masks[condition] = pair_masks(
            scores_with_balances(model.head, bundles["train"], "R3"),
            "signed_discriminative",
            0.10,
        )
    return rows, masks


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    caches = {
        name: load_npz(path)
        for name, path in (
            ("blur_k9", args.blur_features),
            ("jpeg_q30", args.jpeg_features),
            ("resize_25", args.resize_features),
        )
    }
    primary_model, checkpoint = load_model(Path(args.checkpoint), device)
    original_head = head_original(checkpoint, device)
    bundles = {}
    for condition, data in caches.items():
        indices = {
            split: split_indices(checkpoint, split, data["video_ids"])
            for split in ("train", "validation", "test")
        }
        bundles[condition] = {
            split: feature_bundle(primary_model, data, index, device)
            for split, index in indices.items()
        }

    single_validation = []
    single_test = []
    random_validation = []
    random_test = []
    block_validation = []
    block_test = []
    finite_difference = []
    subspace_validation = []
    subspace_test = []
    subspace_components = []
    ridge_rows = []
    class_balance_rows = []
    bootstrap_rows = []
    paired_condition_probabilities = defaultdict(dict)
    paired_condition_base = {}
    paired_condition_groups = {}
    for condition, splits in bundles.items():
        for head_name, head in (
            ("original", original_head),
            ("adapter", primary_model.head),
        ):
            for residual_name in ("R2", "R3"):
                valid, test, selected_ratios = single_selector_experiments(
                    splits,
                    residual_name,
                    head,
                    args.threshold,
                    args.seed,
                )
                single_validation.extend(
                    {
                        "condition": condition,
                        "residual": residual_name,
                        "head": head_name,
                        **row,
                    }
                    for row in valid
                )
                single_test.extend(
                    {
                        "condition": condition,
                        "residual": residual_name,
                        "head": head_name,
                        **row,
                    }
                    for row in test
                )
                if head_name == "adapter":
                    valid_controls, test_controls = random_control_summary(
                        splits,
                        residual_name,
                        {
                            selector: selected_ratios[selector]["ratio"]
                            for selector in SINGLE_SELECTORS
                        },
                        head,
                        args.threshold,
                        args.random_repeats,
                        args.seed,
                    )
                    random_validation.extend(
                        {"condition": condition, **row}
                        for row in valid_controls
                    )
                    random_test.extend(
                        {"condition": condition, **row}
                        for row in test_controls
                    )
                    block_valid, block_test_rows = block_experiments(
                        splits,
                        residual_name,
                        head,
                        args.threshold,
                        args.seed,
                    )
                    block_validation.extend(
                        {"condition": condition, **row}
                        for row in block_valid
                    )
                    block_test.extend(
                        {"condition": condition, **row}
                        for row in block_test_rows
                    )
                    finite_difference.extend(
                        {"condition": condition, **row}
                        for row in finite_difference_block_rows(
                            splits,
                            residual_name,
                            head,
                            args.threshold,
                        )
                    )
                    sub_valid, sub_test, components = subspace_experiments(
                        splits, residual_name, head, args.threshold
                    )
                    subspace_validation.extend(
                        {"condition": condition, **row}
                        for row in sub_valid
                    )
                    subspace_test.extend(
                        {"condition": condition, **row}
                        for row in sub_test
                    )
                    subspace_components.extend(
                        {"condition": condition, **row}
                        for row in components
                    )

        unweighted = scores_with_balances(
            primary_model.head, splits["train"], "R3"
        )["signed_discriminative"]
        balanced = scores_with_balances(
            primary_model.head, splits["train"], "R3"
        )["signed_class_balanced"]
        rank_correlation = (
            float(spearmanr(unweighted.cpu(), balanced.cpu()).statistic)
            if torch.std(unweighted) > 0 and torch.std(balanced) > 0
            else None
        )
        real_videos = int((~splits["train"]["labels"].bool()).sum())
        fake_videos = int(splits["train"]["labels"].bool().sum())
        for ratio in RATIOS:
            class_balance_rows.append(
                {
                    "condition": condition,
                    "ratio": ratio,
                    "real_videos": real_videos,
                    "fake_videos": fake_videos,
                    "real_frame_fraction": real_videos
                    / (real_videos + fake_videos),
                    "rank_spearman": rank_correlation,
                    "topk_jaccard": mask_jaccard(
                        pair_masks(
                            {"magnitude": unweighted},
                            "magnitude",
                            ratio,
                        ),
                        pair_masks(
                            {"magnitude": balanced},
                            "magnitude",
                            ratio,
                        ),
                    ),
                }
            )
        for input_name in ("zd", "za"):
            ridge_rows.extend(
                {"condition": condition, **row}
                for row in residual_predictability(
                    splits,
                    "R3",
                    input_name,
                    primary_model.head,
                    args.threshold,
                )
            )

        labels_np = splits["test"]["labels"].cpu().numpy().astype(bool)
        base_probability = probabilities(primary_model.head(splits["test"]["za"]))
        base_correct = (
            base_probability.cpu().numpy() >= args.threshold
        ) == labels_np
        comparisons = {}
        alpha = validation_interpolation_choice(
            splits["validation"],
            "R3",
            primary_model.head,
            args.threshold,
        )
        comparisons["R3_validation_selected_interpolation"] = probabilities(
            primary_model.head(
                torch.nn.functional.normalize(
                    splits["test"]["za"] + alpha * splits["test"]["R3"],
                    dim=-1,
                )
            )
        )
        _, prediction = fit_ridge_with_validation(
            splits["train"]["za"],
            splits["train"]["R3"],
            splits["validation"]["za"],
            splits["validation"]["R3"],
            splits["test"]["za"],
        )
        ridge_tensor = torch.from_numpy(
            prediction.astype(np.float32)
        ).reshape_as(splits["test"]["za"])
        _, comparisons["ridge_full_R3_from_za"] = evaluate_mask(
            splits["test"]["za"],
            ridge_tensor,
            torch.ones_like(ridge_tensor, dtype=torch.bool),
            primary_model.head,
            splits["test"]["labels"],
            "add",
            args.threshold,
        )
        direction = discriminant_direction(primary_model.head)
        validation_projection = project_residual(
            splits["validation"]["R3"], direction, "parallel"
        )
        test_projection = project_residual(
            splits["test"]["R3"], direction, "parallel"
        )
        direction_alpha = choose_alpha(
            projection_grid(
                splits["validation"]["za"],
                validation_projection,
                primary_model.head,
                splits["validation"]["labels"],
                args.threshold,
            )
        )["alpha"]
        comparisons["discriminant_parallel_R3"] = probabilities(
            primary_model.head(
                torch.nn.functional.normalize(
                    splits["test"]["za"]
                    + direction_alpha * test_projection,
                    dim=-1,
                )
            )
        )
        block_score = contiguous_block_scores(
            scores_with_balances(
                primary_model.head, splits["train"], "R3"
            )["signed_discriminative"],
            16,
        )
        block_mask = blocks_to_mask(
            torch.topk(block_score, max(1, int(round(0.10 * 1024 / 16)))).indices,
            16,
            splits["test"]["za"].shape[-1],
        )
        _, comparisons["block_signed_R3"] = evaluate_mask(
            splits["test"]["za"],
            splits["test"]["R3"],
            block_mask,
            primary_model.head,
            splits["test"]["labels"],
            "add",
            args.threshold,
        )
        for comparison, probability in comparisons.items():
            paired_condition_probabilities[comparison][condition] = probability
            new_correct = (
                probability.cpu().numpy() >= args.threshold
            ) == labels_np
            bootstrap_rows.append(
                {
                    "condition": condition,
                    "comparison": comparison,
                    "wrong_to_correct": int(
                        ((~base_correct) & new_correct).sum()
                    ),
                    "correct_to_wrong": int(
                        (base_correct & ~new_correct).sum()
                    ),
                    "net_correctness_delta": int(
                        (new_correct.astype(int) - base_correct.astype(int)).sum()
                    ),
                    **bootstrap_groups(
                        splits["test"]["group_ids"],
                        base_correct,
                        new_correct,
                        args.bootstrap,
                        args.seed,
                    ),
                }
            )
        paired_condition_base[condition] = base_correct
        paired_condition_groups[condition] = splits["test"]["group_ids"]

    for comparison, probabilities_by_condition in paired_condition_probabilities.items():
        if set(probabilities_by_condition) != set(bundles):
            continue
        new_correct = {}
        base_correct = {}
        for condition in bundles:
            labels_np = bundles[condition]["test"]["labels"].cpu().numpy().astype(bool)
            new_correct[condition] = (
                probabilities_by_condition[condition].cpu().numpy()
                >= args.threshold
            ) == labels_np
            base_correct[condition] = paired_condition_base[condition]
        bootstrap_rows.append(
            {
                "condition": "all_three_paired",
                "comparison": comparison,
                **cross_condition_bootstrap(
                    paired_condition_groups,
                    base_correct,
                    new_correct,
                    args.bootstrap,
                    args.seed,
                ),
            }
        )

    transfer_rows = []
    for residual_name in ("R2", "R3"):
        transfer_rows.extend(
            selector_transfer(
                bundles,
                residual_name,
                primary_model.head,
                args.threshold,
            )
        )

    seed_rows = []
    if args.runs_root:
        runs_root = Path(args.runs_root).expanduser().resolve()
        seed_masks = defaultdict(dict)
        for seed in args.seeds:
            path = runs_root / f"uniform_seed{seed}" / "best.pt"
            if not path.exists():
                continue
            model, seed_checkpoint = load_model(path, device)
            rows, masks = stability_rows_for_checkpoint(
                model,
                caches,
                seed_checkpoint,
                device,
                args.threshold,
                seed,
            )
            seed_rows.extend(rows)
            for condition, mask in masks.items():
                seed_masks[condition][seed] = mask
        for condition, by_seed in seed_masks.items():
            seeds = sorted(by_seed)
            for index, first in enumerate(seeds):
                for second in seeds[index + 1 :]:
                    seed_rows.append(
                        {
                            "seed": f"{first}_vs_{second}",
                            "condition": condition,
                            "residual": "R3_mask_stability",
                            "selected_alpha": 0.10,
                            "mask_jaccard": mask_jaccard(
                                by_seed[first], by_seed[second]
                            ),
                        }
                    )

    write_csv(output_dir / "single_selector_validation.csv", single_validation)
    write_csv(output_dir / "single_selector_test.csv", single_test)
    write_csv(output_dir / "matched_random_validation.csv", random_validation)
    write_csv(output_dir / "matched_random_test.csv", random_test)
    write_csv(output_dir / "block_validation.csv", block_validation)
    write_csv(output_dir / "block_test.csv", block_test)
    write_csv(
        output_dir / "finite_difference_subspace.csv",
        finite_difference,
    )
    write_csv(output_dir / "subspace_validation.csv", subspace_validation)
    write_csv(output_dir / "subspace_test.csv", subspace_test)
    write_csv(output_dir / "subspace_components.csv", subspace_components)
    write_csv(output_dir / "ridge_predictability.csv", ridge_rows)
    write_csv(output_dir / "class_balance_audit.csv", class_balance_rows)
    write_csv(output_dir / "bootstrap.csv", bootstrap_rows)
    write_csv(output_dir / "selector_transfer.csv", transfer_rows)
    write_csv(output_dir / "seed_stability.csv", seed_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "runs_root": (
                    str(Path(args.runs_root).resolve())
                    if args.runs_root
                    else None
                ),
                "seeds": args.seeds,
                "threshold": args.threshold,
                "alphas": ALPHAS,
                "ratios": RATIOS,
                "block_sizes": BLOCK_SIZES,
                "ridge_alphas": RIDGE_ALPHAS,
                "random_repeats": args.random_repeats,
                "bootstrap_repeats": args.bootstrap,
                "reverse_coordinate_rule": {
                    "R1": "z0 - R1 = zd",
                    "R2": "z0 - R2 = za",
                    "R3": "zc - R3 = za",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
