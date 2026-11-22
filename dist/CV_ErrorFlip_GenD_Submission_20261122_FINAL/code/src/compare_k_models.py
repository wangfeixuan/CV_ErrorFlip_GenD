from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reference-k", type=int, default=2)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def group_members(runs: Path) -> tuple[dict[str, list[int]], list[str]]:
    manifest = sorted(runs.glob("*/split_manifest.csv"))[0]
    rows = [row for row in read_csv(manifest) if row["split"] == "test"]
    video_ids = sorted({row["video_id"] for row in rows})
    index = {video_id: position for position, video_id in enumerate(video_ids)}
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(index[row["video_id"]])
    return groups, video_ids


def load_accuracy_by_train_k(
    runs: Path, reference_k: int
) -> dict[int, np.ndarray]:
    samples: dict[int, list[np.ndarray]] = defaultdict(list)
    for path in sorted(runs.glob("k*_random_seed*/eval_test.per_video.csv")):
        result_path = path.with_name("eval_test.json")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        train_k = int(result.get("training_blurred_frames", 2))
        rows = read_csv(path)
        video_ids = sorted({row["video_id"] for row in rows})
        index = {video_id: position for position, video_id in enumerate(video_ids)}
        correct: dict[str, list[bool]] = defaultdict(list)
        for row in rows:
            if row["mask_bits"].count("1") != reference_k:
                continue
            target = row["label"] == "fake"
            predicted = float(row["adapter_probability"]) >= 0.5
            correct[row["video_id"]].append(predicted == target)
        values = np.zeros(len(video_ids), dtype=float)
        for video_id in video_ids:
            values[index[video_id]] = np.mean(correct[video_id])
        samples[train_k].append(values)
    return {
        train_k: np.mean(seed_values, axis=0)
        for train_k, seed_values in samples.items()
    }


def main() -> None:
    args = parse_args()
    runs = Path(args.runs_dir)
    groups, video_ids = group_members(runs)
    values = load_accuracy_by_train_k(runs, args.reference_k)
    reference = values[args.reference_k]
    group_arrays = [np.asarray(groups[name], dtype=np.int64) for name in sorted(groups)]
    rng = np.random.default_rng(args.seed)
    rows = []
    for train_k in sorted(values):
        difference = values[train_k] - reference
        bootstrap_values = np.empty(args.bootstrap, dtype=float)
        for iteration in range(args.bootstrap):
            selected = rng.integers(0, len(group_arrays), len(group_arrays))
            index = np.concatenate([group_arrays[value] for value in selected])
            bootstrap_values[iteration] = difference[index].mean()
        lower, upper = np.quantile(bootstrap_values, [0.025, 0.975])
        rows.append(
            {
                "condition": "blur_k9",
                "metric": f"test_k{args.reference_k}_accuracy",
                "train_k": train_k,
                "reference_train_k": args.reference_k,
                "mean_accuracy": float(values[train_k].mean()),
                "reference_mean_accuracy": float(reference.mean()),
                "difference_vs_reference": float(difference.mean()),
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "bootstrap_groups": len(groups),
                "videos": len(video_ids),
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\n".join(str(row) for row in rows))


if __name__ == "__main__":
    main()
