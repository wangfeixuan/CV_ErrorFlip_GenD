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
    return parser.parse_args()


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_arrays(runs: Path) -> tuple[np.ndarray, dict, list[str]]:
    files = sorted(runs.glob("*/eval_test*.csv"))
    if not files:
        raise FileNotFoundError("No per-video evaluation CSV files")
    observations: dict[
        tuple[str, str],
        list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ] = defaultdict(list)
    labels_by_video: dict[str, bool] = {}
    for path in files:
        result_path = path.with_name(path.name.replace(".per_video.csv", ".json"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        method = result["method"]
        condition = result.get("condition")
        if not condition:
            condition = (
                "blur_k9"
                if result_path.name == "eval_test.json"
                else Path(result.get("feature_file", result_path.stem)).stem
            )
        rows = load_csv(path)
        videos = sorted({row["video_id"] for row in rows})
        video_index = {video: index for index, video in enumerate(videos)}
        valid_masks = sorted(
            {
                int(row["mask_id"])
                for row in rows
                if row["mask_bits"].count("1") == 2
            }
        )
        mask_index = {mask: index for index, mask in enumerate(valid_masks)}
        shape = (len(videos), len(valid_masks))
        base = np.zeros(shape, dtype=bool)
        adapter = np.zeros(shape, dtype=bool)
        base_clean = np.zeros(len(videos), dtype=bool)
        adapter_clean = np.zeros(len(videos), dtype=bool)
        label = np.zeros(len(videos), dtype=bool)
        seen = np.zeros(shape, dtype=bool)
        for row in rows:
            mask = int(row["mask_id"])
            if mask not in mask_index:
                continue
            index, mask_id = video_index[row["video_id"]], mask_index[mask]
            target = row["label"] == "fake"
            label[index] = target
            labels_by_video[row["video_id"]] = target
            base_correct = (float(row["base_probability"]) >= 0.5) == target
            adapter_correct = (
                (float(row["adapter_probability"]) >= 0.5) == target
            )
            base[index, mask_id] = base_correct
            adapter[index, mask_id] = adapter_correct
            seen[index, mask_id] = True
        if not seen.all():
            raise ValueError(f"Incomplete mask matrix in {path}")
        for row in rows:
            if int(row["mask_id"]) != 0:
                continue
            index = video_index[row["video_id"]]
            target = row["label"] == "fake"
            base_clean[index] = (float(row["base_probability"]) >= 0.5) == target
            adapter_clean[index] = (
                (float(row["adapter_probability"]) >= 0.5) == target
            )
        observations[(condition, method)].append(
            (int(result["seed"]), base, adapter, base_clean, adapter_clean)
        )
    return observations, labels_by_video, files


def group_map(runs: Path) -> dict[str, str]:
    manifests = sorted(runs.glob("*/split_manifest.csv"))
    if not manifests:
        raise FileNotFoundError("Missing split manifest")
    return {
        row["video_id"]: row["group_id"]
        for row in load_csv(manifests[0])
        if row["split"] == "test"
    }


def averaged_correctness(
    entries: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seeds = list(entries)
    base = np.mean(
        [entry[1].mean(axis=1) for entry in seeds],
        axis=0,
    )
    adapter = np.mean(
        [entry[2].mean(axis=1) for entry in seeds],
        axis=0,
    )
    base_clean = np.mean([entry[3] for entry in seeds], axis=0)
    adapter_clean = np.mean([entry[4] for entry in seeds], axis=0)
    return base, adapter, base_clean, adapter_clean


def bootstrap_difference(
    difference: np.ndarray,
    group_members: dict[str, list[int]],
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    names = sorted(group_members)
    groups = [np.asarray(group_members[name], dtype=np.int64) for name in names]
    values = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selected])
        values[iteration] = difference[indices].mean()
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(difference.mean()), float(lower), float(upper)


def main() -> None:
    args = parse_args()
    runs = Path(args.runs_dir)
    observations, labels_by_video, files = build_arrays(runs)
    groups = group_map(runs)
    video_ids = sorted(labels_by_video)
    if set(video_ids) != set(groups):
        missing = sorted(set(video_ids) ^ set(groups))
        raise ValueError(f"Manifest/video mismatch: {missing[:5]}")
    group_members: dict[str, list[int]] = defaultdict(list)
    for index, video_id in enumerate(video_ids):
        group_members[groups[video_id]].append(index)
    rng = np.random.default_rng(args.seed)

    rows = []
    for (condition, method), entries in sorted(observations.items()):
        entries = sorted(entries)
        if len({len(entry[2]) for entry in entries}) != 1:
            raise ValueError(f"Seed video sets differ for {condition}/{method}")
        base, adapter, base_clean, adapter_clean = averaged_correctness(entries)
        uniform_entry = observations.get((condition, "uniform"))
        if uniform_entry is None:
            raise ValueError(f"Missing uniform baseline for {condition}")
        _, uniform, _, _ = averaged_correctness(sorted(uniform_entry))
        comparisons = {
            "adapter_vs_base_local": adapter - base,
            "adapter_vs_base_clean": adapter_clean - base_clean,
            "method_vs_uniform_local": adapter - uniform,
        }
        for comparison, difference in comparisons.items():
            point, lower, upper = bootstrap_difference(
                difference,
                group_members,
                args.bootstrap,
                rng,
            )
            rows.append(
                {
                    "condition": condition,
                    "method": method,
                    "comparison": comparison,
                    "difference": point,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "bootstrap_groups": len(group_members),
                }
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
