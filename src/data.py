from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

FRAME_RE = re.compile(
    r"^(?P<dataset>[^/]+)/(?P<label>real|fake)/frames/"
    r"(?P<video_id>[^/]+)/(?P<frame>\d+\.png)$"
)
CELEB_REAL_RE = re.compile(r"^Celeb-real__id(?P<identity>\d+)_")
FAKE_PAIR_RE = re.compile(
    r"^Celeb-synthesis__id(?P<source>\d+)_id(?P<target>\d+)_"
)


def source_group(video_id: str, label: str) -> str:
    """Return a deterministic group that never splits a source video pair."""
    if label == "fake":
        match = FAKE_PAIR_RE.match(video_id)
        if match:
            return f"fake-pair-id{match.group('source')}-id{match.group('target')}"
        return f"fake-video-{video_id}"
    match = CELEB_REAL_RE.match(video_id)
    if match:
        return f"real-id{match.group('identity')}"
    return f"real-video-{video_id}"


def index_zip(path: str | Path) -> list[dict]:
    """Index a Celeb-DF face ZIP without extracting its images."""
    grouped: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    with ZipFile(path) as archive:
        for name in archive.namelist():
            match = FRAME_RE.match(name)
            if not match:
                continue
            label = match.group("label")
            video_id = match.group("video_id")
            frame_name = match.group("frame")
            grouped[(label, video_id)].append((int(frame_name[:-4]), name))

    records: list[dict] = []
    for (label, video_id), frames in sorted(grouped.items()):
        frames.sort()
        frame_names = [name for _, name in frames]
        if len({Path(name).name for name in frame_names}) != len(frame_names):
            raise ValueError(f"Duplicate frame file name in {video_id}")
        records.append(
            {
                "video_id": video_id,
                "label": label,
                "label_index": int(label == "fake"),
                "group_id": source_group(video_id, label),
                "frame_names": frame_names,
                "frame_count": len(frame_names),
            }
        )
    if not records:
        raise ValueError(f"No Celeb-DF face frames found in {path}")
    return records


def paired_zip_indexes(
    original_zip: str | Path, blur_zip: str | Path
) -> list[dict]:
    original = index_zip(original_zip)
    blur = index_zip(blur_zip)
    original_by_key = {(r["label"], r["video_id"]): r for r in original}
    blur_by_key = {(r["label"], r["video_id"]): r for r in blur}
    if original_by_key.keys() != blur_by_key.keys():
        missing_blur = sorted(original_by_key.keys() - blur_by_key.keys())
        missing_original = sorted(blur_by_key.keys() - original_by_key.keys())
        raise ValueError(
            f"Video sets do not match: missing_blur={missing_blur[:3]}, "
            f"missing_original={missing_original[:3]}"
        )
    records = []
    for key in sorted(original_by_key):
        original_record = original_by_key[key]
        blur_record = blur_by_key[key]
        if original_record["frame_names"] != blur_record["frame_names"]:
            raise ValueError(f"Original/blur frame paths differ for {key}")
        records.append(
            {
                **original_record,
                "blur_frame_names": blur_record["frame_names"],
            }
        )
    return records


def apply_group_manifest(
    records: list[dict],
    manifest_path: str | Path,
) -> list[dict]:
    """Replace inferred source groups with audited groups from a CSV manifest."""
    path = Path(manifest_path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Group manifest is empty: {path}")
    required = {"video_id", "label", "group_id"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Group manifest is missing columns: {sorted(missing)}")

    overrides: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row["label"].strip().lower(), row["video_id"].strip())
        group_id = row["group_id"].strip()
        if not group_id:
            raise ValueError(f"Empty group_id for {key}")
        previous = overrides.setdefault(key, group_id)
        if previous != group_id:
            raise ValueError(f"Conflicting group_id values for {key}")

    record_keys = {(record["label"], record["video_id"]) for record in records}
    missing_keys = sorted(record_keys - set(overrides))
    extra_keys = sorted(set(overrides) - record_keys)
    if missing_keys or extra_keys:
        raise ValueError(
            "Group manifest/video mismatch: "
            f"missing={missing_keys[:3]}, extra={extra_keys[:3]}"
        )
    return [
        {**record, "group_id": overrides[(record["label"], record["video_id"])]}
        for record in records
    ]


def _take_groups(
    groups: list[list[dict]], target: int, rng: random.Random
) -> tuple[list[list[dict]], list[list[dict]]]:
    candidates = list(groups)
    rng.shuffle(candidates)
    selected: list[list[dict]] = []
    total = 0
    while candidates and total < target:
        fitting = [group for group in candidates if total + len(group) <= target]
        group = fitting[0] if fitting else min(candidates, key=len)
        selected.append(group)
        total += len(group)
        candidates.remove(group)
    return selected, candidates


def assign_group_splits(
    records: Iterable[dict],
    ratios: dict[str, float],
    seed: int,
) -> list[dict]:
    """Assign deterministic, label-stratified splits without video leakage."""
    records = [dict(record) for record in records]
    ratio_sum = sum(ratios.values())
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to one, got {ratio_sum}")

    groups_by_id: dict[str, list[dict]] = defaultdict(list)
    labels_by_group: dict[str, set[str]] = defaultdict(set)
    for record in records:
        groups_by_id[record["group_id"]].append(record)
        labels_by_group[record["group_id"]].add(record["label"])

    # DFDC links a real source and all derived fakes through the same group ID.
    # Such groups cannot be independently stratified by label or they leak.
    if any(len(labels) > 1 for labels in labels_by_group.values()):
        candidates = list(groups_by_id.values())
        rng = random.Random(f"{seed}:source-groups")
        rng.shuffle(candidates)
        n = len(records)
        validation_groups, remain = _take_groups(
            candidates, int(round(n * ratios["validation"])), rng
        )
        test_groups, train_groups = _take_groups(
            remain, int(round(n * ratios["test"])), rng
        )
        assignments = {
            "train": train_groups,
            "validation": validation_groups,
            "test": test_groups,
        }
        for split, selected_groups in assignments.items():
            for selected_group in selected_groups:
                for record in selected_group:
                    record["split"] = split
        return sorted(records, key=lambda r: (r["label"], r["video_id"]))

    grouped_by_label: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        grouped_by_label[record["label"]][record["group_id"]].append(record)

    group_to_split: dict[str, str] = {}
    for label, group_map in sorted(grouped_by_label.items()):
        groups = list(group_map.values())
        rng = random.Random(f"{seed}:{label}")
        rng.shuffle(groups)
        n = sum(len(group) for group in groups)
        targets = {
            split: int(round(n * ratios[split]))
            for split in ("validation", "test")
        }
        validation_groups, remain = _take_groups(
            groups, targets["validation"], rng
        )
        test_groups, train_groups = _take_groups(
            remain, targets["test"], rng
        )
        assignments = {
            "train": train_groups,
            "validation": validation_groups,
            "test": test_groups,
        }
        for split, selected_groups in assignments.items():
            for selected_group in selected_groups:
                for record in selected_group:
                    previous = group_to_split.setdefault(record["group_id"], split)
                    if previous != split:
                        raise RuntimeError(
                            f"Group assigned to multiple splits: {record['group_id']}"
                        )

    for record in records:
        record["split"] = group_to_split[record["group_id"]]

    if any("split" not in record for record in records):
        raise RuntimeError("Some records were not assigned a split")

    group_splits: dict[str, str] = {}
    for record in records:
        previous = group_splits.setdefault(record["group_id"], record["split"])
        if previous != record["split"]:
            raise RuntimeError(f"Group leakage detected for {record['group_id']}")
    return sorted(records, key=lambda r: (r["label"], r["video_id"]))


def split_summary(records: list[dict]) -> dict:
    summary: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        subset = [record for record in records if record.get("split") == split]
        summary[split] = {
            "videos": len(subset),
            "real": sum(record["label"] == "real" for record in subset),
            "fake": sum(record["label"] == "fake" for record in subset),
            "groups": len({record["group_id"] for record in subset}),
        }
    return summary


def write_manifest(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "video_id",
        "label",
        "label_index",
        "group_id",
        "split",
        "frame_count",
        "frame_names_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in sorted(
            records,
            key=lambda item: (
                item.get("split", ""),
                item["label"],
                item["video_id"],
            ),
        ):
            writer.writerow(
                {
                    **{
                        key: record.get(key, "")
                        for key in fields
                        if key != "frame_names_json"
                    },
                    "frame_names_json": repr(record.get("frame_names", [])),
                }
            )
