from __future__ import annotations

import argparse
import csv
import json
import random
import re
import urllib.request
from pathlib import Path


API_ROOT = (
    "https://hf-mirror.com/api/datasets/xingjunm/WildDeepfake/tree/main/"
    "deepfake_in_the_wild"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-gib-per-class", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def list_partition(partition: str) -> list[dict]:
    url = f"{API_ROOT}/{partition}?expand=true&limit=50"
    entries = []
    while url:
        request = urllib.request.Request(
            url, headers={"User-Agent": "CV-ErrorFlip-GenD/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            entries.extend(json.load(response))
            link = response.headers.get("Link", "")
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = match.group(1) if match else None
    return [entry for entry in entries if entry["type"] == "file"]


def main() -> None:
    args = parse_args()
    target = int(args.target_gib_per_class * 1024**3)
    rows = []
    for partition, label in (("real_train", "real"), ("fake_train", "fake")):
        entries = list_partition(partition)
        random.Random(args.seed).shuffle(entries)
        total = 0
        for entry in entries:
            if total + entry["size"] <= target:
                rows.append(
                    {
                        "label": label,
                        "path": entry["path"],
                        "size_bytes": entry["size"],
                    }
                )
                total += entry["size"]
        print(f"{partition}: {sum(1 for row in rows if row['label'] == label)} shards")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "path", "size_bytes"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["path"]))
    total_size = sum(row["size_bytes"] for row in rows)
    print(f"shards={len(rows)} bytes={total_size} GiB={total_size / 1024**3:.3f}")


if __name__ == "__main__":
    main()
