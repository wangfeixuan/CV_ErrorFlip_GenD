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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in Path(args.runs_dir).glob("*/eval_test*.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("method") not in {"uniform", "restore"}:
            continue
        values[(result["method"], result.get("condition", "unknown"))].append(
            result["feature_restoration"]
        )
    rows = []
    for (method, condition), entries in sorted(values.items()):
        rows.append(
            {
                "method": method,
                "condition": condition,
                "seeds": len(entries),
                "raw_cosine": float(np.mean([x["raw_blur_cosine"] for x in entries])),
                "adapted_cosine": float(
                    np.mean([x["adapted_masked_cosine"] for x in entries])
                ),
                "improvement": float(
                    np.mean([x["cosine_improvement"] for x in entries])
                ),
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
