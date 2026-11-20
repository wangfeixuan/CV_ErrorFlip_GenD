#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/CV_ErrorFlip_GenD
MANIFEST="$ROOT/code/manifests/wilddeepfake_subset_shards.csv"
BASE_URL=https://hf-mirror.com/datasets/xingjunm/WildDeepfake/resolve/main
DEST="$ROOT/data/wilddeepfake_subset/archives"

tail -n +2 "$MANIFEST" | while IFS=, read -r label path size; do
  mkdir -p "$DEST/$label"
  target="$DEST/$label/$(basename "$path")"
  if [[ -f "$target" && "$(stat -c %s "$target")" == "$size" ]]; then
    continue
  fi
  echo "Downloading $path"
  curl -L --fail --retry 5 --retry-delay 3 \
    -o "$target.part" \
    "$BASE_URL/$path"
  mv "$target.part" "$target"
done

du -sh "$DEST"
