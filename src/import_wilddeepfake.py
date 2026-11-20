from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import tarfile
import uuid
from collections import defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZIP_STORED, ZipFile

from PIL import Image
from tqdm import tqdm


SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-manifest", required=True)
    parser.add_argument("--archives-dir", required=True)
    parser.add_argument("--output-zip", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--frames-per-sequence", type=int, default=4)
    return parser.parse_args()


def choose_indices(length: int, count: int) -> list[int]:
    if length <= count:
        return list(range(length))
    return [round(i * (length - 1) / (count - 1)) for i in range(count)]


def safe_sequence_id(label: str, member_parent: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9]+", "__", member_parent).strip("_")
    digest = hashlib.sha256(member_parent.encode("utf-8")).hexdigest()[:12]
    return f"{label}__{readable[-80:]}__{digest}"


def import_shards(args: argparse.Namespace) -> None:
    with Path(args.shard_manifest).open(newline="", encoding="utf-8") as handle:
        shards = list(csv.DictReader(handle))
    archives_root = Path(args.archives_dir)
    output = Path(args.output_zip)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{uuid.uuid4().hex[:8]}.tmp")
    rows = []
    try:
        with ZipFile(temporary, "w", compression=ZIP_STORED) as output_zip:
            for shard in tqdm(shards, desc="WildDeepfake shards"):
                label = shard["label"]
                expected_size = int(shard["size_bytes"])
                archive_path = archives_root / label / Path(shard["path"]).name
                if not archive_path.is_file():
                    raise FileNotFoundError(archive_path)
                if archive_path.stat().st_size != expected_size:
                    raise ValueError(
                        f"Size mismatch for {archive_path}: "
                        f"{archive_path.stat().st_size} != {expected_size}"
                    )
                grouped: dict[str, list[tuple[str, bytes]]] = defaultdict(list)
                with tarfile.open(archive_path, mode="r:gz") as archive:
                    for member in archive:
                        if not member.isfile():
                            continue
                        path = PurePosixPath(member.name)
                        if path.suffix.lower() not in SUFFIXES:
                            continue
                        grouped[str(path.parent)].append(
                            (path.name, archive.extractfile(member).read())
                        )
                for member_parent, images in grouped.items():
                    images.sort(key=lambda item: item[0])
                    selected_indices = choose_indices(
                        len(images), args.frames_per_sequence
                    )
                    if len(selected_indices) != args.frames_per_sequence:
                        continue
                    sequence_id = safe_sequence_id(label, member_parent)
                    selected_names = []
                    for output_number, source_index in enumerate(selected_indices):
                        source_name, payload = images[source_index]
                        with Image.open(io.BytesIO(payload)) as image:
                            rgb = image.convert("RGB")
                            buffer = io.BytesIO()
                            rgb.save(buffer, format="PNG")
                        output_name = f"{output_number:04d}.png"
                        archive_name = (
                            f"WildDeepfake/{label}/frames/{sequence_id}/{output_name}"
                        )
                        output_zip.writestr(archive_name, buffer.getvalue())
                        selected_names.append(source_name)
                    rows.append(
                        {
                            "video_id": sequence_id,
                            "label": label,
                            "source_archive": shard["path"],
                            "source_sequence": member_parent,
                            "source_frame_count": len(images),
                            "selected_frames": "|".join(selected_names),
                        }
                    )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = Path(args.output_manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Sequences: {len(rows)}")
    print(f"Archive: {output}")
    print(f"Manifest: {manifest}")


def main() -> None:
    import_shards(parse_args())


if __name__ == "__main__":
    main()
