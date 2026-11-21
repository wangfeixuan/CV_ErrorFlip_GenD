from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from zipfile import ZIP_STORED, ZipFile

import numpy as np
from tqdm import tqdm


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
ARCFACE_112 = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _opencv():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "DFDC import requires opencv-python-headless; "
            "install requirements.txt before running it"
        ) from error
    return cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a DFDC video shard into four aligned face frames per video."
    )
    parser.add_argument("--videos", required=True, help="DFDC ZIP or directory")
    parser.add_argument("--metadata", help="Official metadata JSON or metadata CSV")
    parser.add_argument("--yunet-model", help="OpenCV YuNet ONNX model")
    parser.add_argument("--output-zip")
    parser.add_argument("--output-manifest")
    parser.add_argument("--failed-csv")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--candidate-frames", type=int, default=8)
    parser.add_argument("--score-threshold", type=float, default=0.70)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=2,
        help="OpenCV CPU threads; keep low when another job shares the host",
    )
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only validate the archive and its metadata coverage",
    )
    return parser.parse_args()


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if length <= count:
        return list(range(length))
    return [
        round(index * (length - 1) / (count - 1))
        for index in range(count)
    ]


def _normalise_label(value: str) -> str:
    label = value.strip().lower()
    if label in {"real", "0"}:
        return "real"
    if label in {"fake", "1"}:
        return "fake"
    raise ValueError(f"Unsupported DFDC label: {value!r}")


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def source_group_id(original: str | None, video_name: str) -> str:
    """Keep a fake video with the real source named by DFDC's original field."""
    source = Path((original or video_name).strip()).name
    if not source:
        source = video_name
    digest = hashlib.sha256(source.lower().encode("utf-8")).hexdigest()[:16]
    return f"dfdc-source-{digest}"


def _records_from_mapping(mapping: dict) -> dict[str, dict]:
    records = {}
    for raw_name, metadata in mapping.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"Metadata for {raw_name} is not an object")
        name = Path(str(raw_name)).name
        label = _normalise_label(str(metadata.get("label", "")))
        original = _optional_text(metadata.get("original"))
        records[name] = {
            "filename": name,
            "label": label,
            "original": original,
        }
    return records


def _metadata_payload(path: str | Path) -> tuple[str, str]:
    source = Path(path)
    if source.suffix.lower() != ".zip":
        return source.name, source.read_text(encoding="utf-8-sig")
    with ZipFile(source) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if not name.endswith("/")
            and (
                Path(name).name.lower() in {"metadata", "metadata.csv", "metadata.json"}
                or name.lower().endswith((".json", ".csv"))
            )
        ]
        preferred = [
            name
            for name in candidates
            if Path(name).name.lower() in {"metadata", "metadata.csv", "metadata.json"}
        ]
        selected = sorted(preferred or candidates)
        if not selected:
            raise FileNotFoundError(f"No metadata JSON/CSV in {source}")
        name = selected[0]
        return name, archive.read(name).decode("utf-8-sig")


def read_metadata(path: str | Path) -> dict[str, dict]:
    name, payload = _metadata_payload(path)
    stripped = payload.lstrip()
    if stripped.startswith("{"):
        parsed = json.loads(stripped)
        return _records_from_mapping(parsed)
    if stripped.startswith("["):
        parsed = json.loads(stripped)
        return _records_from_mapping(
            {str(row["filename"]): row for row in parsed}
        )

    first_line = payload.splitlines()[0] if payload.splitlines() else ""
    try:
        dialect = csv.Sniffer().sniff(first_line, delimiters=",\t;")
        reader = csv.DictReader(io.StringIO(payload), dialect=dialect)
    except csv.Error:
        reader = csv.DictReader(io.StringIO(payload))
    rows = list(reader)
    if not rows or "filename" not in rows[0] or "label" not in rows[0]:
        raise ValueError(
            f"Metadata {name} is neither an official JSON map nor a CSV "
            "with filename,label columns"
        )
    return _records_from_mapping({row["filename"]: row for row in rows})


def list_video_members(path: str | Path) -> dict[str, str]:
    """Return basename -> archive path without extracting the shard."""
    source = Path(path)
    if source.is_file():
        with ZipFile(source) as archive:
            members = [
                name
                for name in archive.namelist()
                if not name.endswith("/")
                and PurePosixPath(name).suffix.lower() in VIDEO_SUFFIXES
            ]
        mapping = {Path(name).name: name for name in members}
        if len(mapping) != len(members):
            raise ValueError("Duplicate video basenames in DFDC archive")
        return mapping
    if source.is_dir():
        members = [
            str(path.relative_to(source))
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        ]
        mapping = {Path(name).name: name for name in members}
        if len(mapping) != len(members):
            raise ValueError("Duplicate video basenames in DFDC directory")
        return mapping
    raise FileNotFoundError(source)


def _materialise_video(source_root: Path, member: str, destination: Path) -> None:
    if source_root.is_file():
        with ZipFile(source_root) as archive, archive.open(member) as input_handle:
            with destination.open("wb") as output_handle:
                while block := input_handle.read(8 * 1024 * 1024):
                    output_handle.write(block)
    else:
        shutil.copyfile(source_root / member, destination)


def _candidate_frames(video_path: Path, count: int) -> tuple[list[tuple[int, np.ndarray]], int]:
    cv2 = _opencv()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return [], 0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        indices = evenly_spaced_indices(len(frames), count)
        return [(index, frames[index]) for index in indices], len(frames)

    frames = []
    for index in evenly_spaced_indices(frame_count, count):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if ok:
            frames.append((index, frame))
    capture.release()
    return frames, frame_count


def _select_detection(
    detections: np.ndarray | None,
    previous_center: tuple[float, float] | None,
    image_shape: tuple[int, ...],
) -> np.ndarray | None:
    if detections is None or not len(detections):
        return None
    if previous_center is None:
        areas = detections[:, 2] * detections[:, 3]
        scores = detections[:, -1]
        return detections[int(np.argmax(areas * scores))]
    height, width = image_shape[:2]
    diagonal = math.hypot(width, height)
    centers = detections[:, :2] + detections[:, 2:4] / 2
    distances = np.linalg.norm(
        centers - np.asarray(previous_center)[None, :], axis=1
    ) / diagonal
    return detections[int(np.argmin(distances))]


def align_face(
    frame: np.ndarray,
    detection: np.ndarray,
    image_size: int,
) -> np.ndarray:
    cv2 = _opencv()
    source_points = detection[4:14].reshape(5, 2).astype(np.float32)
    target_points = ARCFACE_112 * (image_size / 112.0)
    transform, _ = cv2.estimateAffinePartial2D(
        source_points,
        target_points,
        method=cv2.LMEDS,
    )
    if transform is None:
        raise ValueError("Could not estimate face alignment transform")
    aligned = cv2.warpAffine(
        frame,
        transform,
        (image_size, image_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)


def _png_payload(image: np.ndarray) -> bytes:
    cv2 = _opencv()
    success, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("PNG encoding failed")
    return encoded.tobytes()


def import_videos(args: argparse.Namespace) -> None:
    source_root = Path(args.videos).expanduser().resolve()
    members = list_video_members(source_root)
    if args.inspect_only and not args.metadata:
        print(f"Archive videos: {len(members)}")
        return
    metadata = read_metadata(args.metadata)
    missing = sorted(set(members) - set(metadata))
    extra = sorted(set(metadata) - set(members))
    print(
        f"Archive videos: {len(members)}; metadata rows: {len(metadata)}; "
        f"missing metadata: {len(missing)}; unrelated metadata: {len(extra)}",
        flush=True,
    )
    if args.inspect_only:
        return
    if missing:
        raise ValueError(
            "Every archived video needs an explicit label. "
            f"Missing metadata examples: {missing[:5]}"
        )
    if args.candidate_frames < 4:
        raise ValueError("--candidate-frames must be at least 4")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")

    selected_names = sorted(members)
    if args.max_videos is not None:
        selected_names = selected_names[: args.max_videos]
    output = Path(args.output_zip).expanduser().resolve()
    manifest_path = Path(args.output_manifest).expanduser().resolve()
    failed_path = Path(
        args.failed_csv
        or output.with_name(output.stem + ".failed.csv")
    ).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{uuid.uuid4().hex[:8]}.tmp")

    cv2 = _opencv()
    cv2.setNumThreads(max(1, args.num_threads))
    detector = cv2.FaceDetectorYN.create(
        str(Path(args.yunet_model).expanduser().resolve()),
        "",
        (320, 320),
        args.score_threshold,
        0.3,
        5000,
    )
    manifest_rows = []
    failures = []
    try:
        with tempfile.TemporaryDirectory(
            prefix="dfdc_import_", dir=output.parent
        ) as temp_dir, ZipFile(temporary, "w", compression=ZIP_STORED) as output_zip:
            video_path = Path(temp_dir) / "source.mp4"
            for video_name in tqdm(selected_names, desc="DFDC videos"):
                metadata_row = metadata[video_name]
                video_path.unlink(missing_ok=True)
                _materialise_video(source_root, members[video_name], video_path)
                frame_candidates, frame_count = _candidate_frames(
                    video_path, args.candidate_frames
                )
                aligned = []
                previous_center = None
                for source_index, frame in frame_candidates:
                    height, width = frame.shape[:2]
                    detector.setInputSize((width, height))
                    _, detections = detector.detect(frame)
                    detection = _select_detection(
                        detections, previous_center, frame.shape
                    )
                    if detection is None:
                        continue
                    previous_center = (
                        float(detection[0] + detection[2] / 2),
                        float(detection[1] + detection[3] / 2),
                    )
                    aligned.append(
                        (
                            source_index,
                            align_face(frame, detection, args.image_size),
                            float(detection[-1]),
                        )
                    )
                if len(aligned) < 4:
                    failures.append(
                        {
                            "filename": video_name,
                            "label": metadata_row["label"],
                            "reason": f"only {len(aligned)} faces in {frame_count} frames",
                        }
                    )
                    continue
                chosen = [
                    aligned[index]
                    for index in evenly_spaced_indices(len(aligned), 4)
                ]
                stem = re.sub(r"[^A-Za-z0-9]+", "_", Path(video_name).stem)
                digest = hashlib.sha256(video_name.encode("utf-8")).hexdigest()[:10]
                label = metadata_row["label"]
                video_id = f"dfdc-{label}__{stem}__{digest}"
                for frame_index, (_, image, _) in enumerate(chosen):
                    output_zip.writestr(
                        f"DFDC/{label}/frames/{video_id}/{frame_index:04d}.png",
                        _png_payload(image),
                    )
                manifest_rows.append(
                    {
                        "video_id": video_id,
                        "label": label,
                        "label_index": int(label == "fake"),
                        "group_id": source_group_id(
                            metadata_row["original"], video_name
                        ),
                        "source_video": video_name,
                        "source_original": metadata_row["original"] or "",
                        "source_frame_indices": "|".join(
                            str(item[0]) for item in chosen
                        ),
                        "face_scores": "|".join(
                            f"{item[2]:.6f}" for item in chosen
                        ),
                    }
                )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    with failed_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["filename", "label", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)
    summary = {
        "source_videos": len(selected_names),
        "imported_videos": len(manifest_rows),
        "failed_videos": len(failures),
        "real": sum(row["label"] == "real" for row in manifest_rows),
        "fake": sum(row["label"] == "fake" for row in manifest_rows),
        "groups": len({row["group_id"] for row in manifest_rows}),
        "frames_per_video": 4,
        "image_size": args.image_size,
        "face_detector": "OpenCV YuNet",
        "alignment": "ArcFace five-point similarity transform",
        "output_zip": str(output),
        "manifest": str(manifest_path),
        "failed_csv": str(failed_path),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    if not args.inspect_only:
        missing = [
            name
            for name in ("metadata", "yunet_model", "output_zip", "output_manifest")
            if not getattr(args, name)
        ]
        if missing:
            raise ValueError(f"Missing required arguments: {missing}")
    import_videos(args)


if __name__ == "__main__":
    main()
