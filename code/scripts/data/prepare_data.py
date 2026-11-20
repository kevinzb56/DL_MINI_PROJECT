"""Audit paired images and create an immutable, group-separated pilot manifest.

Input CSV columns: image_id,image,mask,dataset,group_id,partition.
Paths are relative to the CSV. partition is source_train, source_test, or target.
Use explicit parent-frame/sequence group IDs; filenames alone do not establish
independence. Missing metadata is a blocker to freezing, not silently invented.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.atomic import write_json


def build_manifest(csv_path, seed=42):
    import numpy as np
    from PIL import Image
    csv_path = Path(csv_path).resolve()
    records, seen_ids, seen_pixels, group_partitions = [], set(), set(), {}
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            identity = (row["dataset"], row["image_id"])
            if identity in seen_ids or not row["group_id"].strip():
                raise ValueError(f"Duplicate ID or missing group: {identity}")
            seen_ids.add(identity)
            partition = row["partition"]
            if partition not in {"source_train", "source_test", "target"}:
                raise ValueError(f"Invalid partition: {partition}")
            group = row["dataset"] + ":" + row["group_id"]
            if group in group_partitions and group_partitions[group] != partition:
                raise ValueError(f"Group spans partitions: {group}")
            group_partitions[group] = partition
            image_path = (csv_path.parent / row["image"]).resolve()
            mask_path = (csv_path.parent / row["mask"]).resolve()
            with Image.open(image_path) as original:
                image = original.convert("RGB")
                image.load()
            with Image.open(mask_path) as original:
                # Do not silently turn colored labels into binary targets.
                if original.mode not in {"1", "L"}:
                    raise ValueError(f"Mask needs explicit grayscale conversion: {mask_path}")
                mask = np.asarray(original.convert("L"))
            if image.size != (mask.shape[1], mask.shape[0]):
                raise ValueError(f"Image/mask size mismatch: {identity}")
            values = set(np.unique(mask).tolist())
            if not values <= {0, 255}:
                raise ValueError(f"Mask must use 0/255, found {values}: {identity}")
            pixel_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            if pixel_hash in seen_pixels:
                raise ValueError(f"Duplicate decoded image: {identity}")
            seen_pixels.add(pixel_hash)
            records.append({**row, "image": str(image_path), "mask": str(mask_path),
                            "group": group, "width": image.width, "height": image.height,
                            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                            "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                            "pixel_sha256": pixel_hash, "foreground_fraction": float((mask > 0).mean())})
    if not records:
        raise ValueError("No image/mask pairs")
    rng = random.Random(seed)
    allocations = {}
    for partition, first, second, fraction in [
        ("source_train", "source_validation", "source_train", .2),
        ("target", "target_acquisition", "target_query", .25),
    ]:
        groups = sorted({r["group"] for r in records if r["partition"] == partition})
        if len(groups) < 2:
            raise ValueError(f"Need at least two groups in {partition}")
        rng.shuffle(groups)
        count = max(1, min(len(groups)-1, round(len(groups)*fraction)))
        allocations.update({g: first if i < count else second for i, g in enumerate(groups)})
    for row in records:
        row["split"] = "source_holdout" if row["partition"] == "source_test" else allocations[row["group"]]
    counts = {s: sum(r["split"] == s for r in records) for s in sorted({r["split"] for r in records})}
    return {"schema_version": 1, "seed": seed, "counts": counts,
            "limitations": ["Exact decoded duplicates checked; near-duplicate/site independence requires manual audit.",
                            "Split fractions apply to groups; image fractions may differ."], "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite a frozen manifest; choose a new version")
    result = build_manifest(args.index, args.seed)
    write_json(args.output, result)
    print(json.dumps(result["counts"], indent=2))
    print("Manifest SHA256:", hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
