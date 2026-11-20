"""Audit the published split table and its 10-image split chunks."""
import argparse
import csv
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED = {"train": 2526, "val": 542, "test": 496}
TABLE = "splits/final_split_all_groups.csv"


def audit(path):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Split archive CRC check failed")
        with archive.open(TABLE) as stream:
            rows = list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")))
    required = {"image_id", "group", "split", "split_chunk_size", "platform", "height_category"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Official split table is empty or missing required columns")
    ids = [row["image_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image_id in split table")
    counts = Counter(row["split"] for row in rows)
    if dict(counts) != EXPECTED:
        raise ValueError(f"Split counts differ from published totals: {dict(counts)}")
    chunks = defaultdict(set)
    group_splits = defaultdict(set)
    for row in rows:
        if row["split"] not in EXPECTED:
            raise ValueError(f"Unknown split: {row['split']}")
        size = int(row["split_chunk_size"])
        suffix = row["image_id"].rsplit("_", 1)[-1]
        if size < 1 or not suffix.isdigit():
            raise ValueError(f"Cannot derive split chunk for {row['image_id']}")
        ordinal = int(suffix)
        # Publisher split units are contiguous ID blocks within each collection group.
        block = (ordinal - 1) // size
        chunks[(row["group"], block)].add(row["split"])
        group_splits[row["group"]].add(row["split"])
    crossing = [{"group": group, "block": block, "splits": sorted(splits)}
                for (group, block), splits in chunks.items() if len(splits) != 1]
    if crossing:
        raise ValueError(f"{len(crossing)} publisher split blocks cross splits")
    return {
        "record": "https://zenodo.org/records/22697998",
        "rows": len(rows), "split_counts": dict(counts),
        "collection_groups": len(group_splits), "split_blocks_of_10": len(chunks),
        "blocks_crossing_splits": 0,
        "groups_present_in_multiple_splits": sum(len(splits) > 1 for splits in group_splits.values()),
        "block_definition": "group plus (numeric image ID suffix - 1) // split_chunk_size",
        "publisher_block_isolation": True,
        "physical_site_independence_certified": False,
        "road_region_masks_in_release": False,
        "training_protocol_note": "Use publisher train only for updates, val for selection, and test once. Mix training records across all downloaded archives each epoch; do not train each archive to completion in sequence.",
        "source_split_table": TABLE,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.splits_zip)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
