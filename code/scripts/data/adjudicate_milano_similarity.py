"""Create and summarize a provenance-based review register for Milano pairs."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "output/milano_v1/milano_similarity_candidates.csv"
REGISTER = ROOT / "output/milano_v1/milano_similarity_adjudication.csv"
FIELDS = [
    "image_a", "image_b", "grayscale_64x64_pearson", "grayscale_64x64_rmse",
    "average_hash_distance_8x8", "current_split_a", "current_split_b",
    "source_or_capture_record_a", "source_or_capture_record_b", "same_capture_group",
    "same_physical_site", "relation", "review_status", "evidence_reference",
    "reviewer", "decision_notes",
]
ALLOWED = {"pending", "same_capture_group", "same_site_different_capture", "different_sites", "uncertain"}


def initialize(source=SOURCE, output=REGISTER):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing adjudications: {output}")
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows = [row for row in rows if float(row["grayscale_64x64_pearson"]) >= 0.95]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            a, b = row["image_a"], row["image_b"]
            writer.writerow({
                "image_a": a, "image_b": b,
                "grayscale_64x64_pearson": row["grayscale_64x64_pearson"],
                "grayscale_64x64_rmse": row["grayscale_64x64_rmse"],
                "average_hash_distance_8x8": row["average_hash_distance_8x8"],
                "current_split_a": a.split("/", 1)[0],
                "current_split_b": b.split("/", 1)[0],
                "same_capture_group": "", "same_physical_site": "",
                "relation": "", "review_status": "pending",
                "evidence_reference": "", "reviewer": "", "decision_notes": "",
            })
    return output


def summarize(path=REGISTER):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    invalid = [i + 2 for i, row in enumerate(rows) if row["review_status"] not in ALLOWED]
    if invalid:
        raise ValueError(f"Invalid review_status at CSV rows: {invalid}")
    resolved = [row for row in rows if row["review_status"] != "pending"]
    missing_evidence = [row for row in resolved if not row["evidence_reference"].strip()]
    missing_reviewer = [row for row in resolved if not row["reviewer"].strip()]
    invalid_decisions = []
    for line, row in enumerate(rows, start=2):
        if row["review_status"] == "pending":
            continue
        same_group = row["same_capture_group"].lower()
        same_site = row["same_physical_site"].lower()
        valid = same_group in {"true", "false"} and same_site in {"true", "false"}
        if row["review_status"] == "same_capture_group":
            valid &= same_group == "true"
        elif row["review_status"] == "same_site_different_capture":
            valid &= same_group == "false" and same_site == "true"
        elif row["review_status"] == "different_sites":
            valid &= same_group == "false" and same_site == "false"
        elif row["review_status"] == "uncertain":
            valid &= True
        if not valid:
            invalid_decisions.append(line)
    if invalid_decisions:
        raise ValueError(f"Inconsistent/blank relationship fields at CSV rows: {invalid_decisions}")
    # Split overlap remains unresolved for any pair without source-backed adjudication.
    unresolved = [row for row in rows if row["review_status"] in {"pending", "uncertain"}]
    cross_split = [row for row in rows if row["current_split_a"] != row["current_split_b"]]
    cross_split_reviewed = [row for row in cross_split if row["review_status"] not in {"pending", "uncertain"}
                            and row["evidence_reference"].strip() and row["reviewer"].strip()]
    confirmed_group_overlap = [row for row in cross_split if row["same_capture_group"].lower() == "true"]
    confirmed_site_overlap = [row for row in cross_split if row["same_physical_site"].lower() == "true"]
    return {
        "status": "review_pending" if unresolved else "review_recorded",
        "candidate_pairs": len(rows), "resolved_pairs": len(resolved),
        "unresolved_pairs": len(unresolved), "cross_split_pairs": len(cross_split),
        "cross_split_unresolved": sum(row in unresolved for row in cross_split),
        "resolved_missing_evidence": len(missing_evidence),
        "resolved_missing_reviewer": len(missing_reviewer),
        "cross_split_pairs_reviewed_with_evidence": len(cross_split_reviewed),
        "confirmed_cross_split_capture_group_overlap": len(confirmed_group_overlap),
        "confirmed_cross_split_site_overlap": len(confirmed_site_overlap),
        "can_claim_no_cross_split_capture_overlap": bool(rows) and len(cross_split_reviewed) == len(cross_split)
            and not confirmed_group_overlap,
        "can_claim_site_independence": bool(rows) and len(cross_split_reviewed) == len(cross_split)
            and not confirmed_site_overlap,
        "register": str(Path(path)),
        "guidance": "Use source/capture records. Visual similarity alone cannot establish site or sequence identity; keep the test result exploratory while any cross-split candidate is unresolved.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--source", type=Path, default=SOURCE)
    init.add_argument("--output", type=Path, default=REGISTER)
    report = sub.add_parser("report")
    report.add_argument("--register", type=Path, default=REGISTER)
    report.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = initialize(args.source, args.output) if args.command == "init" else summarize(args.register)
    if args.command == "report" and args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result if isinstance(result, dict) else {"created": str(result)}, indent=2))
