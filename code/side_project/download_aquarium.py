"""Download and verify the public Aquarium Combined image and label files."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "aquarium_combined"
REPO = "LibreYOLO/aquarium-qlnqy"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main/"


def fetch(url):
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == 7:
                raise
            time.sleep(min(2 ** attempt, 20))


def save(item):
    rel = item["path"]
    target = DEST / rel
    if target.exists() and target.stat().st_size == item["size"]:
        return rel
    target.parent.mkdir(parents=True, exist_ok=True)
    data = fetch(RAW + urllib.parse.quote(rel, safe="/"))
    if len(data) != item["size"]:
        raise ValueError(f"Size mismatch for {rel}: {len(data)} != {item['size']}")
    temp = target.with_suffix(target.suffix + ".part")
    temp.write_bytes(data)
    temp.replace(target)
    return rel


def main():
    files = []
    for split in ("train", "valid", "test"):
        for kind in ("images", "labels"):
            folder = split + "/" + kind
            items = json.loads(fetch(API + folder + "?limit=1000"))
            files.extend(item for item in items if item["type"] == "file")
    for name in ("data.yaml", "README.dataset.txt", "README.roboflow.txt"):
        files.append({"path": name, "size": len(fetch(RAW + name))})
    DEST.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(save, item) for item in files]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 100 == 0 or index == len(files):
                print(f"Downloaded/verified {index}/{len(files)} files", flush=True)
    counts = {}
    label_classes = set()
    for split in ("train", "valid", "test"):
        images = list((DEST / split / "images").glob("*.jpg"))
        labels = list((DEST / split / "labels").glob("*.txt"))
        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        if image_stems != label_stems:
            raise ValueError(f"Image/label pairing mismatch in {split}")
        for p in labels:
            for line in p.read_text().splitlines():
                if line.strip():
                    label_classes.add(int(line.split()[0]))
        counts[split] = {"images": len(images), "labels": len(labels)}
    manifest = {
        "repository": f"https://huggingface.co/datasets/{REPO}",
        "source": "https://universe.roboflow.com/brad-dwyer/aquarium-combined",
        "counts": counts,
        "classes_present_in_original_labels": sorted(label_classes),
        "original_labels_are_reference_only": True,
        "files": {item["path"]: hashlib.sha256((DEST / item["path"]).read_bytes()).hexdigest()
                  for item in files},
    }
    (DEST / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
