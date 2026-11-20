"""Resume verified downloads of the six public Zenodo dataset archives."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

RECORD = "22697998"
API = f"https://zenodo.org/api/records/{RECORD}"
ARCHIVES = [
    "01_ground_gopro.zip", "02_ground_iphone.zip", "03_uav_low_3_10m.zip",
    "04_uav_mid_15_30m.zip", "05_uav_high_35_100m.zip", "06_uav_other.zip",
]
METADATA = ["README.md", "LICENSE.txt", "SHA256SUMS.txt", "splits.zip"]
CHUNK_SHA256 = {
    "01_ground_gopro.zip": "2a2f01d5a746f1e7cfb6a3da9b0ccba5594c5bcdf489eed3c7d682b4aa16124d",
    "02_ground_iphone.zip": "99cb2fd7b22603cec14ec78843883cf911dc26fd230c5c6bd5c6114fbac77688",
    "03_uav_low_3_10m.zip": "33b7b31acb6dcd5ea474ee55b77fec55d267816b3039ea71127c6a27f9b3fad7",
    "04_uav_mid_15_30m.zip": "269d423bed34aa81c11f31ad66fc1028e4f1a4bd58c6de175c445ae522ef81b6",
    "05_uav_high_35_100m.zip": "b438623402c156bd12fafa226a9b6b36221025731e0d9110ea8c143cd1b3832c",
    "06_uav_other.zip": "a60c0741132dbf1947ad234e5144f088d0b0c357a45dd0cd74ac838a9dde1ec1",
}
USER_AGENT = "CrackLab-Multisource-Downloader/1.0"


def fetch_json(url):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_file(root, name, info, segment_mib):
    final = root / name
    partial = root / (name + ".part")
    url = f"https://zenodo.org/records/{RECORD}/files/{quote(name)}?download=1"
    expected_size = int(info["size"])
    checksum_kind, checksum_value = info["checksum"].split(":", 1)
    if final.exists():
        if final.stat().st_size == expected_size and digest(final, checksum_kind) == checksum_value:
            return {"file": name, "status": "already_verified", "bytes": expected_size}
        raise ValueError(f"Existing file fails size/checksum; preserving it: {final}")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise ValueError(f"Partial file is larger than Zenodo metadata: {partial}")
    segment = segment_mib * 1024 * 1024
    print(f"{name}: resuming at {offset}/{expected_size} bytes", flush=True)
    while offset < expected_size:
        end = min(offset + segment - 1, expected_size - 1)
        request = Request(url, headers={"User-Agent": USER_AGENT,
                                        "Range": f"bytes={offset}-{end}"})
        last_error = None
        for attempt in range(5):
            try:
                with urlopen(request, timeout=180) as response:
                    if offset == 0 and response.status == 200:
                        mode = "wb"
                        written = 0
                        with partial.open(mode) as target:
                            for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                                target.write(block); written += len(block)
                            target.flush()
                        offset = written
                    else:
                        content_range = response.headers.get("Content-Range", "")
                        if response.status != 206 or not content_range.startswith(f"bytes {offset}-{end}/"):
                            raise IOError(f"Server did not honor byte range {offset}-{end}: {response.status} {content_range}")
                        expected_segment = end - offset + 1
                        received = 0
                        with partial.open("ab") as target:
                            while received < expected_segment:
                                block = response.read(min(4 * 1024 * 1024, expected_segment - received))
                                if not block:
                                    break
                                target.write(block); received += len(block)
                            target.flush()
                        if received != expected_segment:
                            raise IOError(f"Short segment: got {received}, expected {expected_segment}")
                        offset += received
                break
            except (HTTPError, URLError, TimeoutError, OSError) as error:
                last_error = error
                if partial.exists() and partial.stat().st_size != offset:
                    with partial.open("r+b") as target:
                        target.truncate(offset)
                if attempt == 4:
                    raise RuntimeError(f"Download paused at {offset}/{expected_size} bytes; rerun to resume") from error
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError("Download retries exhausted") from last_error
        if partial.stat().st_size != offset:
            raise IOError("Partial-file length changed unexpectedly")
        print(f"{name}: {offset}/{expected_size} bytes ({100*offset/expected_size:.1f}%)", flush=True)
    if offset != expected_size or partial.stat().st_size != expected_size:
        raise ValueError(f"Downloaded size mismatch for {name}")
    actual = digest(partial, checksum_kind)
    if actual != checksum_value:
        raise ValueError(f"Zenodo {checksum_kind} mismatch for {name}; partial retained for diagnosis")
    partial.replace(final)
    return {"file": name, "status": "downloaded_verified", "bytes": expected_size,
            "checksum": f"{checksum_kind}:{actual}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Persistent folder with enough free space; prefer an external drive")
    parser.add_argument("--files", nargs="+", required=True,
                        help="Choose metadata, all, or one or more archive filenames")
    parser.add_argument("--segment-mib", type=int, default=64,
                        help="HTTP resume segment size (default: 64 MiB)")
    args = parser.parse_args()
    if args.segment_mib < 1 or args.segment_mib > 512:
        parser.error("--segment-mib must be between 1 and 512")
    if "all" in args.files:
        wanted = set(METADATA + ARCHIVES)
    elif "metadata" in args.files:
        wanted = set(METADATA)
    else:
        wanted = set(args.files)
        if wanted.intersection(ARCHIVES):
            wanted.update(METADATA)
    unknown = wanted - set(METADATA + ARCHIVES)
    if unknown:
        parser.error("Unknown file(s): " + ", ".join(sorted(unknown)))
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    record = fetch_json(API)
    if str(record.get("id")) != RECORD:
        raise ValueError("Zenodo API record ID mismatch")
    remote = {entry["key"]: entry for entry in record["files"]}
    missing = wanted - set(remote)
    if missing:
        raise ValueError("Expected files missing from Zenodo record: " + ", ".join(sorted(missing)))
    pending_bytes = sum(int(remote[name]["size"]) - (root/(name+".part")).stat().st_size
                        if (root/(name+".part")).exists() else int(remote[name]["size"])
                        for name in wanted if not (root/name).exists())
    free = shutil.disk_usage(root).free
    safety = 512 * 1024 * 1024
    if free < pending_bytes + safety:
        raise OSError(f"Insufficient space at {root}: need {pending_bytes/1024**3:.2f} GiB plus 0.50 GiB headroom; have {free/1024**3:.2f} GiB")
    sums = None
    sums_path = root / "SHA256SUMS.txt"
    if "SHA256SUMS.txt" in wanted:
        result = download_file(root, "SHA256SUMS.txt", remote["SHA256SUMS.txt"], args.segment_mib)
        sums = {}
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            fields = line.split(maxsplit=1)
            if len(fields) == 2 and len(fields[0]) == 64:
                filename = fields[1].replace("\\", "/").rsplit("/", 1)[-1]
                if filename.endswith(".zip"):
                    sums[filename] = fields[0].lower()
        if any(sums.get(name) != CHUNK_SHA256[name] for name in ARCHIVES):
            raise ValueError("Published SHA256SUMS entries differ from the pinned dataset archive hashes")
    results = []
    for name in [*METADATA, *ARCHIVES]:
        if name not in wanted or name == "SHA256SUMS.txt":
            continue
        result = download_file(root, name, remote[name], args.segment_mib)
        if name in CHUNK_SHA256:
            if not sums:
                raise ValueError("Download/verify SHA256SUMS.txt before downloading dataset archives")
            if digest(root/name) != sums[name]:
                raise ValueError(f"Published SHA256SUMS mismatch for {name}")
            result["sha256_verified"] = True
        results.append(result)
    status_path = root / "DOWNLOAD_STATUS.json"
    prior = {}
    if status_path.exists():
        prior = json.loads(status_path.read_text(encoding="utf-8"))
        if str(prior.get("record")) != RECORD:
            raise ValueError("Existing download status belongs to a different Zenodo record")
    known = {entry["file"]: entry for entry in prior.get("files", [])}
    known.update({entry["file"]: entry for entry in results})
    ordered = [known[name] for name in [*METADATA, *ARCHIVES] if name in known]
    inventory = {"record": RECORD, "version": record.get("metadata", {}).get("version"),
                 "license": "CC BY 4.0", "download_root": str(root), "files": ordered}
    status_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
