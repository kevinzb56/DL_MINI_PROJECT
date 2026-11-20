# Multi-source pavement crack dataset workflow

Dataset: [Zenodo record 22697998](https://zenodo.org/records/22697998), version
1.0.0. Its authors license the dataset, masks, metadata and documentation under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite the Zenodo DOI and
retain attribution when publishing results.

## What has been checked

The small metadata files are downloaded under
`data/external/multisource_cracks_v1/`. Their source checksums passed. The split
table contains 3,564 records: 2,526 train, 542 validation and 496 test. It defines
10-image blocks; auditing the numeric image IDs within each publisher group found
340 blocks, with zero blocks crossing splits. All 21 broad collection groups
appear in all three splits, so the official split is block-separated but does not
establish physical-site independence. The release has crack masks but no road
region masks.

The audit is saved as `data/external/multisource_cracks_v1/SPLIT_AUDIT.json` and
can be regenerated with:

```powershell
python scripts/data/audit_multisource_splits.py `
  --splits-zip data/external/multisource_cracks_v1/splits.zip `
  --output data/external/multisource_cracks_v1/SPLIT_AUDIT.json
```

## Downloading the six archives

The six archives total about 15.76 GB (16.2 GB rounded by the Zenodo page) before
extraction. At the last check C: had about 19.41 GiB free, enough for the remaining
compressed archives but not a second extracted copy. Recheck disk space before
continuing. The downloader resumes `.part` files, verifies the Zenodo checksum,
then checks each dataset archive against the published SHA256 list. It refuses
to overwrite an existing file with a different checksum and merges progress into
`DOWNLOAD_STATUS.json`.

The GoPro archive is already downloaded and SHA256-verified in
`data/external/multisource_cracks_v1/`. For the remaining five, run the commands
in [the handoff runbook](MULTISOURCE_DATASET_HANDOFF.md); it includes exact sizes,
hashes and current progress.

```powershell
python scripts/data/download_multisource_cracks.py --root E:\road_crack_data\multisource_v1 --files 01_ground_gopro.zip
python scripts/data/download_multisource_cracks.py --root E:\road_crack_data\multisource_v1 --files 02_ground_iphone.zip
python scripts/data/download_multisource_cracks.py --root E:\road_crack_data\multisource_v1 --files 03_uav_low_3_10m.zip
python scripts/data/download_multisource_cracks.py --root E:\road_crack_data\multisource_v1 --files 04_uav_mid_15_30m.zip
python scripts/data/download_multisource_cracks.py --root E:\road_crack_data\multisource_v1 --files 05_uav_high_35_100m.zip
python scripts/data/download_multisource_cracks.py --root E:\road_crack_data\multisource_v1 --files 06_uav_other.zip
```

Each command also ensures the README, license, SHA256 list and official split
archive are present. A single archive is a download unit, not a train/validation/
test unit. Do not use the archive names as split assignments.

## Training/evaluation protocol

An explicitly authorized, three-epoch archive-1 pipeline pilot is documented in
[MULTISOURCE_ARCHIVE1_PILOT.md](MULTISOURCE_ARCHIVE1_PILOT.md). It keeps test
evaluation disabled and does not replace the full-dataset protocol below.

First evaluate the already-trained Milano checkpoints on the new official test
split before training on this dataset; that gives the cleanest cross-dataset
transfer measurement. Save those predictions and metrics separately. For new
training on this dataset, use only official `train` for weight updates, `val` for
checkpoint selection, and `test` once after the method is frozen.

"Chunk-wise" means the six archives can be downloaded and staged one at a time.
For model training, mix/shuffle examples from all available training archives
within each epoch. Training one whole archive to completion before starting the
next would confound the result with camera/platform shifts and can cause forgetting.
Keep the publisher's 10-image block assignments intact. The split table's
`image_path` and `mask_path` columns are original author machine paths, so a local
preparation step must pair by `image_id` and confirm every file against the
downloaded archive before training. ZIP-backed loading is implemented for the
archive-1 pilot; six-archive manifest preparation still needs implementation
and validation. Do not begin training until the
archive layout, image-mask pairing, binary mask values, dimensions, and official
split assignments are validated.
