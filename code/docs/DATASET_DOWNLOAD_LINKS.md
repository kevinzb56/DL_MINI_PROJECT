# Original multisource dataset download links

Saved 16 November 2026. The publisher record was checked on this date.

Dataset: [Multi-Source Pavement Crack Segmentation Dataset and Trained Deep Learning Models from Ground-Level and UAV Imagery](https://zenodo.org/records/22697998).

- Version: 1.0.0
- DOI: [10.5281/zenodo.22697998](https://doi.org/10.5281/zenodo.22697998)
- Authors: Chaudhry Abu Bakar Imran and Flavia-Ioana Patrascu
- License: CC BY 4.0 (retain publisher attribution and license)
- Restore directory: `data/external/multisource_cracks_v1/`

## Six original archives

| Archive / direct download | Exact bytes | GiB |
|---|---:|---:|
| [01_ground_gopro.zip](https://zenodo.org/records/22697998/files/01_ground_gopro.zip?download=1) | 2,103,023,240 | 1.959 |
| [02_ground_iphone.zip](https://zenodo.org/records/22697998/files/02_ground_iphone.zip?download=1) | 1,637,391,230 | 1.525 |
| [03_uav_low_3_10m.zip](https://zenodo.org/records/22697998/files/03_uav_low_3_10m.zip?download=1) | 2,307,063,695 | 2.149 |
| [04_uav_mid_15_30m.zip](https://zenodo.org/records/22697998/files/04_uav_mid_15_30m.zip?download=1) | 3,445,893,965 | 3.209 |
| [05_uav_high_35_100m.zip](https://zenodo.org/records/22697998/files/05_uav_high_35_100m.zip?download=1) | 4,357,800,972 | 4.059 |
| [06_uav_other.zip](https://zenodo.org/records/22697998/files/06_uav_other.zip?download=1) | 1,906,268,391 | 1.775 |

## Integrity checks

These SHA-256 values are preserved from the publisher SHA256SUMS file and pinned by the project downloader.

```text
2a2f01d5a746f1e7cfb6a3da9b0ccba5594c5bcdf489eed3c7d682b4aa16124d  01_ground_gopro.zip
99cb2fd7b22603cec14ec78843883cf911dc26fd230c5c6bd5c6114fbac77688  02_ground_iphone.zip
33b7b31acb6dcd5ea474ee55b77fec55d267816b3039ea71127c6a27f9b3fad7  03_uav_low_3_10m.zip
269d423bed34aa81c11f31ad66fc1028e4f1a4bd58c6de175c445ae522ef81b6  04_uav_mid_15_30m.zip
b438623402c156bd12fafa226a9b6b36221025731e0d9110ea8c143cd1b3832c  05_uav_high_35_100m.zip
a60c0741132dbf1947ad234e5144f088d0b0c357a45dd0cd74ac838a9dde1ec1  06_uav_other.zip
```

## Supporting files

Keep the official split definitions for reproducible train/validation/test membership.
- [splits.zip](https://zenodo.org/records/22697998/files/splits.zip?download=1)
- [SHA256SUMS.txt](https://zenodo.org/records/22697998/files/SHA256SUMS.txt?download=1)
- [LICENSE.txt](https://zenodo.org/records/22697998/files/LICENSE.txt?download=1)
- [README.md](https://zenodo.org/records/22697998/files/README.md?download=1)

## Restore

The existing downloader supports resuming interrupted transfers and verifies the pinned archive hashes. Run from the project root:

```powershell
python scripts/data/download_multisource_cracks.py --root data/external/multisource_cracks_v1 --files SHA256SUMS.txt LICENSE.txt README.md splits.zip 01_ground_gopro.zip 02_ground_iphone.zip 03_uav_low_3_10m.zip 04_uav_mid_15_30m.zip 05_uav_high_35_100m.zip 06_uav_other.zip
```

For a manually downloaded archive, check its hash and compare against the values above:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath data/external/multisource_cracks_v1/01_ground_gopro.zip
```

These links preserve a route to reacquire the publisher dataset; they are not a separate backup of the files. This document does not replace trained checkpoints, project manifests, or experiment results.
