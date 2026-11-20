import csv
import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
from scripts.data.prepare_data import build_manifest


def make_index(root):
    rows = []
    for i in range(8):
        image = np.full((8, 8, 3), i * 20, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[:, 3] = 255
        Image.fromarray(image).save(root / f"{i}.png")
        Image.fromarray(mask).save(root / f"{i}_mask.png")
        rows.append({"image_id": str(i), "image": f"{i}.png", "mask": f"{i}_mask.png",
                     "dataset": "fixture", "group_id": str(i),
                     "partition": "source_train" if i < 4 else "target"})
    index = root / "index.csv"
    with index.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return index


def test_deterministic_disjoint_allocation(tmp_path):
    index = make_index(tmp_path)
    first = build_manifest(index)
    assert first == build_manifest(index)
    groups = {}
    for row in first["records"]:
        groups.setdefault(row["group"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in groups.values())
    assert first["counts"]["target_query"] == 3


def test_identical_decoded_image_rejected(tmp_path):
    index = make_index(tmp_path)
    (tmp_path / "4.png").write_bytes((tmp_path / "0.png").read_bytes())
    with pytest.raises(ValueError, match="Duplicate decoded"):
        build_manifest(index)


def test_nonbinary_mask_rejected(tmp_path):
    index = make_index(tmp_path)
    Image.fromarray(np.full((8, 8), 50, dtype=np.uint8)).save(tmp_path / "0_mask.png")
    with pytest.raises(ValueError, match="0/255"):
        build_manifest(index)
