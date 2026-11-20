"""Verify the in-progress Zenodo download and publish its final filename."""
import hashlib
import json
from pathlib import Path
import time
import zipfile

folder = Path(__file__).resolve().parents[4] / 'data/external/milano_pavement_2026'
archive = folder / 'Pavement_Crack_Segmentation_Dataset.zip.part'
status = folder / 'download_status.json'
expected = '4b530eba05708a195355a0064e520db4'
source = 'https://zenodo.org/records/20040434'

def record(state, **details):
    status.write_text(json.dumps(dict(state=state, source=source,
                                    expected_md5=expected, **details), indent=2))

record('downloading')
deadline = time.monotonic() + 4 * 3600
while time.monotonic() < deadline:
    if archive.exists() and zipfile.is_zipfile(archive):
        record('verifying', bytes=archive.stat().st_size)
        digest = hashlib.md5()
        with archive.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected:
            record('checksum_failed', actual_md5=actual)
            raise SystemExit('Checksum mismatch; partial archive preserved.')
        with zipfile.ZipFile(archive) as bundle:
            entries = bundle.infolist()
            summary = dict(files=sum(not x.is_dir() for x in entries),
                           uncompressed_bytes=sum(x.file_size for x in entries))
        final = archive.with_suffix('')
        archive.rename(final)
        record('verified', actual_md5=actual, bytes=final.stat().st_size,
               archive=final.name, **summary)
        print('Download complete; publisher MD5 verified.', flush=True)
        break
    time.sleep(15)
else:
    record('verification_timeout', bytes=archive.stat().st_size if archive.exists() else 0)
    raise SystemExit('Download did not complete within four hours; partial preserved.')
