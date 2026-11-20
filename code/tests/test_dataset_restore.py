import hashlib
from pathlib import Path
import tempfile
import unittest
from scripts.data.restore_datasets import verify_inventory


class RestoreTests(unittest.TestCase):
    def test_hash_verification_rejects_modified_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'image').write_bytes(b'image')
            (root/'mask').write_bytes(b'mask')
            row={'image_id':'fixture','image':'image','mask':'mask',
                 'image_sha256':hashlib.sha256(b'image').hexdigest(),
                 'mask_sha256':hashlib.sha256(b'mask').hexdigest()}
            verify_inventory(root,[row])
            (root/'mask').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                verify_inventory(root,[row])
            self.assertEqual((root/'mask').read_bytes(),b'changed')

    def test_missing_data_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                verify_inventory(Path(directory),[{'image_id':'fixture','image':'absent'}])
