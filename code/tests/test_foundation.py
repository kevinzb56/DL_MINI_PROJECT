import tempfile
from pathlib import Path
import unittest

from src.atomic import atomic_write, write_json


class AtomicTests(unittest.TestCase):
    def test_failed_write_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.json"
            write_json(path, {"step": 10})
            previous = path.read_bytes()
            def fail(stream):
                stream.write(b"partial")
                raise RuntimeError("simulated interruption")
            with self.assertRaises(RuntimeError):
                atomic_write(path, fail)
            self.assertEqual(previous, path.read_bytes())
            self.assertEqual([path], list(Path(directory).iterdir()))


if __name__ == "__main__":
    unittest.main()
