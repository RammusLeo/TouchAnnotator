"""The public downloader must reject corrupt or truncated model assets."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / 'scripts/download_assets.py'
spec = importlib.util.spec_from_file_location('download_assets', path)
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class DownloadIntegrityTest(unittest.TestCase):
    payload = b'verified student checkpoint'

    def fake_download(self, repo_id, filename, revision, local_dir, token):
        self.assertIs(token, False)
        self.assertEqual(revision, 'v1.0')
        destination = Path(local_dir) / filename
        if filename == 'config.yaml':
            destination.write_text('model: {}\n')
        elif filename == 'manifest.json':
            destination.write_text(json.dumps({
                'bytes': len(self.payload),
                'sha256': hashlib.sha256(self.payload).hexdigest(),
            }))
        else:
            destination.write_bytes(self.downloaded_payload)
        return str(destination)

    def test_anonymous_download_verifies_the_release(self):
        self.downloaded_payload = self.payload
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(downloader, 'hf_hub_download', self.fake_download):
                downloader.download_student(Path(directory), 'v1.0')
            self.assertEqual((Path(directory) / 'model.pth').read_bytes(), self.payload)

    def test_corruption_and_truncation_are_rejected(self):
        for bad_payload, message in [
            (b'X' + self.payload[1:], 'SHA256 mismatch'),
            (self.payload[:-1], 'size mismatch'),
        ]:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                self.downloaded_payload = bad_payload
                with patch.object(downloader, 'hf_hub_download', self.fake_download):
                    with self.assertRaisesRegex(RuntimeError, message):
                        downloader.download_student(Path(directory), 'v1.0')


if __name__ == '__main__':
    unittest.main()
