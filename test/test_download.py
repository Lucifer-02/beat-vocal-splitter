from unittest import TestCase
from pathlib import Path

from download import download_audio


class TestDownload(TestCase):
    def test_download_audio(self):
        download_audio(
            output=Path("audio/"), url="https://www.youtube.com/watch?v=6JYIGclVQdw"
        )
        self.assertTrue(Path("audio/6JYIGclVQdw.opus").exists())
