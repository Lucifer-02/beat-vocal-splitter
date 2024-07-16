from pathlib import Path

import yt_dlp


def download_audio(output: Path, urls: list[str]) -> None:
    assert output.is_dir(), "Output path must be a existing directory."

    config = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
            }
        ],
        "outtmpl": f"{str(output)}/%(id)s.%(ext)s",  # Output filename format
    }
    with yt_dlp.YoutubeDL(config) as video:
        video.download(urls)
