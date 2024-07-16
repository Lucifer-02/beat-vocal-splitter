import re
from pathlib import Path
import pprint

import yt_dlp


def get_title(url: str) -> str:
    with yt_dlp.YoutubeDL() as ydl:
        info_dict = ydl.extract_info(url, download=False)

        assert info_dict is not None, "Could not extract info from the video."
        return info_dict["title"]


def extract_video_id(url: str) -> str:
    video_id = ""
    regex = r"(?:https?:\/\/)?(?:www\.)?youtu\.?be(?:\.com)?\/?.*(?:watch|embed)?(?:.*v=|v\/|\/)([\w\-_]+)\&?"
    match = re.search(regex, url)
    if match and len(match.group(1)) == 11:
        video_id = match.group(1)
    else:
        raise ValueError("Invalid YouTube URL")
    return video_id


def get_request_url(request_file: Path) -> list[str]:
    with open(request_file, "r") as file:
        return file.read().split("\n")[:-1]
