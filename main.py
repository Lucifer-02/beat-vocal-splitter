from pathlib import Path
import os

from url_utils import get_request_url
from download import download_audio
from split_audio import split

if __name__ == "__main__":
    # urls = get_request_url(Path("./videoUrls.txt"))
    # download_audio(output=Path("./audio"), urls=urls)

    # get all audio files in the audio directory
    audio_files = [Path("./audio/") / file for file in os.listdir("./audio")]
    print(audio_files)

    for audio in audio_files:
        split(input=audio, output=Path("./split/"), tta=True)
        break

    # split(input=Path("./audio/2IyDFVCvKiU.opus"), output=Path("./split/"))
