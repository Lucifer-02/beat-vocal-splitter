import argparse
import os

import librosa
from matplotlib.pyplot import get
import numpy as np
import soundfile as sf
import torch

import sys

sys.path.append("vocal_remover/")
sys.path.append("vocal_remover/models")

from lib import dataset
from lib import nets
from lib import spec_utils
from lib import utils
import inference


def split_audio(vid_id: str, codec: str):
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", "-g", type=int, default=-1)
    p.add_argument(
        "--pretrained_model",
        "-P",
        type=str,
        default="./vocal_remover/models/baseline.pth",
    )
    p.add_argument("--input", "-i", required=False)
    p.add_argument("--sr", "-r", type=int, default=44100)
    p.add_argument("--n_fft", "-f", type=int, default=2048)
    p.add_argument("--hop_length", "-H", type=int, default=1024)
    p.add_argument("--batchsize", "-B", type=int, default=4)
    p.add_argument("--cropsize", "-c", type=int, default=256)
    p.add_argument("--output_image", "-I", action="store_true")
    p.add_argument("--postprocess", "-p", action="store_true")
    p.add_argument("--tta", "-t", action="store_true")
    p.add_argument("--output_dir", "-o", type=str, default="")

    args = p.parse_args()

    # my configs
    # args.gpu = 0
    args.output_dir = "split"
    args.input = f"audio/{vid_id}.{codec}"

    # print("gpu", args.gpu)
    # print("input", args.input)
    # print("sr", args.sr)
    # print("pretrained", args.pretrained_model)
    # print("hop length", args.hop_length)
    # print("n fft", args.n_fft)
    # print("batchsize", args.batchsize)
    # print("cropsize", args.cropsize)
    # print("postprocess", args.postprocess)
    # print("tta", args.tta)
    # print("output_dir", args.output_dir)
    # print("output_image", args.output_image)

    print("loading model...", end=" ")
    device = torch.device("cpu")
    model = nets.CascadedNet(args.n_fft, 32, 128)
    model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    if args.gpu >= 0:
        if torch.cuda.is_available():
            device = torch.device("cuda:{}".format(args.gpu))
            model.to(device)
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
            model.to(device)
    print("done")

    print("loading wave source...", end=" ")
    X, sr = librosa.load(
        args.input, sr=args.sr, mono=False, dtype=np.float32, res_type="kaiser_fast"
    )
    basename = os.path.splitext(os.path.basename(args.input))[0]
    print("done")

    if X.ndim == 1:
        # mono to stereo
        X = np.asarray([X, X])

    print("stft of wave source...", end=" ")
    X_spec = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)
    print("done")

    sp = inference.Separator(
        model, device, args.batchsize, args.cropsize, args.postprocess
    )

    if args.tta:
        y_spec, v_spec = sp.separate_tta(X_spec)
    else:
        y_spec, v_spec = sp.separate(X_spec)

    # print("validating output directory...", end=" ")
    # output_dir = args.output_dir
    # if output_dir != "":  # modifies output_dir if theres an arg specified
    #     output_dir = output_dir.rstrip("/") + "/"
    #     os.makedirs(output_dir, exist_ok=True)
    # print("done")

    print("inverse stft of instruments...", end=" ")
    wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length)
    print("done")
    sf.write(f"{args.output_dir}/music/{basename}.mp3", wave.T, sr)

    print("inverse stft of vocals...", end=" ")
    wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length)
    print("done")
    sf.write(f"{args.output_dir}/vocal/{basename}.mp3", wave.T, sr)

    # if args.output_image:
    #     image = spec_utils.spectrogram_to_image(y_spec)
    #     utils.imwrite("{}{}_Instruments.jpg".format(output_dir, basename), image)
    #
    #     image = spec_utils.spectrogram_to_image(v_spec)
    #     utils.imwrite("{}{}_Vocals.jpg".format(output_dir, basename), image)


import pandas as pd


def check_duplicate(req_urls: list, list_file: str) -> list:
    log = pd.read_csv(list_file, index_col=False, sep=";")["url"]
    reqs = pd.Series(req_urls)

    result = reqs[~reqs.isin(log)]
    return result.tolist()


def check_dup_ids(req_ids: list, list_file: str) -> list:
    log = pd.read_csv(list_file, index_col=False, sep=";")["id"]
    reqs = pd.Series(req_ids)
    result = reqs[~reqs.isin(log)]
    return result.tolist()


def get_requests(request_file) -> list:
    with open(request_file, "r") as file:
        return file.read().split("\n")[:-1]


import yt_dlp


# download audio by id
def download_audio(save_path: str, vid_id: str, codec: str) -> str | None:
    with yt_dlp.YoutubeDL(
        {
            # download audio equivalent to yt-dlp -f bestaudio --extract-audio --audio-quality 0 <ID>
            "format": f"bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": codec,
                    "preferredquality": "0",
                }
            ],
            # save file to save_path
            "outtmpl": f"{save_path}/%(id)s",
        }
    ) as video:
        video.download(vid_id)
        info_dict = video.extract_info(vid_id)

        if info_dict is None:
            return None

        return info_dict["title"]


import re


def extract_video_id(url) -> str | None:
    video_id = ""
    regex = r"(?:https?:\/\/)?(?:www\.)?youtu\.?be(?:\.com)?\/?.*(?:watch|embed)?(?:.*v=|v\/|\/)([\w\-_]+)\&?"
    match = re.search(regex, url)
    if match and len(match.group(1)) == 11:
        video_id = match.group(1)
    else:
        return None
        # raise ValueError("Invalid YouTube URL")
    return video_id


def get_ids(request_file) -> list:
    with open(request_file, "r") as file:
        urls = file.read().split("\n")[:-1]
        ids = []
        for url in urls:
            vid_id = extract_video_id(url)
            if vid_id is None:
                print("\t" * 3 + f"Failed to extract video id from {url}")
            else:
                ids.append(vid_id)
        return ids


def check_dup_id(video_id: str, list_file: str) -> bool:
    audio_list = pd.read_csv(list_file, index_col=False, sep=";")["id"]
    return video_id in audio_list.values


import datetime

if __name__ == "__main__":
    request_file = "videoUrls.txt"
    audio_dir = "audio"
    log_file = "log.csv"
    list_audios = "split/list.xlsx"
    codec = "opus"

    with open(log_file, "a") as file:
        for url in get_requests(request_file):
            vid_id = extract_video_id(url)
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            valid = True
            is_duplicate = False

            # timestamp;url;valid;id;id_duplicate;title;download_status;split_status
            if vid_id is None:
                print("\t" * 6 + f"Failed to extract video id from {url}")
                valid = False
                file.write(f"{ts};{url};{valid};;;;;\n")
                continue

            if check_dup_id(vid_id, list_audios):
                print("\t" * 6 + f"Duplicate video id {vid_id}")
                is_duplicate = True
                file.write(f"{ts};{url};{valid};{vid_id};{is_duplicate};;;\n")
                continue

            vid_title = "None"
            download_status = "failed"
            split_status = "failed"
            try:
                vid_title = download_audio(audio_dir, vid_id, codec)
                download_status = "success"
                try:
                    split_audio(vid_id, codec)
                    split_status = "success"
                except:
                    pass
            except yt_dlp.DownloadError:
                download_status = "download error"

            file.write(
                f"{ts};{url};{valid};{vid_id};{is_duplicate};{vid_title};{download_status};{split_status}\n"
            )

            # save to videos list as google sheet
            if download_status == "success":
                with open(list_audios, "a") as list_file:
                    list_file.write(f"{vid_id};{vid_title};{url}\n")
