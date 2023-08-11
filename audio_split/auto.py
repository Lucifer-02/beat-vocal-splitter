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
    args.gpu = 0
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

    print("validating output directory...", end=" ")
    output_dir = args.output_dir
    if output_dir != "":  # modifies output_dir if theres an arg specified
        output_dir = output_dir.rstrip("/") + "/"
        os.makedirs(output_dir, exist_ok=True)
    print("done")

    print("inverse stft of instruments...", end=" ")
    wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length)
    print("done")
    sf.write(f"{output_dir}/beats/{basename}.wav", wave.T, sr)

    print("inverse stft of vocals...", end=" ")
    wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length)
    print("done")
    sf.write(f"{output_dir}/vocals/{basename}.wav", wave.T, sr)

    if args.output_image:
        image = spec_utils.spectrogram_to_image(y_spec)
        utils.imwrite("{}{}_Instruments.jpg".format(output_dir, basename), image)

        image = spec_utils.spectrogram_to_image(v_spec)
        utils.imwrite("{}{}_Vocals.jpg".format(output_dir, basename), image)


import pandas as pd


def check_duplicate(req_urls: list, log_file: str) -> list:
    log = pd.read_csv(log_file, index_col=False)["url"]
    reqs = pd.Series(req_urls)

    result = reqs[~reqs.isin(log)]
    return result.tolist()


def get_requests(request_file) -> list:
    with open(request_file, "r") as file:
        return file.read().split("\n")[:-1]


import yt_dlp


def download_audio(save_path: str, link: str, codec: str):
    with yt_dlp.YoutubeDL(
        {
            "extract-audio": True,
            "format": "ba",
            "postprocessors": [
                {  # Extract audio using ffmpeg
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": codec,
                }
            ],
            "outtmpl": f"{save_path}/%(id)s",
        }
    ) as video:
        info_dict = video.extract_info(link, download=True)
        video.download(link)
        return info_dict["id"], info_dict["title"]


if __name__ == "__main__":
    request_file = "videoUrls.txt"
    audio_dir = "audio"
    log_file = "log.csv"
    codec = "opus"
    # split_audio("tA47kgzzY7M", codec)

    # df = pd.read_csv(log_file, index_col=False)
    # print(df["url"])

    with open(log_file, "a") as file:
        for url in check_duplicate(
            req_urls=get_requests(request_file), log_file=log_file
        ):
            download_status = "failed"
            split_status = "failed"
            vid_id = "None"
            vid_title = "None"
            try:
                vid_id, vid_title = download_audio(audio_dir, url, codec)
                download_status = "success"
                # try:
                #     split_audio(vid_id, codec)
                #     split_status = "success"
                # except:
                #     pass
            except yt_dlp.DownloadError:
                download_status = "download error"

            file.write(f"{url},{vid_id},{vid_title},{download_status},{split_status}\n")
