from subprocess import call
from pathlib import Path


def split(input: Path, output: Path, tta: bool = True, gpu: bool = True):
    command = [
        "python3",
        "./vocal-remover/inference.py",
        "--input",
        str(input),
        "--output_dir",
        str(output),
    ]

    if tta:
        command.append("--tta")

    if gpu:
        command.append("--gpu")
        command.append("0")

    call(command)
