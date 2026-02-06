
import argparse

from model import FunASRNano
from ov_model_helper import FunAsrNanoConverterWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Fun-ASR-Nano model to OpenVINO."
    )
    parser.add_argument(
        "--pt_model_dir", "-p",
        type=str,
        default="../Fun-ASR-Nano-2512",
        help="Path to the original Fun-ASR-Nano model directory (contains model weights & example).",
    )
    parser.add_argument(
        "--ov_model_dir", "-o",
        type=str,
        default="../Fun-ASR-Nano-2512-ov",
        help="Output directory for OpenVINO converted model.",
    )
    return parser.parse_args()


def main(args):
    # args.ov_model_dir = "../Fun-ASR-Nano-2512-ov"
    # pt_model_dir = "../Fun-ASR-Nano-2512"
    m, kwargs = FunASRNano.from_pretrained(model=args.pt_model_dir, device="cpu")
    m.eval()

    wav_path = f"{kwargs['model_path']}/example/zh.mp3"

    ov_convert = FunAsrNanoConverterWrapper(m, kwargs, args.ov_model_dir)
    res = ov_convert.inference(data_in=[wav_path], **kwargs)

if __name__ == "__main__":
    _args = parse_args()
    main(_args)

