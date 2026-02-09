import numpy as np
import soundfile as sf
import torch
import argparse

from tools.utils import load_audio

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Fun-ASR-Nano inference with PyTorch or OpenVINO."
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
        default="../Fun-ASR-Nano-2512-OV",
        help="Path to the OpenVINO Fun-ASR-Nano model directory.",
    )
    parser.add_argument(
        "--type", "-t",
        type=str,
        default="B",
        help="O for OpenVINO, P for PyTorch. B for both",
    )
    parser.add_argument(
        "--audio", "-a",
        type=str,
        default="./M02_000085.wav",
        help="Optional path to an input audio file (wav/mp3/flac...). "
             "If not set, defaults to: ./M02_000085.wav",
    )
    parser.add_argument(
        "--chunk_size", "-c",
        type=int,
        default=4,
        help="chunk_size for split long term audio. small chunk size make more accurate results, but slower inference. "
    )
    return parser.parse_args()

def pt_run_short(pt_model, kwargs, wav_path):
    tokenizer = kwargs.get("tokenizer", None)
    res = pt_model.inference(data_in=[wav_path], **kwargs)
    # text = res[0][0]['text']
    text = res[0][0]    
    print(f"Results: {text}")

def pt_run_long(pt_model, kwargs, wav_path):
    tokenizer = kwargs.get("tokenizer", None)
    # chunk_size = 0.72
    chunk_size = kwargs.get('chunk_size', 0.72)
    duration = sf.info(wav_path).duration
    cum_durations = np.arange(chunk_size, duration + chunk_size, chunk_size)
    prev_text = ""
    for idx, cum_duration in enumerate(cum_durations):
        audio, rate = load_audio(wav_path, 16000, duration=round(cum_duration, 3))
        prev_text = pt_model.inference([torch.tensor(audio)], prev_text=prev_text, **kwargs)[0][0]["text"]
        if idx != len(cum_durations) - 1:
            prev_text = tokenizer.decode(tokenizer.encode(prev_text)[:-5]).replace("�", "")
    # if prev_text:
    print(f"Results: {prev_text}")

def pt_run(model_dir, wav_path, chunk_size) :
    print(f"### RUN PyTorch Inference ###， model_dir={model_dir}， wav_path={wav_path}, chunk_size={chunk_size}")
    from model import FunASRNano
    # model_dir = "../Fun-ASR-Nano-2512"
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    pt_model, kwargs = FunASRNano.from_pretrained(model=model_dir, device=device)
    pt_model.eval()

    kwargs['language']="中文"
    kwargs['itn']=True
    kwargs['chunk_size']=chunk_size

    duration = sf.info(wav_path).duration
    if duration > 30:
        pt_run_long(pt_model, kwargs, wav_path)
    else :
        pt_run_short(pt_model, kwargs, wav_path)

def ov_run_short(ov_model, kwargs, wav_path):
    res = ov_model.inference(data_in=[wav_path], **kwargs)
    # text = res[0][0]['text']
    text = res[0][0]
    print(f"Results: {text}")

def ov_run_long(ov_model, kwargs, wav_path):
    chunk_size = kwargs.get('chunk_size', 0.72)
    tokenizer = kwargs.get("tokenizer", None)
    duration = sf.info(wav_path).duration
    cum_durations = np.arange(chunk_size, duration + chunk_size, chunk_size)
    prev_text = ""
    for idx, cum_duration in enumerate(cum_durations):
        audio, rate = load_audio(wav_path, 16000, duration=round(cum_duration, 3))
        prev_text = ov_model.inference([torch.tensor(audio)], prev_text=prev_text, **kwargs)[0][0]["text"]
        if idx != len(cum_durations) - 1:
            prev_text = tokenizer.decode(tokenizer.encode(prev_text)[:-5]).replace("�", "")
    # if prev_text:
    #     print(prev_text)
    print(f"Results: {prev_text}")

def ov_run(model_dir, wav_path, chunk_size, disable_ctc=True) :
    print(f"### RUN OpenVINO Inference ### disable_ctc={disable_ctc}, chunk_size={chunk_size}")
    from ov_operator_async import FunAsrNanoEncDecModel
    # model_dir = "../Fun-ASR-Nano-2512-ov"
    ov_model = FunAsrNanoEncDecModel(ov_core=None,
                                        model_path=model_dir,
                                        enc_type="f32",
                                        dec_type="f32",
                                        cache_size=1024,
                                        disable_ctc=disable_ctc)
    kwargs = {}
    kwargs['tokenizer']=ov_model.tokenizer
    #kwargs['hotwords']=["开放时间"]
    kwargs['language']="中文"
    kwargs['itn']=True
    kwargs['chunk_size']=chunk_size

    duration = sf.info(wav_path).duration
    if duration > 30:
        ov_run_long(ov_model, kwargs, wav_path)
    else :
        ov_run_short(ov_model, kwargs, wav_path)


def main(args) :
    chunk_size = args.chunk_size
    if args.type.lower() == 'b' or args.type.lower() == 'o':
        ov_run(args.ov_model_dir, args.audio, args.chunk_size, True)
        ov_run(args.ov_model_dir, args.audio, args.chunk_size, False)
    if args.type.lower() == 'b' or args.type.lower() == 'p':
        pt_run(args.pt_model_dir, args.audio, args.chunk_size)

if __name__ == "__main__":
    _args = parse_args()
    main(_args)
