import numpy as np
import soundfile as sf
import torch
import argparse
import os
import time

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

def run_short(ov_model, kwargs, wav_path):
    st = time.perf_counter()
    res = ov_model.inference(data_in=[wav_path], **kwargs)
    latency = time.perf_counter() - st
    text = res[0][0]['text']
    print(f"{wav_path}: {text}")
    return latency

def run_long(ov_model, kwargs, wav_path):
    chunk_size = kwargs.get('chunk_size', 0.72)
    tokenizer = kwargs.get("tokenizer", None)
    duration = sf.info(wav_path).duration
    st = time.perf_counter()
    cum_durations = np.arange(chunk_size, duration + chunk_size, chunk_size)
    print(f"cum_durations={cum_durations}")
    prev_text = ""
    for idx, cum_duration in enumerate(cum_durations):
        audio, rate = load_audio(wav_path, 16000, duration=round(cum_duration, 3))
        prev_text = ov_model.inference([torch.tensor(audio)], prev_text=prev_text, **kwargs)[0][0]["text"]
        if idx != len(cum_durations) - 1:
            prev_text = tokenizer.decode(tokenizer.encode(prev_text)[:-5]).replace("�", "")
    latency = time.perf_counter() - st
    print(f"{wav_path}: {prev_text}")
    return latency

def run(mode, model, wav_path, kwargs) :
    if os.path.isdir(wav_path):
        audio_files = sorted([
            (os.path.join(wav_path, file_name))
            for file_name in os.listdir(wav_path)
            if os.path.isfile(os.path.join(wav_path, file_name))
        ])
    else :
        audio_files = [wav_path]
    
    total_duration = 0.0
    total_latency = 0.0
    for filepath in audio_files:
        duration = sf.info(filepath).duration
        total_duration += duration
        if duration > 30:
            latency = run_long(model, kwargs, filepath)
        else :
            latency = run_short(model, kwargs, filepath)
        total_latency += latency
    rtf = total_latency / total_duration if total_duration > 0 else float('inf')
    print(f"{mode} inference rtf : {rtf:.3f}")

def ov_run(model_dir, wav_path, chunk_size) :
    print(f"### RUN OpenVINO Inference ###")
    from ov_operator_async import FunAsrNanoEncDecModel
    # model_dir = "../Fun-ASR-Nano-2512-ov"
    ov_model = FunAsrNanoEncDecModel(ov_core=None,
                                        model_path=model_dir,
                                        enc_type="f16",
                                        dec_type="bf16",
                                        cache_size=1024,
                                        for_dialect=True,
                                        disable_ctc=True)
    kwargs = {}
    kwargs['tokenizer']=ov_model.tokenizer
    #kwargs['hotwords']=["开放时间"]
    # kwargs['language']="中文"
    # kwargs['itn']=True
    kwargs['chunk_size']=chunk_size
    return run("OpenVINO", ov_model, wav_path, kwargs)

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
    kwargs['dialect']=True
    kwargs['chunk_size']=chunk_size
    return run("PyTorch", pt_model, wav_path, kwargs)

def main(args) :
    chunk_size = args.chunk_size
    if args.type.lower() == 'b' or args.type.lower() == 'o':
        ov_run(args.ov_model_dir, args.audio, args.chunk_size)
    if args.type.lower() == 'b' or args.type.lower() == 'p':
        pt_run(args.pt_model_dir, args.audio, args.chunk_size)

if __name__ == "__main__":
    _args = parse_args()
    main(_args)
