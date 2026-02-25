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
        default="../Fun-ASR-Nano-2512-ov",
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
    parser.add_argument(
        "--loop", "-l",
        type=int,
        default=5,
        help="loop count for benchmark. "
    )
    parser.add_argument(
        "--dialect", "-d",
        action="store_true",
        default=True,
        help="Using for dialect detection (default: True).",
    )
    parser.add_argument(
        "--no-dialect",
        dest="dialect",
        action="store_false",
        help="Disable dialect detection.",
    )
    parser.add_argument(
        "--dec_dtype",
        type=str,
        choices=["f32", "bf16", "f16"],
        default="BF16",
        help="Data type for OpenVINO models: F32/BF16/F16 (default: BF16).",
    )
    parser.add_argument(
        "--enc_dtype",
        type=str,
        choices=["f32", "bf16", "f16"],
        default="BF16",
        help="Data type for OpenVINO models: F32/BF16/F16 (default: BF16).",
    )
    parser.add_argument(
        "--cores", "-n",
        type=int,
        default=4,
        help="loop count for benchmark. "
    )
    return parser.parse_args()

def run_short(ov_model, kwargs, wav_path):
    st = time.perf_counter()
    res = ov_model.inference(data_in=[wav_path], **kwargs)
    latency = time.perf_counter() - st
    text = res[0][0]['text']
    print(f"run_short {wav_path}: {text}")
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
    print(f"run_long {wav_path}: {prev_text}")
    return latency

def run(mode, model, wav_path, loop, kwargs) :
    if os.path.isdir(wav_path):
        audio_files = sorted([
            (os.path.join(wav_path, file_name))
            for file_name in os.listdir(wav_path)
            if os.path.isfile(os.path.join(wav_path, file_name))
        ])
    else :
        audio_files = [wav_path]

    for filepath in audio_files:
        duration = sf.info(filepath).duration
        if duration > 30:
            latency = run_long(model, kwargs, filepath)
        else :
            latency = run_short(model, kwargs, filepath)
    
    print(f"{mode} inference benchmark {loop} times...")
    total_duration = 0.0
    total_latency = 0.0
    for _ in range(loop):
        for filepath in audio_files:
            duration = sf.info(filepath).duration
            total_duration += duration
            if duration > 30:
                latency = run_long(model, kwargs, filepath)
            else :
                latency = run_short(model, kwargs, filepath)
            total_latency += latency
    rtf = total_latency / total_duration if total_duration > 0 else float('inf')
    print(f"{mode} inference average rtf : {rtf:.3f}")

def ov_run(model_dir, wav_path, chunk_size, loop, enc_dtype, dec_dtype, dialect) :
    name = f"OpenVINO_{enc_dtype}_{dec_dtype}"
    print(f"### RUN {name} Inference ###")
    from ov_operator_async import FunAsrNanoEncDecModel
    ov_model = FunAsrNanoEncDecModel(ov_core=None,
                                        model_path=model_dir,
                                        enc_type=enc_dtype,
                                        dec_type=dec_dtype,
                                        cache_size=1024,
                                        for_dialect=dialect,
                                        disable_ctc=True)
    kwargs = {}
    kwargs['tokenizer']=ov_model.tokenizer
    kwargs['chunk_size']=chunk_size
    if dialect :
        kwargs['dialect'] = True
    else :
        kwargs['dialect'] = False
        kwargs['language']="中文"
        kwargs['itn']=True
    return run(name, ov_model, wav_path, loop, kwargs)

def pt_run(model_dir, wav_path, chunk_size, loop, cores, dialect) :
    print(f"### RUN PyTorch Inference ###， model_dir={model_dir}， wav_path={wav_path}, chunk_size={chunk_size}")
    from model import FunASRNano
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    pt_model, kwargs = FunASRNano.from_pretrained(model=model_dir, device=device, ncpu=cores)
    pt_model.eval()

    kwargs['chunk_size']=chunk_size

    if dialect :
        kwargs['dialect'] = True
    else :
        kwargs['dialect'] = False
        kwargs['language']="中文"
        kwargs['itn']=True
    return run("PyTorch", pt_model, wav_path, loop, kwargs)

def main(args) :
    chunk_size = args.chunk_size
    if args.type.lower() == 'a' or args.type.lower() == 'b' or args.type.lower() == 'p':
        pt_run(args.pt_model_dir, args.audio, args.chunk_size, args.loop, args.cores, args.dialect)
    if args.type.lower() == 'a' :
        ov_run(args.ov_model_dir, args.audio, args.chunk_size, args.loop, 'f32', 'f32', args.dialect)
        ov_run(args.ov_model_dir, args.audio, args.chunk_size, args.loop, 'bf16', 'bf16', args.dialect)
        ov_run(args.ov_model_dir, args.audio, args.chunk_size, args.loop, 'f16', 'bf16', args.dialect)
        # ov_run(args.ov_model_dir, args.audio, args.chunk_size, args.loop, 'f16', args.dialect)
    elif args.type.lower() == 'b' or args.type.lower() == 'o':
        ov_run(args.ov_model_dir, args.audio, args.chunk_size, args.loop, args.enc_dtype, args.dec_dtype, args.dialect)

if __name__ == "__main__":
    _args = parse_args()
    main(_args)
