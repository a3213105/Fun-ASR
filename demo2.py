import numpy as np
import soundfile as sf
import torch

from tools.utils import load_audio

pt_model = None
pt_kwargs = None
ov_model = None

def pt_run(wav_path):
    from model import FunASRNano
    model_dir = "../Fun-ASR-Nano-2512"
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    if pt_model is None or pt_kwargs is None:
        pt_model, pt_kwargs = FunASRNano.from_pretrained(model=model_dir, device=device)
        pt_model.eval()

    tokenizer = pt_kwargs.get("tokenizer", None)

    res = pt_model.inference(data_in=[wav_path], **pt_kwargs)
    text = res[0][0]['text']
    print(text)

    chunk_size = 0.72
    duration = sf.info(wav_path).duration
    cum_durations = np.arange(chunk_size, duration + chunk_size, chunk_size)
    prev_text = ""
    for idx, cum_duration in enumerate(cum_durations):
        audio, rate = load_audio(wav_path, 16000, duration=round(cum_duration, 3))
        prev_text = m.inference([torch.tensor(audio)], prev_text=prev_text, **pt_kwargs)[0][0]["text"]
        if idx != len(cum_durations) - 1:
            prev_text = tokenizer.decode(tokenizer.encode(prev_text)[:-5]).replace("�", "")
    if prev_text:
        print(prev_text)


def ov_run(wav_path):
    from ov_operator_async import FunAsrNanoEncDecModel
    model_dir = "../Fun-ASR-Nano-2512-ov"
    if ov_model is None :
        ov_model = FunAsrNanoEncDecModel(ov_core=None,
                                        model_path=model_dir,
                                        enc_type="bf16",
                                        dec_type="bf16",
                                        cache_size=1024,
                                        disable_ctc=True)
    tokenizer = ov_model.tokenizer
    kwargs = {}

    res = ov_model.inference(data_in=[wav_path], **kwargs)
    text = res[0][0]['text']
    print(text)

    chunk_size = 0.72
    duration = sf.info(wav_path).duration
    cum_durations = np.arange(chunk_size, duration + chunk_size, chunk_size)
    prev_text = ""
    for idx, cum_duration in enumerate(cum_durations):
        audio, rate = load_audio(wav_path, 16000, duration=round(cum_duration, 3))
        prev_text = ov_model.inference([torch.tensor(audio)], prev_text=prev_text, **kwargs)[0][0]["text"]
        if idx != len(cum_durations) - 1:
            prev_text = tokenizer.decode(tokenizer.encode(prev_text)[:-5]).replace("�", "")
    if prev_text:
        print(prev_text)

def main() :
    wav_paths = ["./example/zh.mp3",
                 "./example/en.mp3",
                 "./M02_000085.wav",
                 ]
    for wav_path in wav_paths:
        ov_run(wav_path)
        print(f"################")
        pt_run(wav_path)
        print(f"################")

if __name__ == "__main__":
    main()
