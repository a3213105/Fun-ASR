from model import FunASRNano
from ov_model_helper import FunAsrNanoConverterWrapper
from ov_operator_async import FunAsrNanoEncDecModel
import time

def main():
    ov_model_dir = "../Fun-ASR-Nano-2512-ov"
    model_dir = "../Fun-ASR-Nano-2512"
    m, kwargs = FunASRNano.from_pretrained(model=model_dir, device="cpu")
    # m = m.float().eval()
    # m.llm.float().eval()
    m.eval()
    m.llm.config._attn_implementation = "eager"

    wav_path = f"{kwargs['model_path']}/example/zh.mp3"
    kwargs['hotwords']=["开放时间"]
    kwargs['language']="中文"
    kwargs['itn']=True

    ov_convert = FunAsrNanoConverterWrapper(m, kwargs, ov_model_dir)
    res = ov_convert.inference(data_in=[wav_path], **kwargs)
    text = res[0][0]["text"]
    print(text)

    ov_model = FunAsrNanoEncDecModel(ov_core=None,
                                    model_path=ov_model_dir,
                                    enc_type="bf16",
                                    dec_type="bf16",
                                    cache_size=32,
                                    disable_ctc = True)
    
    ov_ctc_model = FunAsrNanoEncDecModel(ov_core=None,
                                    model_path=ov_model_dir,
                                    enc_type="bf16",
                                    dec_type="bf16",
                                    cache_size=32)


    # kwargs_ov = {"frontend": ov_model.frontend, "tokenizer": ov_model.tokenizer}
    kwargs_ov = {}
    kwargs_ov['hotwords']=["开放时间"]
    kwargs_ov['language']="中文"
    kwargs_ov['itn']=True

    res = ov_model.inference(data_in=[wav_path], **kwargs_ov)
    text = res[0][0]["text"]
    print(text)
    loop = 1
    start_t = time.perf_counter()
    for i in range(loop):
        res = ov_model.inference(data_in=[wav_path], **kwargs_ov)
    d0 = time.perf_counter() - start_t
    print(f"### OV model inference time: {d0/loop:.2f} sec")
    
    res = ov_ctc_model.inference(data_in=[wav_path], **kwargs_ov)
    text = res[0][0]["text"]
    print(text)
    start_t = time.perf_counter()
    for i in range(loop):
        res = ov_ctc_model.inference(data_in=[wav_path], **kwargs_ov)
    d0 = time.perf_counter() - start_t
    print(f"### OV CTC model inference time: {d0/loop:.2f} sec")

    res = m.inference(data_in=[wav_path], **kwargs)
    text = res[0][0]["text"]
    print(text)
    start_t = time.perf_counter()
    for i in range(loop):
        res = m.inference(data_in=[wav_path], **kwargs)
    d0 = time.perf_counter() - start_t
    print(f"### Torch model inference time: {d0/loop:.2f} sec")



if __name__ == "__main__":
    main()

