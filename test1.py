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

    # ov_convert = FunAsrNanoConverterWrapper(m, kwargs, ov_model_dir)
    # res = ov_convert.inference(data_in=[wav_path], **kwargs)
    # text = res[0][0]["text"]
    # print(text)

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
    loop = 10

    results = ov_model.inference(data_in=[wav_path], **kwargs_ov)
    start_t = time.perf_counter()
    for i in range(loop):
        results = ov_model.inference(data_in=[wav_path], **kwargs_ov)
    d0 = time.perf_counter() - start_t
    print(f"### OV model inference time: {d0/loop:.2f} sec")
    text = results[0][0]["text"]
    print(text)

    outputs, contents, key, meta_data = ov_model.preprocess(data_in=[wav_path], **kwargs_ov)
    start_t = time.perf_counter()
    for i in range(loop):
        generated_ids, ctc_results = ov_model.predict(outputs, key, **kwargs)
    d0 = time.perf_counter() - start_t
    print(f"### OV model predict time: {d0/loop:.2f} sec")
    results = ov_model.postprocess(generated_ids, ctc_results, contents, key, meta_data, **kwargs)
    text = results[0][0]["text"]
    print(text)

    results = ov_ctc_model.inference(data_in=[wav_path], **kwargs_ov)
    start_t = time.perf_counter()
    for i in range(loop):
        results = ov_ctc_model.inference(data_in=[wav_path], **kwargs_ov)
    d0 = time.perf_counter() - start_t
    print(f"### OV CTC model inference time: {d0/loop:.2f} sec")
    text = results[0][0]["text"]
    print(text)

    outputs, contents, key, meta_data = ov_ctc_model.preprocess(data_in=[wav_path], **kwargs_ov)
    start_t = time.perf_counter()
    for i in range(loop):
        generated_ids, ctc_results = ov_ctc_model.predict(outputs, key, **kwargs)
    d0 = time.perf_counter() - start_t
    print(f"### OV CTC model predict time: {d0/loop:.2f} sec")
    results = ov_ctc_model.postprocess(generated_ids, ctc_results, contents, key, meta_data, **kwargs)
    text = results[0][0]["text"]
    print(text)
    
    kwargs['bf16'] = False
    kwargs['llm_dtype'] = 'fp32'
    results = m.inference(data_in=[wav_path], **kwargs)
    start_t = time.perf_counter()
    for i in range(loop):
        results = m.inference(data_in=[wav_path], **kwargs)
    d0 = time.perf_counter() - start_t
    print(f"### Torch fp32 model inference time: {d0/loop:.2f} sec")
    text = results[0][0]["text"]
    print(text)

    batch, meta_data, contents, key = m.preprocess(data_in=[wav_path], **kwargs)
    generated_ids, ctc_results, loss = m.predict(batch, key, meta_data, **kwargs)
    start_t = time.perf_counter()
    for i in range(loop):
        generated_ids, ctc_results, loss = m.predict(batch, key, meta_data, **kwargs)
    d0 = time.perf_counter() - start_t
    print(f"### Torch fp32 model predict time: {d0/loop:.2f} sec")
    results = m.postprocess(generated_ids, ctc_results, loss, key, contents, meta_data, **kwargs)
    text = results[0][0]["text"]
    print(text)

    kwargs['bf16'] = True
    kwargs['llm_dtype'] = 'bf16'
    results = m.inference(data_in=[wav_path], **kwargs)
    start_t = time.perf_counter()
    for i in range(loop):
        results = m.inference(data_in=[wav_path], **kwargs)
    d0 = time.perf_counter() - start_t
    print(f"### Torch bf16 model inference time: {d0/loop:.2f} sec")
    text = results[0][0]["text"]
    print(text)

    batch, meta_data, contents, key = m.preprocess(data_in=[wav_path], **kwargs)
    generated_ids, ctc_results, loss = m.predict(batch, key, meta_data, **kwargs)
    start_t = time.perf_counter()
    for i in range(loop):
        generated_ids, ctc_results, loss = m.predict(batch, key, meta_data, **kwargs)
    d0 = time.perf_counter() - start_t
    print(f"### Torch bf16 model predict time: {d0/loop:.2f} sec")
    results = m.postprocess(generated_ids, ctc_results, loss, key, contents, meta_data, **kwargs)
    text = results[0][0]["text"]
    print(text)
    
    kwargs['bf16'] = False
    kwargs['llm_dtype'] = 'bf16'
    results = m.inference(data_in=[wav_path], **kwargs)
    start_t = time.perf_counter()
    for i in range(loop):
        results = m.inference(data_in=[wav_path], **kwargs)
    d0 = time.perf_counter() - start_t
    print(f"### Torch f32-bf16 model inference time: {d0/loop:.2f} sec")
    text = results[0][0]["text"]
    print(text)

    batch, meta_data, contents, key = m.preprocess(data_in=[wav_path], **kwargs)
    generated_ids, ctc_results, loss = m.predict(batch, key, meta_data, **kwargs)
    start_t = time.perf_counter()
    for i in range(loop):
        generated_ids, ctc_results, loss = m.predict(batch, key, meta_data, **kwargs)
    d0 = time.perf_counter() - start_t
    print(f"### Torch f32-bf16 model predict time: {d0/loop:.2f} sec")
    results = m.postprocess(generated_ids, ctc_results, loss, key, contents, meta_data, **kwargs)
    text = results[0][0]["text"]
    print(text)
    
    kwargs['bf16'] = True
    kwargs['llm_dtype'] = 'fp32'
    results = m.inference(data_in=[wav_path], **kwargs)
    start_t = time.perf_counter()
    for i in range(loop):
        results = m.inference(data_in=[wav_path], **kwargs)
    d0 = time.perf_counter() - start_t
    print(f"### Torch bf16-f32 model inference time: {d0/loop:.2f} sec")
    text = results[0][0]["text"]
    print(text)

    batch, meta_data, contents, key = m.preprocess(data_in=[wav_path], **kwargs)
    generated_ids, ctc_results, loss = m.predict(batch, key, meta_data, **kwargs)
    start_t = time.perf_counter()
    for i in range(loop):
        generated_ids, ctc_results, loss = m.predict(batch, key, meta_data, **kwargs)
    d0 = time.perf_counter() - start_t
    print(f"### Torch bf16-f32 model predict time: {d0/loop:.2f} sec")
    results = m.postprocess(generated_ids, ctc_results, loss, key, contents, meta_data, **kwargs)
    text = results[0][0]["text"]
    print(text)

if __name__ == "__main__":
    main()

