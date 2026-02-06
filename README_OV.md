# 1. prepare models for OpenVINO
```bash
python convert_ov_model.py -p /PATH/TO/Fun-ASR-Nano/model -o /PATH/TO/Fun-ASR-Nano-OV/model
```

# 2. inference with OpenVINO
you can try demo.py for both PyTorch and OpenVINO inference
```bash
python demo.py -p /PATH/TO/Fun-ASR-Nano/model -o /PATH/TO/Fun-ASR-Nano-OV/model -a /PATH/TO/AUDIO/FILE -t B -c 5
```
or only using OpenVINO 
```bash
python demo_ov.py -o /PATH/TO/Fun-ASR-Nano-OV/model -a /PATH/TO/AUDIO/FILE -c 5
```
*Note the parameter chunk_size / c, which controls long‑audio segmentation. A smaller chunk size generally yields more accurate results, but it also slows down inference.*