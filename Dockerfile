FROM nvcr.io/nvidia/pytorch:24.04-py3

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir triton bitsandbytes transformers lm_eval accelerate tqdm datasets

RUN sed -i 's/tl\.libdevice\.llrint/tl\.extra\.cuda\.libdevice\.llrint/g' \
    /usr/local/lib/python3.10/dist-packages/bitsandbytes/triton/quantize_global.py \
    /usr/local/lib/python3.10/dist-packages/bitsandbytes/triton/quantize_rowwise.py \
    /usr/local/lib/python3.10/dist-packages/bitsandbytes/triton/quantize_columnwise_and_transpose.py
    
CMD ["bash"]    
