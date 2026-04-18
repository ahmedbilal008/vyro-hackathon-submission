PYTHON ?= python

all: data train quantize

setup:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) data/generate_data.py --out data/train.jsonl --n 2000 --manual data/manual_examples.jsonl

train:
	$(PYTHON) train.py --model_id Qwen/Qwen2.5-0.5B-Instruct --data data/train.jsonl --out models/adapter

quantize:
	$(PYTHON) quantize.py --base Qwen/Qwen2.5-0.5B-Instruct --adapter models/adapter --out models/quantized/model.gguf

quantize_bonus:
	$(PYTHON) quantize.py --base Qwen/Qwen2.5-0.5B-Instruct --adapter models/adapter --out models/quantized/model.gguf --quant q3_k_s --max_mb 250

demo:
	MODEL_PATH=models/quantized/model.gguf $(PYTHON) demo/app.py
