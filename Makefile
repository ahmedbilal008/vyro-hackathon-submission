PYTHON ?= python

all: data train quantize eval_public

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

starter:
	$(PYTHON) starter/build_starter_files.py --manual data/manual_examples.jsonl --out starter

eval_public:
	$(PYTHON) eval_public.py --test starter/public_test.jsonl --out eval_public_summary.json

demo:
	MODEL_PATH=models/quantized/model.gguf $(PYTHON) demo/app.py
