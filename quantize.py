import argparse
import gc
import os
import subprocess
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def run(cmd, cwd=None, env=None):
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def build_cmake_target(build_dir, target, jobs):
    safe_jobs = max(1, int(jobs))
    print(f"Building {target} with {safe_jobs} job(s)")
    run(
        [
            "cmake",
            "--build",
            build_dir,
            "--config",
            "Release",
            "--target",
            target,
            "--parallel",
            str(safe_jobs),
        ]
    )


def find_quantize_bin(path):
    candidates = [
        os.path.join(path, "llama-quantize"),
        os.path.join(path, "quantize"),
        os.path.join(path, "build", "bin", "llama-quantize"),
        os.path.join(path, "build", "bin", "quantize"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return ""


def ensure_llama_cpp(path, build_jobs=1):
    if not os.path.exists(path):
        run(["git", "clone", "https://github.com/ggml-org/llama.cpp", path])

    quantize_bin = find_quantize_bin(path)
    if quantize_bin:
        return quantize_bin

    build_dir = os.path.join(path, "build")
    run(
        [
            "cmake",
            "-S",
            path,
            "-B",
            build_dir,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
        ]
    )

    job_attempts = [max(1, int(build_jobs))]
    if 1 not in job_attempts:
        job_attempts.append(1)

    for target in ("llama-quantize", "quantize"):
        for jobs in job_attempts:
            try:
                build_cmake_target(build_dir, target, jobs)
            except subprocess.CalledProcessError:
                print(f"Build failed for target={target} jobs={jobs}")
                continue

            quantize_bin = find_quantize_bin(path)
            if quantize_bin:
                return quantize_bin

    quantize_bin = find_quantize_bin(path)
    if quantize_bin:
        return quantize_bin

    # Legacy fallback for older llama.cpp versions.
    try:
        run(["make", "-C", path, "llama-quantize", "-j1"])
    except subprocess.CalledProcessError:
        run(["make", "-C", path, "quantize", "-j1"])

    quantize_bin = find_quantize_bin(path)
    if not quantize_bin:
        raise FileNotFoundError("Unable to locate llama.cpp quantize binary after build")
    return quantize_bin


def merge_lora(base_id, adapter_dir, merged_dir):
    os.makedirs(merged_dir, exist_ok=True)
    use_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        device_map={"": 0} if use_cuda else "cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    tokenizer.save_pretrained(merged_dir)
    del model
    del tokenizer
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()


def convert_and_quantize(merged_dir, out_path, llama_dir, quant_type, quantize_bin):
    f16_path = out_path + ".f16.gguf"
    convert_script = os.path.join(llama_dir, "convert_hf_to_gguf.py")
    run([sys.executable, convert_script, "--outtype", "f16", "--outfile", f16_path, merged_dir])
    run([quantize_bin, f16_path, out_path, quant_type])


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default="models/adapter")
    parser.add_argument("--merged", default="models/merged")
    parser.add_argument("--out", default="models/quantized/model.gguf")
    parser.add_argument("--llama_dir", default="llama.cpp")
    parser.add_argument("--quant", default="q4_k_m")
    parser.add_argument("--build_jobs", type=int, default=1)
    parser.add_argument("--max_mb", type=float, default=0.0)
    parser.add_argument("--reuse_merged", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    quantize_bin = ensure_llama_cpp(args.llama_dir, build_jobs=args.build_jobs)
    merged_ready = os.path.exists(os.path.join(args.merged, "config.json"))
    if args.reuse_merged and merged_ready:
        print(f"Using existing merged model at {args.merged}")
    else:
        merge_lora(args.base, args.adapter, args.merged)
    convert_and_quantize(args.merged, args.out, args.llama_dir, args.quant, quantize_bin)
    size_mb = file_size_mb(args.out)
    print(f"Wrote quantized model to {args.out} ({size_mb:.2f} MB)")
    if args.max_mb > 0 and size_mb > args.max_mb:
        raise SystemExit(f"Quantized model exceeds size limit: {size_mb:.2f} MB > {args.max_mb:.2f} MB")


if __name__ == "__main__":
    main()
