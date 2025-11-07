#!/usr/bin/env python3
"""
torch.compile Decode Speed Benchmark
Proper measurement with compilation overhead excluded
Uses Qwen chat template and realistic prompts
"""

import os
os.environ['HF_HOME'] = '/workspace/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/workspace/huggingface_cache'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc
import time
import numpy as np
from datetime import datetime
import json
import socket

HOSTNAME = socket.gethostname()
print("="*60)
print("TORCH.COMPILE DECODE SPEED BENCHMARK")
print("="*60)
print(f"\nSystem Info:")
print(f"  Hostname: {HOSTNAME}")
print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.version.cuda}")
print()

CACHE_DIR = '/workspace/huggingface_cache'
model_name = "Qwen/Qwen2.5-7B-Instruct"

print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    cache_dir=CACHE_DIR,
    low_cpu_mem_usage=True,
    device_map="auto"
)

# Qwen chat template - system + user messages
# Qwen uses: system message is optional, user/assistant alternation
test_conversations = [
    {
        "system": "You are a helpful and harmless AI assistant.",
        "user": "Explain the key differences between supervised and unsupervised machine learning. Include examples of when each approach is most useful."
    },
    {
        "system": "You are a helpful and harmless AI assistant.",
        "user": "Write a detailed explanation of how photosynthesis works, suitable for a high school biology student."
    },
    {
        "system": "You are a helpful and harmless AI assistant.",
        "user": "What are the main causes of climate change? Discuss both natural and human factors, and explain the scientific consensus."
    }
]

def format_conversation(conversation):
    """Format conversation using Qwen chat template"""
    messages = [
        {"role": "system", "content": conversation["system"]},
        {"role": "user", "content": conversation["user"]}
    ]
    
    # Use tokenizer's chat template
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return formatted

def benchmark_generation(model, tokenizer, conversations, max_new_tokens=200, num_runs=5, warmup_runs=2):
    """
    Benchmark generation speed
    Returns: list of times per conversation, tokens/sec stats
    """
    all_times = []
    all_tokens_per_sec = []
    
    for conv_idx, conversation in enumerate(conversations):
        prompt = format_conversation(conversation)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_tokens = inputs['input_ids'].shape[1]
        
        print(f"\n  Conversation {conv_idx + 1}:")
        print(f"    Prompt tokens: {prompt_tokens}")
        
        # Warmup
        if warmup_runs > 0:
            print(f"    Warming up ({warmup_runs} runs)...")
            for _ in range(warmup_runs):
                with torch.no_grad():
                    _ = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                torch.cuda.synchronize()
        
        # Timed runs
        print(f"    Timing ({num_runs} runs)...")
        run_times = []
        
        for run in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            # Count actual tokens generated
            generated_tokens = outputs.shape[1] - prompt_tokens
            tokens_per_sec = generated_tokens / elapsed
            
            run_times.append(elapsed)
            all_tokens_per_sec.append(tokens_per_sec)
            
            if run == 0:
                generated_text = tokenizer.decode(
                    outputs[0][prompt_tokens:],
                    skip_special_tokens=True
                )
                print(f"    Generated ({generated_tokens} tokens):")
                print(f"      '{generated_text[:150]}...'")
        
        mean_time = np.mean(run_times)
        std_time = np.std(run_times)
        mean_tps = np.mean(all_tokens_per_sec[-num_runs:])
        
        print(f"    Time: {mean_time:.3f}s ± {std_time:.3f}s")
        print(f"    Speed: {mean_tps:.1f} tokens/sec")
        
        all_times.extend(run_times)
        
        del inputs, outputs
        torch.cuda.empty_cache()
    
    return all_times, all_tokens_per_sec

max_new_tokens = 200
num_runs = 5
warmup_runs = 2

# ============================================================================
# PHASE 1: EAGER MODE
# ============================================================================

print("\n" + "="*60)
print("PHASE 1: EAGER MODE (no compilation)")
print("="*60)

eager_times, eager_tps = benchmark_generation(
    model, tokenizer, test_conversations,
    max_new_tokens=max_new_tokens,
    num_runs=num_runs,
    warmup_runs=warmup_runs
)

eager_mean = np.mean(eager_times)
eager_std = np.std(eager_times)
eager_mean_tps = np.mean(eager_tps)

print(f"\n{'='*60}")
print("EAGER MODE SUMMARY")
print(f"{'='*60}")
print(f"  Mean time: {eager_mean:.3f}s ± {eager_std:.3f}s")
print(f"  Mean speed: {eager_mean_tps:.1f} tokens/sec")

# Clean up eager model
del model
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

# ============================================================================
# PHASE 2: COMPILED MODE
# ============================================================================

print("\n" + "="*60)
print("PHASE 2: COMPILED MODE")
print("="*60)

# Reload model
print("\nReloading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    cache_dir=CACHE_DIR,
    low_cpu_mem_usage=True,
    device_map="auto"
)

# Compile
print("\nCompiling model with mode='default'...")
model.forward = torch.compile(model.forward, mode="default")
print("✓ Model compiled")

# Trigger compilation with a dummy run
print("\nTriggering compilation (this is slow, ~1-2 minutes)...")
dummy_prompt = format_conversation(test_conversations[0])
dummy_inputs = tokenizer(dummy_prompt, return_tensors="pt").to("cuda")

torch.cuda.synchronize()
compile_start = time.perf_counter()

with torch.no_grad():
    _ = model.generate(
        **dummy_inputs,
        max_new_tokens=50,  # Shorter for compilation
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

torch.cuda.synchronize()
compile_time = time.perf_counter() - compile_start

print(f"✓ Compilation complete ({compile_time:.1f}s)")
print(f"  (This overhead is excluded from speed measurements)")

del dummy_inputs
torch.cuda.empty_cache()

# Now benchmark with compiled model
print("\nBenchmarking compiled model...")

compiled_times, compiled_tps = benchmark_generation(
    model, tokenizer, test_conversations,
    max_new_tokens=max_new_tokens,
    num_runs=num_runs,
    warmup_runs=warmup_runs
)

compiled_mean = np.mean(compiled_times)
compiled_std = np.std(compiled_times)
compiled_mean_tps = np.mean(compiled_tps)

print(f"\n{'='*60}")
print("COMPILED MODE SUMMARY")
print(f"{'='*60}")
print(f"  Mean time: {compiled_mean:.3f}s ± {compiled_std:.3f}s")
print(f"  Mean speed: {compiled_mean_tps:.1f} tokens/sec")
print(f"  Compilation overhead: {compile_time:.1f}s (excluded)")

# ============================================================================
# COMPARISON
# ============================================================================

speedup = eager_mean / compiled_mean
tps_improvement = compiled_mean_tps / eager_mean_tps

print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)
print(f"\nTime per generation:")
print(f"  Eager:    {eager_mean:.3f}s ± {eager_std:.3f}s")
print(f"  Compiled: {compiled_mean:.3f}s ± {compiled_std:.3f}s")
print(f"  Speedup:  {speedup:.2f}x")

print(f"\nThroughput:")
print(f"  Eager:    {eager_mean_tps:.1f} tokens/sec")
print(f"  Compiled: {compiled_mean_tps:.1f} tokens/sec")
print(f"  Improvement: {tps_improvement:.2f}x ({(tps_improvement-1)*100:.1f}% faster)")

print(f"\nCompilation cost:")
print(f"  One-time overhead: {compile_time:.1f}s")
print(f"  Break-even after: {compile_time / (eager_mean - compiled_mean):.0f} generations")

# ============================================================================
# SAVE RESULTS
# ============================================================================

output = {
    "experiment": "torch_compile_decode_speed_benchmark",
    "timestamp": datetime.now().isoformat(),
    "model": model_name,
    "hardware": {
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "hostname": HOSTNAME
    },
    "config": {
        "max_new_tokens": max_new_tokens,
        "num_conversations": len(test_conversations),
        "runs_per_conversation": num_runs,
        "warmup_runs": warmup_runs,
        "compile_mode": "default"
    },
    "results": {
        "eager": {
            "mean_time_sec": float(eager_mean),
            "std_time_sec": float(eager_std),
            "mean_tokens_per_sec": float(eager_mean_tps),
            "all_times": [float(t) for t in eager_times]
        },
        "compiled": {
            "mean_time_sec": float(compiled_mean),
            "std_time_sec": float(compiled_std),
            "mean_tokens_per_sec": float(compiled_mean_tps),
            "compilation_time_sec": float(compile_time),
            "all_times": [float(t) for t in compiled_times]
        },
        "comparison": {
            "speedup": float(speedup),
            "throughput_improvement": float(tps_improvement),
            "break_even_generations": float(compile_time / (eager_mean - compiled_mean)) if eager_mean > compiled_mean else None
        }
    }
}

output_file = f"torch_compile_speed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
output_path = f"/workspace/{output_file}"

with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Results saved to {output_path}")

print("\n" + "="*60)
print("BENCHMARK COMPLETE")
print("="*60)
print(f"\nKey findings:")
print(f"  • Speedup: {speedup:.2f}x")
print(f"  • Throughput improvement: {(tps_improvement-1)*100:.1f}%")
print(f"  • Compilation overhead: {compile_time:.1f}s")
print(f"  • Worth it for production: {'✓ Yes' if speedup > 1.1 else '⚠ Marginal' if speedup > 1.0 else '✗ No'}")
print("="*60)
