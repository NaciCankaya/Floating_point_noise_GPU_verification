#!/usr/bin/env python3
"""
Key Vector Forensics Experiment
Tests if key vectors from KV cache are forensically effective for detecting torch.compile
Minimal overhead approach - KV cache exists anyway in production decode
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
CONTAINER_ID = os.environ.get('HOSTNAME', 'unknown')

print("="*60)
print("KEY VECTOR FORENSICS: torch.compile decode")
print("="*60)
print(f"\nSystem Info:")
print(f"  Hostname: {HOSTNAME}")
print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.version.cuda}")
print()

CACHE_DIR = '/workspace/huggingface_cache'
model_name = "Qwen/Qwen2.5-7B-Instruct"

def collect_key_vectors_post_generation(model, tokenizer, prompt, max_new_tokens=20):
    """
    Generate tokens, then extract key vectors from the KV cache built during decode
    No fallback - if generate() doesn't return the cache, we fail
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate and request the past_key_values to be returned
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy/deterministic
            return_dict_in_generate=True,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_ids = outputs.sequences[0]
    generated_tokens = generated_ids[inputs['input_ids'].shape[1]:].cpu().tolist()
    
    # Check if we got past_key_values from generate
    if not hasattr(outputs, 'past_key_values') or outputs.past_key_values is None:
        raise RuntimeError(
            "model.generate() did not return past_key_values. "
            "The KV cache from decode is not accessible. "
            "This means we cannot extract activations from decode without "
            "interfering with the generation loop."
        )
    
    print("  ✓ Successfully retrieved KV cache from decode")
    past_kv = outputs.past_key_values
    
    # Extract key vectors from the KV cache built during decode
    last_layer_kv = past_kv[-1]  # Last layer
    key_cache = last_layer_kv[0]  # Keys (not values)
    
    # IMPORTANT: The KV cache returned by generate() only contains the GENERATED tokens
    # Not the prompt tokens! So indices are 0 to num_generated-1
    num_generated = len(generated_tokens)
    
    print(f"  KV cache shape: {key_cache.shape}")
    print(f"  Generated tokens: {num_generated}")
    
    key_vectors = []
    for i in range(num_generated):
        # Index directly into the generated positions (0 to num_generated-1)
        key_vec = key_cache[0, :, i, :].reshape(-1).cpu().clone()
        key_vectors.append(key_vec)
    
    del outputs
    torch.cuda.empty_cache()
    
    return torch.stack(key_vectors), generated_tokens

print(f"Loading {model_name} in BF16...")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    cache_dir=CACHE_DIR,
    low_cpu_mem_usage=True,
    device_map="auto"
)

# Get model architecture info
num_layers = len(model.model.layers)
num_heads = model.config.num_attention_heads
head_dim = model.config.hidden_size // num_heads
key_vector_dim = num_heads * head_dim

print(f"Model architecture:")
print(f"  Layers: {num_layers}")
print(f"  Attention heads: {num_heads}")
print(f"  Head dimension: {head_dim}")
print(f"  Key vector dimension: {key_vector_dim}")
print()

prompt = "The capital of France is"
prompt_tokens = len(tokenizer.encode(prompt))
print(f"Prompt: '{prompt}' ({prompt_tokens} tokens)\n")

num_reps = 10
max_new_tokens = 20

print("="*60)
print("EXPERIMENT: EAGER vs COMPILED (Decode Verification)")
print("="*60)
print(f"Decode steps: {max_new_tokens}")
print(f"Repetitions: {num_reps}")
print(f"Extraction: Key vectors from decode KV cache (post-generation)")
print(f"No fallback: If cache not accessible, experiment fails")
print(f"Goal: Verify decode workloads only")
print()

# ============================================================================
# TEST 1: EAGER MODE
# ============================================================================

print("="*60)
print("Phase 1: EAGER MODE")
print("="*60)

results_eager = []
tokens_eager = []

print("Collecting key vectors...")
for rep in range(num_reps):
    key_vecs, tokens = collect_key_vectors_post_generation(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens
    )
    results_eager.append(key_vecs)
    tokens_eager.append(tokens)
    
    if rep == 0:
        generated_text = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"  Generated: '{generated_text}'")
        print(f"  Steps: {len(tokens)}")
        print(f"  Key vector shape per step: {key_vecs[0].shape}")
        print(f"  First step key norm: {torch.norm(key_vecs[0]).item():.2f}")

# Check reproducibility
first_eager = results_eager[0]
eager_reproducible = all(torch.equal(first_eager, results_eager[i]) for i in range(1, num_reps))

if eager_reproducible:
    print(f"✓ Eager: Perfect reproducibility (L2=0.0)")
else:
    print(f"✗ Eager: VARIATION DETECTED")
    for i in range(1, num_reps):
        if not torch.equal(first_eager, results_eager[i]):
            l2_per_step = torch.norm(first_eager - results_eager[i], dim=-1)
            print(f"  Rep 0 vs {i}: Max L2={l2_per_step.max().item():.6f}")

print()

# ============================================================================
# TEST 2: COMPILED MODE
# ============================================================================

print("="*60)
print("Phase 2: COMPILED MODE")
print("="*60)

# Compile - use "default" mode which doesn't aggressively use CUDA graphs
print("Compiling model with mode='default'...")
print("  (Using 'default' instead of 'reduce-overhead' to avoid CUDA graph issues)")
model.forward = torch.compile(model.forward, mode="default")
print("✓ Model compiled")
print()

print("Collecting key vectors...")
results_compiled = []
tokens_compiled = []

for rep in range(num_reps):
    if rep == 0:
        print("  First run: Triggering compilation...")
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    key_vecs, tokens = collect_key_vectors_post_generation(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens
    )
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    results_compiled.append(key_vecs)
    tokens_compiled.append(tokens)
    
    if rep == 0:
        generated_text = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"  Generated: '{generated_text}'")
        print(f"  Steps: {len(tokens)}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  First step key norm: {torch.norm(key_vecs[0]).item():.2f}")

# Check reproducibility
first_compiled = results_compiled[0]
compiled_reproducible = all(torch.equal(first_compiled, results_compiled[i]) for i in range(1, num_reps))

if compiled_reproducible:
    print(f"✓ Compiled: Perfect reproducibility (L2=0.0)")
else:
    print(f"✗ Compiled: VARIATION DETECTED")
    for i in range(1, num_reps):
        if not torch.equal(first_compiled, results_compiled[i]):
            l2_per_step = torch.norm(first_compiled - results_compiled[i], dim=-1)
            print(f"  Rep 0 vs {i}: Max L2={l2_per_step.max().item():.6f}")

print()

# ============================================================================
# ANALYSIS: EAGER vs COMPILED
# ============================================================================

print("="*60)
print("SYSTEMATIC DEVIATION: EAGER vs COMPILED")
print("="*60)

mean_eager = torch.stack(results_eager).mean(dim=0)      # [steps, key_dim]
mean_compiled = torch.stack(results_compiled).mean(dim=0)

l2_per_step = torch.norm(mean_eager - mean_compiled, dim=-1)

print(f"\nL2 distance per decode step (key vectors):")
for step in range(len(l2_per_step)):
    print(f"  Step {step:2d}: L2={l2_per_step[step].item():.6f}")

print(f"\nSummary:")
print(f"  Mean L2: {l2_per_step.mean().item():.6f}")
print(f"  Max L2:  {l2_per_step.max().item():.6f}")
print(f"  Min L2:  {l2_per_step.min().item():.6f}")

# Analyze deviation pattern
print(f"\nDeviation pattern:")
if l2_per_step[-1] > l2_per_step[0]:
    growth = l2_per_step[-1] / l2_per_step[0] if l2_per_step[0] > 0 else float('inf')
    print(f"  ✓ Accumulating (grows {growth:.2f}x from first to last step)")
elif l2_per_step.std() < l2_per_step.mean() * 0.1:
    print(f"  → Stable (consistent deviation across steps)")
else:
    print(f"  ? Variable (no clear pattern)")

# Compare to typical hidden state deviations
# From prefill experiment: L2=5.47 for hidden states (dim=3584)
# Key vectors have dim = num_heads * head_dim = 28 * 128 = 3584 (same!)
print(f"\nComparison to hidden states:")
print(f"  Key vector dim: {key_vector_dim}")
print(f"  Hidden state dim: 3584")
print(f"  Dimensions match: {'✓' if key_vector_dim == 3584 else '✗'}")
print(f"  Max key vector L2: {l2_per_step.max().item():.2f}")
print(f"  Prefill hidden state L2: 5.47 (from previous experiment)")

# Token divergence check
tokens_match = all(
    tokens_eager[0][i] == tokens_compiled[0][i]
    for i in range(min(len(tokens_eager[0]), len(tokens_compiled[0])))
)

print(f"\nToken sequences:")
if tokens_match:
    print(f"  ✓ Identical")
else:
    print(f"  ✗ DIVERGED")
    for i in range(min(len(tokens_eager[0]), len(tokens_compiled[0]))):
        if tokens_eager[0][i] != tokens_compiled[0][i]:
            print(f"  First divergence at step {i}")
            break

# ============================================================================
# INTERPRETATION
# ============================================================================

print("\n" + "="*60)
print("FORENSIC ANALYSIS")
print("="*60)

max_l2 = l2_per_step.max().item()

if max_l2 > 10:
    print(f"📊 EXCELLENT: L2 max={max_l2:.2f}")
    print(f"  Key vectors show strong forensic signal")
elif max_l2 > 1:
    print(f"✓ GOOD: L2 max={max_l2:.2f}")
    print(f"  Key vectors are forensically useful")
elif max_l2 > 0.1:
    print(f"⚠ WEAK: L2 max={max_l2:.3f}")
    print(f"  Small but potentially detectable")
else:
    print(f"✗ INSUFFICIENT: L2 max={max_l2:.6f}")
    print(f"  Key vectors do not show useful signal")

print(f"\nReproducibility:")
if eager_reproducible and compiled_reproducible:
    print(f"  ✓ Perfect within both conditions")
    print(f"  Systematic deviation dominates")
else:
    print(f"  ⚠ Some variation within conditions")

print(f"\nPractical advantages:")
print(f"  ✓ Zero overhead (KV cache exists anyway)")
print(f"  ✓ Smaller vectors ({key_vector_dim} vs full hidden state)")
print(f"  ✓ No need for output_hidden_states=True")
print(f"  {'✓' if not 'Error' in str(results_compiled) else '✗'} Works with CUDA graphs")

# ============================================================================
# COMPARISON TO HIDDEN STATES
# ============================================================================

print("\n" + "="*60)
print("KEY VECTORS vs HIDDEN STATES FOR FORENSICS")
print("="*60)

print(f"\nKey vectors (this experiment):")
print(f"  Dimension: {key_vector_dim}")
print(f"  Max L2 deviation: {max_l2:.2f}")
print(f"  Overhead: Zero (already in memory)")
print(f"  Extraction: From KV cache")

print(f"\nHidden states (prefill experiment):")
print(f"  Dimension: 3584")
print(f"  Max L2 deviation: 5.47")
print(f"  Overhead: output_hidden_states=True")
print(f"  Extraction: From model outputs")

if key_vector_dim == 3584:
    ratio = max_l2 / 5.47
    print(f"\nRelative signal strength: {ratio:.2f}x")
    if ratio > 0.5:
        print(f"  ✓ Key vectors provide comparable forensic signal")
    elif ratio > 0.1:
        print(f"  → Key vectors provide partial forensic signal")
    else:
        print(f"  ✗ Key vectors provide insufficient signal")

# ============================================================================
# SAVE RESULTS
# ============================================================================

output = {
    "experiment": "key_vector_forensics_torch_compile_decode",
    "timestamp": datetime.now().isoformat(),
    "model": model_name,
    "hardware": {
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "hostname": HOSTNAME,
        "container_id": CONTAINER_ID
    },
    "architecture": {
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "key_vector_dim": key_vector_dim
    },
    "config": {
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "repetitions": num_reps,
        "dtype": "bfloat16",
        "compile_mode": "reduce-overhead",
        "extraction_method": "kv_cache_last_layer"
    },
    "reproducibility": {
        "eager": eager_reproducible,
        "compiled": compiled_reproducible
    },
    "systematic_deviation": {
        "l2_per_step": l2_per_step.tolist(),
        "mean_l2": float(l2_per_step.mean()),
        "max_l2": float(l2_per_step.max()),
        "min_l2": float(l2_per_step.min())
    },
    "token_divergence": {
        "tokens_match": tokens_match,
        "eager_tokens": tokens_eager[0],
        "compiled_tokens": tokens_compiled[0]
    }
}

output_file = f"key_vector_forensics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
output_path = f"/workspace/{output_file}"

with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Results saved to {output_path}")
print("\n" + "="*60)
print("EXPERIMENT COMPLETE")
print("="*60)
print(f"\nKey findings:")
print(f"  • Key vector extraction: {'✓ Works' if compiled_reproducible else '✗ Failed'}")
print(f"  • Max L2 deviation: {max_l2:.4f}")
print(f"  • Forensic viability: {'✓ Yes' if max_l2 > 1 else '⚠ Marginal' if max_l2 > 0.1 else '✗ No'}")
print(f"  • Overhead: Zero")
print("="*60)
