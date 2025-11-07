#!/usr/bin/env python3
"""
Diagnostic test: Check if baseline activations are truly reproducible

This script tests whether:
1. Baseline activations are bit-exact across multiple extractions
2. Concurrent condition activations are deterministic
3. The experiment methodology is sound

Expected results if methodology is correct:
- Baseline should be bit-exact across runs (no concurrent work)
- L2 distances should VARY across runs if concurrent work is non-deterministic
- Identical L2 distances indicate either caching or deterministic races
"""

import os
os.environ['HF_HOME'] = '/workspace/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/huggingface_cache'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import json

print("="*70)
print("BASELINE REPRODUCIBILITY DIAGNOSTIC")
print("="*70)

model_name = "Qwen/Qwen2.5-7B-Instruct"
device = "cuda"
layer_indices = [1, 4, 10, 18, 28]

# Same prompt as original experiment
prompt = """You are a senior research scientist at a leading AI safety institution. Your task is to write a comprehensive technical report analyzing the current state of AI alignment research, potential risks from advanced AI systems, and proposed mitigation strategies.

The report should cover the following areas in depth:

1. Introduction to AI Alignment
   - Historical context and evolution of the field
   - Key terminology and conceptual frameworks
   - Relationship to broader AI safety and governance efforts
   - Current stakeholders and institutional landscape

2. Technical Challenges in AI Alignment
   - The outer alignment problem: specifying correct objectives
   - The inner alignment problem: ensuring mesa-optimizers are aligned
   - Robustness and distributional shift
   - Scalable oversight and interpretability
   - Deceptive alignment and treacherous turns
   - Value learning and inverse reinforcement learning
   - Corrigibility and shutdown problems

3. Current Research Approaches
   - Reinforcement learning from human feedback (RLHF)
   - Constitutional AI and other oversight methods
   - Debate and amplification techniques
   - Interpretability research and mechanistic understanding
   - Formal verification approaches
   - Multi-agent systems and cooperation
   - Impact measures and side-effect minimization"""

print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager"
)
model.eval()
print("Model loaded\n")

def extract_activations_simple(model, tokenizer, prompt, layer_indices):
    """Extract activations without any monitoring or complexity"""

    # Clear GPU
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Tokenize
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Storage
    activations = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            activations[f"layer_{layer_idx}"] = hidden[:, -1, :].detach().cpu().float().numpy().flatten()
        return hook

    # Register hooks
    hooks = []
    for idx in layer_indices:
        if idx == 0:
            layer = model.model.embed_tokens
        else:
            layer = model.model.layers[idx - 1]
        hooks.append(layer.register_forward_hook(make_hook(idx)))

    # Run inference
    with torch.no_grad():
        _ = model(**inputs)

    # Clean up
    for hook in hooks:
        hook.remove()

    torch.cuda.synchronize()

    return activations

# TEST 1: Baseline reproducibility
print("="*70)
print("TEST 1: Baseline Reproducibility (5 independent extractions)")
print("="*70)

baseline_runs = []
for i in range(5):
    print(f"\nBaseline extraction {i+1}/5...")
    acts = extract_activations_simple(model, tokenizer, prompt, layer_indices)
    baseline_runs.append(acts)
    print(f"  Extracted {len(acts)} layers")

# Compare all baselines
print("\nComparing baseline runs:")
all_baseline_exact = True
for i in range(1, 5):
    print(f"\nBaseline run {i+1} vs run 1:")
    for layer in baseline_runs[0].keys():
        arr1 = baseline_runs[0][layer]
        arr2 = baseline_runs[i][layer]
        bit_exact = np.array_equal(arr1, arr2)

        if bit_exact:
            print(f"  {layer}: ✓ BIT-EXACT")
        else:
            all_baseline_exact = False
            l2 = float(np.linalg.norm(arr1 - arr2))
            print(f"  {layer}: ✗ DIFFERS (L2={l2:.6e})")

if all_baseline_exact:
    print("\n✓ BASELINE IS BIT-EXACT REPRODUCIBLE")
    print("  This is expected for deterministic solo inference")
else:
    print("\n⚠ BASELINE IS NOT REPRODUCIBLE")
    print("  This suggests non-determinism even without concurrent work")
    print("  Could indicate: background processes, CUDA non-determinism, or model issues")

# TEST 2: Check if concurrent work produces deterministic deviations
print("\n" + "="*70)
print("TEST 2: Concurrent Workload Determinism (3 independent runs)")
print("="*70)

# Use a simple concurrent workload
class SimpleConcurrent:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.running = False
        self.thread = None

        # Simple prompt
        self.inputs = tokenizer(["The capital of France is"], return_tensors="pt")
        self.inputs = {k: v.to(device) for k, v in self.inputs.items()}

    def _worker(self):
        import threading
        import time
        while self.running:
            with torch.no_grad():
                _ = self.model(**self.inputs)
            time.sleep(0.001)

    def start(self):
        import threading
        import time
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        time.sleep(0.5)

    def stop(self):
        import time
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        torch.cuda.synchronize()
        time.sleep(0.5)

concurrent_runs = []
baseline_ref = baseline_runs[0]  # Use first baseline as reference

for i in range(3):
    print(f"\nConcurrent run {i+1}/3...")
    concurrent = SimpleConcurrent(model, tokenizer)
    concurrent.start()
    print("  Concurrent inference started")

    acts = extract_activations_simple(model, tokenizer, prompt, layer_indices)
    concurrent_runs.append(acts)

    concurrent.stop()
    print("  Concurrent inference stopped")

# Compute L2 distances for each concurrent run
print("\nL2 distances (baseline vs concurrent):")
l2_matrix = {layer: [] for layer in layer_indices}

for i, concurrent_acts in enumerate(concurrent_runs):
    print(f"\nRun {i+1}:")
    for layer_name in baseline_ref.keys():
        base = baseline_ref[layer_name]
        conc = concurrent_acts[layer_name]

        bit_exact = np.array_equal(base, conc)
        if bit_exact:
            l2 = 0.0
            print(f"  {layer_name}: BIT-EXACT (L2=0.0)")
        else:
            l2 = float(np.linalg.norm(base - conc))
            print(f"  {layer_name}: L2={l2:.10f}")

        layer_num = int(layer_name.split('_')[1])
        l2_matrix[layer_num].append(l2)

# Check if L2 distances are identical across runs
print("\n" + "="*70)
print("ANALYSIS: Are L2 distances identical across runs?")
print("="*70)

for layer_num in layer_indices:
    l2_vals = l2_matrix[layer_num]
    unique_vals = set(l2_vals)

    print(f"\nlayer_{layer_num}:")
    print(f"  Values: {[f'{v:.6f}' for v in l2_vals]}")

    if len(unique_vals) == 1:
        print(f"  ⚠ IDENTICAL across all runs!")
        print(f"  This suggests: deterministic concurrent effects OR caching")
    else:
        print(f"  ✓ VARIES across runs")
        print(f"  This is expected for non-deterministic concurrent workloads")

# TEST 3: Check if the original experiment's baseline was clean
print("\n" + "="*70)
print("TEST 3: Compare to original experiment baselines")
print("="*70)

original_files = [
    'realistic_parallel_forensics_20251106_153028.json',
    'realistic_parallel_forensics_20251106_153243.json',
    'realistic_parallel_forensics_20251106_153309.json'
]

print("\nOriginal baseline GPU utilization (p95):")
for fname in original_files:
    try:
        with open(fname, 'r') as f:
            data = json.load(f)
            gpu_p95 = data['baseline']['gpu_utilization']['p95']
            print(f"  {fname.split('_')[-1].replace('.json', '')}: {gpu_p95:.1f}%")
    except FileNotFoundError:
        print(f"  {fname}: NOT FOUND")

print("\nExpected baseline GPU p95: ~85% (from notebook's successful run)")
print("If original baselines show >90%, they likely had background contamination")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
This diagnostic test checks:
1. Whether baseline is truly reproducible (should be bit-exact)
2. Whether concurrent workload effects are deterministic (should vary!)
3. Whether original baselines were contaminated by background processes

If L2 distances are identical across concurrent runs, the race conditions
are deterministic - which contradicts the assumption of statistical noise.

If original baselines had high GPU utilization, they were contaminated.
""")
