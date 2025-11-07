import os
os.environ['HF_HOME'] = '/workspace/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/workspace/huggingface_cache'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from datetime import datetime
import json
import socket

# Capture system info for verification
HOSTNAME = socket.gethostname()
CONTAINER_ID = os.environ.get('HOSTNAME', 'unknown')

print(f"System Info:")
print(f"  Hostname: {HOSTNAME}")
print(f"  Container: {CONTAINER_ID}")
print()

# Capture relevant environment variables
print("Environment Variables:")
env_vars = {}
for key in sorted(os.environ.keys()):
    if any(x in key.upper() for x in ['CUDA', 'TORCH', 'NCCL', 'CUDNN', 'PYTORCH']):
        env_vars[key] = os.environ[key]
        print(f"  {key}={os.environ[key]}")
if not env_vars:
    print("  (No CUDA/TORCH env vars set)")
print()

def collect_activations_parallel_batch(model, tokenizer, base_prompt, batch_size=1, device="cuda"):
    """Forward pass where element 0 is ALWAYS base_prompt, but batch has dummy elements
    
    CRITICAL: This tests if parallel processing of OTHER sequences affects element 0's activations
    - bs=1: [base_prompt]
    - bs=2: [base_prompt, dummy_prompt]
    - bs=4: [base_prompt, dummy1, dummy2, dummy3]
    
    We ALWAYS extract from element 0, which has identical input across all batch sizes.
    """
    torch.cuda.empty_cache()
    
    if batch_size == 1:
        prompts = [base_prompt]
    else:
        # Element 0 is always base_prompt (unchanged)
        # Elements 1+ are dummy sequences to fill the batch
        dummy_prompts = [f"This is dummy sequence {i} to fill the batch with different content." for i in range(1, batch_size)]
        prompts = [base_prompt] + dummy_prompts
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    
    # DEBUG: Verify setup
    actual_batch_size = inputs['input_ids'].shape[0]
    seq_len = inputs['input_ids'].shape[1]
    if actual_batch_size != batch_size:
        print(f"WARNING: Expected batch_size={batch_size}, got {actual_batch_size}")
    
    if batch_size > 1:
        # Verify element 0 and element 1 are different
        elem0_last5 = inputs['input_ids'][0, -5:].tolist()
        elem1_last5 = inputs['input_ids'][1, -5:].tolist()
        different = elem0_last5 != elem1_last5
        print(f"  Batch verification: seq_len={seq_len}, elem[0]≠elem[1]: {different}")
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    
    # CRITICAL: ALWAYS extract from element 0 (which has identical input across all batch sizes)
    last_layer_last_pos = outputs.hidden_states[-1][0, -1, :].cpu().clone()
    
    del outputs
    del inputs
    torch.cuda.empty_cache()
    
    return last_layer_last_pos

# Setup
CACHE_DIR = '/workspace/huggingface_cache'
EXP_NUMBER = 4  # Corrected parallel batch experiment
model_name = "Qwen/Qwen2.5-7B-Instruct"

print(f"Loading {model_name} in BF16...")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    cache_dir=CACHE_DIR,
    low_cpu_mem_usage=True,
    device_map="auto"
)

# Same long prompt as before
prompt = """The development of large language models has fundamentally transformed natural language processing 
and artificial intelligence more broadly. These models, trained on vast corpora of text data, have demonstrated 
remarkable capabilities across a wide range of tasks, from translation and summarization to question answering 
and creative writing. However, their deployment raises significant challenges related to computational efficiency, 
interpretability, and safety.

One critical challenge in deploying large language models at scale is ensuring computational efficiency. 
Modern language models can contain hundreds of billions of parameters, requiring substantial computational 
resources for both training and inference. Inference optimization techniques have become increasingly important 
as these models are deployed in production environments. Key approaches include quantization, where model 
weights and activations are represented with reduced precision; knowledge distillation, where a smaller 
student model learns to mimic a larger teacher model; and architectural innovations such as mixture-of-experts 
models that activate only relevant subnetworks for each input.

Quantization techniques have evolved significantly in recent years. Early approaches focused on post-training 
quantization, where a trained full-precision model is converted to lower precision. More recently, quantization-aware 
training has gained prominence, allowing models to learn weight distributions that are more amenable to quantization. 
Techniques such as INT8 quantization can reduce model size by 75% while maintaining most of the original model's 
performance. More aggressive quantization schemes, including INT4 and even binary networks, push the boundaries 
of how much precision can be sacrificed while maintaining acceptable performance.

The inference stack itself introduces numerous sources of variation in model outputs. Floating-point arithmetic 
is inherently non-associative, meaning that the order of operations affects the final result. In distributed 
inference scenarios, where computation is parallelized across multiple GPUs, different parallelization strategies 
can lead to different operation orderings and thus different numerical results, even when using identical model 
weights and inputs. Factors such as batch size, communication patterns between GPUs, the specific CUDA kernels 
selected for various operations, and even the GPU architecture itself can all contribute to variations in output.

These variations, while often small, can have important implications for model verification and monitoring in 
high-stakes applications. Consider a scenario where a data center is required to report all inference computations 
performed on its hardware. If the data center operator wishes to perform unauthorized computations, they might 
attempt to hide them by improving the efficiency of their declared computations, freeing up computational capacity 
for undeclared work. Detecting such evasion requires the ability to verify that claimed inference outputs were 
indeed produced by the claimed computational setup.

One approach to such verification is output forensics: carefully analyzing the numerical properties of model 
outputs to detect signatures of the computational setup that produced them. The non-determinism inherent in 
GPU computation might initially seem to preclude such forensics, as outputs from the same model with the same 
inputs might vary across runs. However, a more nuanced understanding distinguishes between systematic deviation, 
which is reproducible when the computational setup is replicated, and statistical noise, which varies unpredictably 
even with identical setups.

Systematic deviation arises from consistent choices in the inference stack: the precision of arithmetic operations, 
the specific algorithms used for matrix multiplications and other kernels, the parallelization strategy, and the 
software versions. When these factors are held constant, outputs should be reproducible. Statistical noise, by 
contrast, arises from inherently non-deterministic aspects of GPU execution, such as the scheduling of operations 
across streaming multiprocessors and the use of atomic operations in reductions.

If statistical noise can be minimized or characterized, systematic deviations become detectable and attributable 
to specific computational setup choices. This opens the possibility of using output forensics to verify claims 
about computational setups. An auditor with access to claimed model weights and inputs could attempt to reproduce 
claimed outputs on a trusted reference system. Systematic deviations between the claimed outputs and reproduced 
outputs could indicate differences in the computational setup, potentially revealing unauthorized optimizations 
or entirely different model executions.

The feasibility of such forensics depends on several factors. First, the magnitude of systematic deviations must 
be large enough to be reliably detectable above any statistical noise. Second, different computational setup 
choices must produce distinguishable deviation patterns. Third, the forensics must be robust to legitimate 
variations in setup that don't affect the computational capacity available for unauthorized work. For example, 
differences in CUDA or PyTorch versions might be unavoidable and should not trigger false alarms, while 
differences in batch size could indicate capacity freed for unauthorized computations.

Empirical investigation of these questions requires careful experimentation with different inference setups. 
A systematic approach would begin by characterizing statistical noise under tightly controlled conditions, 
ensuring that all setup parameters are held constant across multiple inference runs. This establishes a 
baseline for the inherent variability of the system. Subsequently, individual setup parameters can be varied 
one at a time, measuring the systematic deviation introduced by each change. Parameters of particular interest 
include batch size, floating-point precision, parallelization strategy, software versions, and GPU architecture.

The relationship between model scale and the magnitude of both systematic deviation and statistical noise is 
also important to understand. Larger models involve more computation and more opportunities for numerical 
errors to accumulate. They may also be more likely to trigger different computational paths in libraries like 
cuBLAS, which may select different algorithms based on problem size. Understanding how forensic detectability 
scales with model size is crucial for assessing the practical applicability of this verification approach."""

prompt_tokens = len(tokenizer.encode(prompt))
print(f"\nBase prompt token count: {prompt_tokens} tokens")
print("CRITICAL: Element 0 will be IDENTICAL across all batch sizes")
print("Only the presence of parallel dummy sequences will vary\n")

# Test batch sizes 1, 2, 4 with 10 repetitions each
batch_sizes = [1, 2, 4]
num_repetitions = 10
results = {}
all_activations = {}

print(f"{'='*60}")
print(f"Starting H100 PARALLEL BATCH TEST at {datetime.now().isoformat()}")
print(f"Model: {model_name}")
print(f"Precision: BF16 (bfloat16)")
print(f"Base prompt tokens: {prompt_tokens}")
print(f"Operation: Single forward pass (prefill only)")
print(f"CRITICAL: Element 0 input is IDENTICAL across batch sizes")
print(f"Repetitions per batch size: {num_repetitions}")
print(f"{'='*60}\n")

for bs in batch_sizes:
    print(f"Collecting batch_size={bs} ({num_repetitions} repetitions)...")
    if bs == 1:
        print(f"  Batch: [base_prompt]")
    else:
        print(f"  Batch: [base_prompt, dummy1, dummy2, ...] (extracting from elem 0)")
    
    runs = []
    for rep in range(num_repetitions):
        activation = collect_activations_parallel_batch(model, tokenizer, prompt, batch_size=bs, device="cuda")
        runs.append(activation)
        if rep == 0:
            print(f"  Rep 0: norm={torch.norm(activation).item():.6f}, first_val={activation[0].item():.6f}")
        if (rep + 1) % 3 == 0:
            print(f"  Completed {rep + 1}/{num_repetitions} repetitions")
    
    # Check if repetitions are different
    first_rep = runs[0]
    all_identical = all(torch.equal(first_rep, runs[i]) for i in range(1, num_repetitions))
    if all_identical:
        print(f"  ✓ All {num_repetitions} repetitions are identical (expected)")
    else:
        print(f"  ⚠ Repetitions vary (unexpected!)")
    
    results[bs] = torch.stack(runs)
    all_activations[f"batch_size_{bs}"] = results[bs].float().numpy().tolist()
    
    mean_activation = results[bs].mean(dim=0)
    deviations = torch.stack([torch.norm(results[bs][i] - mean_activation) for i in range(num_repetitions)])
    std_noise = deviations.std().item()
    mean_noise = deviations.mean().item()
    
    print(f"  Statistical noise: mean={mean_noise:.6f}, std={std_noise:.6f}")
    print(f"  Activation norm: {torch.norm(mean_activation).item():.2f}\n")
    
    torch.cuda.empty_cache()

# Compare systematic deviations between batch sizes
print("\n" + "="*60)
print("=== SYSTEMATIC DEVIATION MATRIX (position n-1) ===")
print("="*60)
print("     ", end="")
for bs in batch_sizes:
    print(f"bs={bs:2d}  ", end="")
print()

systematic_deviations = {}
for bs1 in batch_sizes:
    print(f"bs={bs1:2d} ", end="")
    for bs2 in batch_sizes:
        if bs1 == bs2:
            print("  -    ", end="")
        else:
            mean1 = results[bs1].mean(dim=0)
            mean2 = results[bs2].mean(dim=0)
            l2 = torch.norm(mean1 - mean2).item()
            systematic_deviations[f"bs{bs1}_vs_bs{bs2}"] = l2
            print(f"{l2:6.3f} ", end="")
    print()

print("\n" + "="*60)
print("=== ACTIVATION SCALE ANALYSIS ===")
print("="*60)

bs1_mean = results[1].mean(dim=0)
bs2_mean = results[2].mean(dim=0)

print(f"Activation vector dimension: {bs1_mean.shape[0]}")
print(f"bs=1 mean activation norm: {torch.norm(bs1_mean).item():.2f}")
print(f"bs=2 mean activation norm: {torch.norm(bs2_mean).item():.2f}")
print(f"L2 distance (bs1 vs bs2): {torch.norm(bs1_mean - bs2_mean).item():.4f}")
if torch.norm(bs1_mean) > 0:
    print(f"Relative difference: {(torch.norm(bs1_mean - bs2_mean) / torch.norm(bs1_mean)).item():.6f}")

diff = (bs1_mean - bs2_mean).abs()
print(f"Max absolute difference: {diff.max().item():.6f}")
print(f"Dimensions with |diff| > 0.01: {(diff > 0.01).sum().item()}/{diff.shape[0]}")

# CRITICAL CHECK: Does parallel batch processing affect element 0?
bs1_vs_bs2_deviation = systematic_deviations.get("bs1_vs_bs2", 0)
print("\n" + "="*60)
print("=== VERDICT ===")
print("="*60)
print(f"Element 0 input was IDENTICAL across all batch sizes")
print(f"bs1 vs bs2 deviation = {bs1_vs_bs2_deviation:.6f}\n")

if bs1_vs_bs2_deviation > 0.1:
    print(f"✓ DETECTION POSSIBLE: Parallel batch processing affects activations")
    print(f"  Even though element 0 had identical input, having parallel dummy")
    print(f"  sequences in the batch changed its activations by L2={bs1_vs_bs2_deviation:.4f}")
    print(f"  → Forensic verification CAN detect hidden batch capacity on H100")
elif bs1_vs_bs2_deviation > 0.001:
    print(f"⚠ WEAK SIGNAL: Small but detectable deviation")
    print(f"  Parallel batch processing has minimal effect (L2={bs1_vs_bs2_deviation:.6f})")
    print(f"  → Forensics might work with careful analysis")
else:
    print(f"✗ NO DETECTION: H100 batch processing is perfectly isolated")
    print(f"  Element 0's activations are unchanged by parallel batch elements")
    print(f"  → Cannot detect hidden batch capacity on H100 using this method")

# Save results
output = {
    "experiment": "H100_parallel_batch_forensics_test",
    "timestamp": datetime.now().isoformat(),
    "model": model_name,
    "hardware": {
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "hostname": HOSTNAME,
        "container_id": CONTAINER_ID
    },
    "environment": env_vars,
    "config": {
        "batch_sizes": batch_sizes,
        "repetitions": num_repetitions,
        "operation": "single_forward_pass_prefill_only",
        "dtype": "bfloat16",
        "use_cache": False,
        "input_strategy": "element_0_identical_across_batch_sizes",
        "dummy_sequences": "different_text_in_elements_1plus",
        "base_prompt_tokens": prompt_tokens,
        "hidden_dim": int(bs1_mean.shape[0])
    },
    "statistical_noise": {
        f"batch_size_{bs}": {
            "mean": float(torch.stack([torch.norm(results[bs][i] - results[bs].mean(dim=0)) 
                                       for i in range(num_repetitions)]).mean()),
            "std": float(torch.stack([torch.norm(results[bs][i] - results[bs].mean(dim=0)) 
                                     for i in range(num_repetitions)]).std())
        }
        for bs in batch_sizes
    },
    "systematic_deviations": systematic_deviations,
    "activation_norms": {
        f"batch_size_{bs}": float(torch.norm(results[bs].mean(dim=0))) 
        for bs in batch_sizes
    },
    "forensics_test_result": {
        "bs1_vs_bs2_deviation": bs1_vs_bs2_deviation,
        "detection_viable": bs1_vs_bs2_deviation > 0.1,
        "interpretation": "Can forensics detect hidden parallel batch capacity?"
    },
    "raw_activations": all_activations
}

output_file = f"{torch.cuda.get_device_name(0).replace(' ', '_')}_parallel_batch_exp{EXP_NUMBER}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
output_path = f"/workspace/{output_file}"

with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Results saved to {output_path}")
print(f"✓ File size: ~{len(json.dumps(output)) / 1024:.1f} KB")
print("\n" + "="*60)
print("PARALLEL BATCH TEST COMPLETE")
print("="*60)