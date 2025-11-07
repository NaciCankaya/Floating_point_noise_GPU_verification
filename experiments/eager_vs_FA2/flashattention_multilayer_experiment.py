import os
os.environ['HF_HOME'] = '/workspace/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/workspace/huggingface_cache'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc
import numpy as np
from datetime import datetime
import json
import socket
import sys

# ============================================================================
# STEP 1: CHECK FLASHATTENTION AVAILABILITY
# ============================================================================

print("="*60)
print("CHECKING FLASHATTENTION AVAILABILITY")
print("="*60)

try:
    import flash_attn
    FLASH_ATTN_VERSION = flash_attn.__version__ if hasattr(flash_attn, '__version__') else "unknown"
    print(f"✓ FlashAttention installed: version {FLASH_ATTN_VERSION}")
    
    has_fa2 = hasattr(flash_attn, 'flash_attn_func')
    print(f"  FlashAttention 2 API: {'✓ Available' if has_fa2 else '✗ Not found'}")
    
    if not has_fa2:
        print("\n✗ ERROR: FlashAttention 2 API not found")
        sys.exit(1)
    
    print("\n✓ FlashAttention available - proceeding with test\n")
    
except ImportError as e:
    print("✗ FlashAttention NOT installed")
    print(f"   Import error: {e}")
    print("\nInstall with: pip install flash-attn --no-build-isolation")
    sys.exit(1)

# ============================================================================
# STEP 2: SYSTEM INFO AND MEMORY CLEANUP
# ============================================================================

print("="*60)
print("CLEANING GPU MEMORY")
print("="*60)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated(0) / 1024**3
    print(f"GPU memory allocated: {mem_before:.2f} GB")
    if mem_before > 10:
        print(f"⚠ WARNING: {mem_before:.1f} GB already allocated - consider restarting Python")
print()

HOSTNAME = socket.gethostname()
CONTAINER_ID = os.environ.get('HOSTNAME', 'unknown')

print(f"System Info:")
print(f"  Hostname: {HOSTNAME}")
print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.version.cuda}")
print()

# ============================================================================
# STEP 3: HELPER FUNCTIONS
# ============================================================================

def verify_attention_implementation(model):
    """Check what attention implementation the model is using via config"""
    config_impl = getattr(model.config, '_attn_implementation', 'not set')
    print(f"  Model config._attn_implementation: {config_impl}")
    print(f"  Model config.attn_implementation: {getattr(model.config, 'attn_implementation', 'not set')}")
    
    # Check first attention layer class (just for info)
    first_layer = model.model.layers[0]
    attn_class_name = first_layer.self_attn.__class__.__name__
    print(f"  Actual attention class: {attn_class_name}")
    
    # In transformers 4.57.1+, there's only one Qwen2Attention class
    # that dispatches internally based on config._attn_implementation
    is_using_flash = (config_impl == "flash_attention_2")
    print(f"  Using FlashAttention: {is_using_flash}")
    
    return attn_class_name, is_using_flash

def load_model_with_attention(attn_impl, cache_dir):
    """Load model with specified attention implementation"""
    print(f"\nLoading model with {attn_impl} attention...")
    
    # Clear GPU before loading
    gc.collect()
    torch.cuda.empty_cache()
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation=attn_impl
    )
    
    # Verify what we got
    attn_class, is_using_flash = verify_attention_implementation(model)
    
    mem_after = torch.cuda.memory_allocated(0) / 1024**3
    print(f"  GPU memory after load: {mem_after:.2f} GB")
    
    return model, attn_class, is_using_flash

def collect_activations_multilayer(model, tokenizer, prompt, device="cuda"):
    """Extract hidden states from multiple layers to check error propagation"""
    # Ensure all prior ops are complete and GPU is clean
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    
    inputs = tokenizer([prompt], return_tensors="pt", padding=True)
    seq_len = inputs['input_ids'].shape[1]
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    
    last_valid_pos = inputs['attention_mask'][0].sum() - 1
    
    # Extract from multiple layers
    # outputs.hidden_states[0] is embedding layer
    # outputs.hidden_states[1] is after first transformer layer
    # outputs.hidden_states[-1] is after last transformer layer
    num_layers = len(outputs.hidden_states) - 1  # -1 because first is embedding
    
    # Sample: layers across the network for fine-grained divergence tracking
    # Dense sampling in early layers (1-4) to catch initial divergence
    layer_indices = [1, 2, 3, 4, 7, 10, 14, 18, 22, num_layers]
    
    # Extract immediately and move to CPU to minimize GPU memory usage
    activations = {}
    for idx in layer_indices:
        # Clone and move to CPU immediately, then delete reference
        layer_activation = outputs.hidden_states[idx][0, last_valid_pos, :].cpu().clone()
        activations[f"layer_{idx}"] = layer_activation
    
    # Aggressive cleanup - delete everything from GPU immediately
    del outputs.hidden_states  # Delete the tuple of hidden states first
    del outputs
    del inputs
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    return activations, seq_len, layer_indices

def unload_model(model):
    """Completely remove model from memory"""
    print(f"  Unloading model...")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated(0) / 1024**3
    print(f"  GPU memory after unload: {mem_after:.2f} GB")

# ============================================================================
# STEP 4: SETUP
# ============================================================================

CACHE_DIR = '/workspace/huggingface_cache'
model_name = "Qwen/Qwen2.5-7B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)

# Load prompt from file
prompt_file = "dummytext.txt"
try:
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt = f.read().strip()
    print(f"✓ Loaded prompt from {prompt_file}")
except FileNotFoundError:
    print(f"⚠ Warning: {prompt_file} not found, using default prompt")
    prompt = """The automated data-processing pipeline ingests raw telemetry from distributed sensors across multiple geographic locations. A proprietary algorithm then normalizes the dataset, filtering for anomalies based on predefined statistical parameters derived from historical patterns. The resulting output is a clean, structured matrix ready for machine learning model ingestion."""

prompt_tokens = len(tokenizer.encode(prompt))
print(f"Test prompt: {prompt_tokens} tokens")
print()

# Create output directory
os.makedirs('/workspace/experiments', exist_ok=True)

# ============================================================================
# STEP 5: RUN EXPERIMENTS
# ============================================================================

num_repetitions = 5
results = {}  # Will store {condition: {layer_name: tensor}}
all_activations = {}
attention_info = {}
layer_indices = None

conditions = [
    ("eager", "eager"),
    ("flash_attention_2", "flash_attention_2")
]

print(f"{'='*60}")
print(f"FLASHATTENTION MULTI-LAYER FORENSICS")
print(f"Model: {model_name}")
print(f"Precision: BF16 (bfloat16)")
print(f"Prompt tokens: {prompt_tokens}")
print(f"Strategy: Extract activations from 8 layers [1,4,7,10,14,18,22,28]")
print(f"Purpose: Fine-grained error propagation tracking")
print(f"Repetitions per condition: {num_repetitions}")
print(f"{'='*60}\n")

for condition_name, attn_impl in conditions:
    print(f"="*60)
    print(f"TESTING: {condition_name}")
    print(f"="*60)
    
    # Load model fresh for this condition
    model, attn_class, is_using_flash = load_model_with_attention(attn_impl, CACHE_DIR)
    
    mem_after_load = torch.cuda.memory_allocated(0) / 1024**3
    print(f"  GPU memory after model load: {mem_after_load:.2f} GB")
    
    attention_info[condition_name] = {
        "requested": attn_impl,
        "actual_class": attn_class,
        "is_using_flash": is_using_flash,
        "config_attn_implementation": model.config._attn_implementation
    }
    
    # Collect activations
    print(f"\nCollecting multi-layer activations ({num_repetitions} repetitions)...")
    
    # Initialize storage for this condition
    results[condition_name] = {}
    
    for rep in range(num_repetitions):
        mem_before = torch.cuda.memory_allocated(0) / 1024**3
        
        activations, seq_len, layer_idx_list = collect_activations_multilayer(
            model, tokenizer, prompt, device="cuda"
        )
        
        mem_after = torch.cuda.memory_allocated(0) / 1024**3
        
        # First time through, set up layer indices and storage
        if layer_indices is None:
            layer_indices = layer_idx_list
            print(f"  Extracting from layers: {layer_indices}")
        
        # Initialize storage on first rep of each condition
        if rep == 0:
            for layer_name in activations.keys():
                results[condition_name][layer_name] = []
        
        # Store activations for each layer
        for layer_name, activation in activations.items():
            results[condition_name][layer_name].append(activation)
        
        if rep == 0:
            print(f"  Rep 0 norms: {', '.join([f'{k}={torch.norm(v).item():.2f}' for k, v in activations.items()])}")
            print(f"  Rep 0 memory: before={mem_before:.2f}GB, after={mem_after:.2f}GB, delta={mem_after-mem_before:.3f}GB")
        if (rep + 1) % 3 == 0:
            mem_current = torch.cuda.memory_allocated(0) / 1024**3
            print(f"  Completed {rep + 1}/{num_repetitions} repetitions (GPU: {mem_current:.2f} GB)")
        
        # Aggressive memory cleanup after each repetition
        del activations
        gc.collect()
        torch.cuda.empty_cache()
    
    # Stack repetitions into tensors
    for layer_name in results[condition_name].keys():
        results[condition_name][layer_name] = torch.stack(results[condition_name][layer_name])
    
    # Aggressive cleanup after stacking
    gc.collect()
    torch.cuda.empty_cache()
    
    # Check repeatability for last layer
    last_layer_name = f"layer_{layer_indices[-1]}"
    first_rep = results[condition_name][last_layer_name][0]
    all_identical = all(
        torch.equal(first_rep, results[condition_name][last_layer_name][i]) 
        for i in range(1, num_repetitions)
    )
    print(f"  Repeatability (last layer): {'✓ All identical' if all_identical else '⚠ Varies'}")
    
    # Convert to numpy/lists for JSON BEFORE next condition
    # This frees the GPU tensors
    condition_activations = {}
    for layer_name, tensor in results[condition_name].items():
        condition_activations[layer_name] = tensor.float().cpu().numpy().tolist()
        # Delete the GPU tensor immediately after conversion
        del tensor
    
    all_activations[condition_name] = condition_activations
    
    # Clear the GPU tensors from results to save memory
    # (we'll need them for analysis so keep only CPU copies)
    results_cpu = {
        layer_name: tensor.cpu().float() 
        for layer_name, tensor in results[condition_name].items()
    }
    results[condition_name] = results_cpu
    
    # Final cleanup before unloading model
    gc.collect()
    torch.cuda.empty_cache()
    
    print()
    unload_model(model)
    print()

# ============================================================================
# STEP 6: LAYER-BY-LAYER ANALYSIS
# ============================================================================

print("="*60)
print("=== ATTENTION IMPLEMENTATION VERIFICATION ===")
print("="*60)
for condition_name, info in attention_info.items():
    print(f"\n{condition_name}:")
    print(f"  Requested: {info['requested']}")
    print(f"  Config reports: {info['config_attn_implementation']}")
    print(f"  Actual class: {info['actual_class']}")
    print(f"  Using FlashAttention: {info['is_using_flash']}")

print("\n" + "="*60)
print("=== LAYER-BY-LAYER DEVIATION ANALYSIS ===")
print("="*60)

layer_deviations = {}
layer_names = [f"layer_{idx}" for idx in layer_indices]

print(f"\nComparing: eager vs flash_attention_2")
print(f"Layers analyzed: {layer_indices}\n")

for layer_name, layer_idx in zip(layer_names, layer_indices):
    eager_mean = results["eager"][layer_name].mean(dim=0)
    flash_mean = results["flash_attention_2"][layer_name].mean(dim=0)
    
    l2_distance = torch.norm(eager_mean - flash_mean).item()
    relative_diff = (l2_distance / torch.norm(eager_mean)).item() if torch.norm(eager_mean) > 0 else 0
    
    diff = (eager_mean - flash_mean).abs()
    max_diff = diff.max().item()
    dims_affected = (diff > 0.01).sum().item()
    dims_total = diff.shape[0]
    
    layer_deviations[layer_name] = {
        "layer_index": layer_idx,
        "l2_distance": l2_distance,
        "relative_difference": relative_diff,
        "max_absolute_diff": max_diff,
        "dims_affected": dims_affected,
        "dims_total": dims_total,
        "eager_norm": float(torch.norm(eager_mean)),
        "flash_norm": float(torch.norm(flash_mean))
    }
    
    print(f"Layer {layer_idx}:")
    print(f"  L2 distance: {l2_distance:.6f}")
    print(f"  Relative diff: {relative_diff:.6f} ({relative_diff*100:.3f}%)")
    print(f"  Max |diff|: {max_diff:.6f}")
    print(f"  Dims with |diff| > 0.01: {dims_affected}/{dims_total}")
    print(f"  Eager norm: {torch.norm(eager_mean).item():.2f}")
    print(f"  Flash norm: {torch.norm(flash_mean).item():.2f}")
    print()

# ============================================================================
# STEP 7: ERROR PROPAGATION ANALYSIS
# ============================================================================

print("="*60)
print("=== ERROR PROPAGATION DIAGNOSIS ===")
print("="*60)

# Check if both actually used different implementations
eager_config = attention_info["eager"]["config_attn_implementation"]
flash_config = attention_info["flash_attention_2"]["config_attn_implementation"]

if eager_config == "eager" and flash_config == "flash_attention_2":
    print(f"✓ Confirmed: Comparing eager vs FlashAttention 2\n")
    
    # Analyze progression
    l2_distances = [layer_deviations[f"layer_{idx}"]["l2_distance"] for idx in layer_indices]
    relative_diffs = [layer_deviations[f"layer_{idx}"]["relative_difference"] for idx in layer_indices]
    
    print(f"L2 distance progression:")
    for idx, l2 in zip(layer_indices, l2_distances):
        print(f"  Layer {idx}: {l2:.6f}")
    
    print(f"\nRelative difference progression:")
    for idx, rel in zip(layer_indices, relative_diffs):
        print(f"  Layer {idx}: {rel:.6f} ({rel*100:.3f}%)")
    
    # Check if error is growing
    is_growing = all(l2_distances[i] <= l2_distances[i+1] for i in range(len(l2_distances)-1))
    growth_rate = l2_distances[-1] / l2_distances[0] if l2_distances[0] > 0 else float('inf')
    
    print(f"\n{'='*60}")
    print(f"DIAGNOSIS:")
    print(f"{'='*60}")
    
    if is_growing:
        print(f"✓ ERROR PROPAGATION CONFIRMED")
        print(f"  Deviation grows across layers: {is_growing}")
        print(f"  Growth factor (first→last): {growth_rate:.2f}x")
        print(f"  First layer L2: {l2_distances[0]:.6f}")
        print(f"  Last layer L2: {l2_distances[-1]:.6f}")
        
        if l2_distances[0] < 1.0:
            print(f"\n✓ LEGITIMATE: Small initial deviation suggests genuine")
            print(f"  algorithmic difference accumulating through layers")
        else:
            print(f"\n⚠ SUSPICIOUS: Large deviation even in first layer")
            print(f"  May indicate models differ in more than just attention")
    else:
        print(f"⚠ UNEXPECTED PATTERN")
        print(f"  Deviation does NOT grow monotonically")
        print(f"  This is unusual for error propagation")
        print(f"  Possible issues:")
        print(f"    - Models not actually using different implementations")
        print(f"    - Normalization layers resetting accumulated error")
        print(f"    - Other systematic differences beyond attention")
    
    # Final assessment
    print(f"\n{'='*60}")
    print(f"VERIFICATION VIABILITY:")
    print(f"{'='*60}")
    
    last_l2 = l2_distances[-1]
    if last_l2 > 50:
        print(f"📊 EXCELLENT SIGNAL: L2={last_l2:.1f}")
        print(f"  This deviation is easily detectable for forensics")
    elif last_l2 > 10:
        print(f"✓ STRONG SIGNAL: L2={last_l2:.1f}")
        print(f"  Clearly detectable for forensics applications")
    elif last_l2 > 1:
        print(f"✓ DETECTABLE: L2={last_l2:.1f}")
        print(f"  Forensics feasible with careful measurement")
    elif last_l2 > 0.1:
        print(f"⚠ WEAK SIGNAL: L2={last_l2:.3f}")
        print(f"  May require multiple samples")
    else:
        print(f"✗ POOR SIGNAL: L2={last_l2:.3f}")
        print(f"  Below practical detection threshold")
    
else:
    print(f"⚠ WARNING: Config mismatch detected")
    print(f"  Eager config: {eager_config}")
    print(f"  Flash config: {flash_config}")
    print(f"  Results may not reflect true implementation differences")

# ============================================================================
# STEP 8: SAVE RESULTS
# ============================================================================

output = {
    "experiment": "flashattention_multilayer_forensics",
    "timestamp": datetime.now().isoformat(),
    "model": model_name,
    "hardware": {
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "hostname": HOSTNAME,
        "container_id": CONTAINER_ID
    },
    "flash_attention": {
        "version": FLASH_ATTN_VERSION
    },
    "attention_implementations": attention_info,
    "config": {
        "strategy": "reload_model_extract_multiple_layers",
        "layers_extracted": layer_indices,
        "conditions_tested": [c[0] for c in conditions],
        "repetitions": num_repetitions,
        "prompt_tokens": prompt_tokens,
        "dtype": "bfloat16",
        "hidden_dim": dims_total
    },
    "layer_by_layer_deviations": layer_deviations,
    "error_propagation_analysis": {
        "is_growing": is_growing,
        "growth_rate": growth_rate,
        "l2_progression": l2_distances,
        "relative_diff_progression": relative_diffs
    },
    "raw_activations": all_activations
}

gpu_name = torch.cuda.get_device_name(0).replace(' ', '_')
output_file = f"{gpu_name}_flashattn_multilayer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
output_path = f"/workspace/experiments/{output_file}"

with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Results saved to {output_path}")
print(f"✓ File size: ~{len(json.dumps(output)) / 1024:.1f} KB")
print("\n" + "="*60)
print("EXPERIMENT COMPLETE")
print("="*60)
