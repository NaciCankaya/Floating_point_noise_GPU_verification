#!/usr/bin/env python3
"""
Experiment 0: Reference Baseline

Establishes baseline measurement with default configuration:
- Model: Qwen/Qwen3-30B-A3B (unquantized)
- Batch size: 1
- Compile: False
- Quantization: None
- Attention: flash_attention_2
- Concurrent work: False
- CUDA: default (12.8)
- TP size: 1
- EP size: 1

Runs 3 repetitions to verify bit-exact reproducibility.
"""

import os
import sys
import torch
import socket
from pathlib import Path
from datetime import datetime

# Add common utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import (
    load_model,
    get_model_info,
    load_prompt_from_pdf,
    run_multiple_repetitions,
    ExperimentWriter,
    DEFAULT_PDF,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Experiment parameters
EXPERIMENT_TYPE = "reference"
VARIABLE_TESTED = "none"
NUM_REPETITIONS = 3
MAX_NEW_TOKENS = 30

# Model parameters
MODEL_NAME = "Qwen/Qwen3-30B-A3B"  # Unquantized 30B MoE model
QUANTIZATION = "none"
ATTN_IMPLEMENTATION = "flash_attention_2"
TORCH_COMPILE = False

# Extraction parameters
LAYER_INDICES = [1, 4, 39]  # First, fourth, and last layer (reduced for smaller file size)
POSITIONS = [-3, -2, -1]
TOP_K_LOGPROBS = 10

# Paths
CACHE_DIR = os.environ.get('HF_HOME', '/workspace/huggingface_cache')
PDF_PATH = str(DEFAULT_PDF)
OUTPUT_FILE = "reference_baseline.json"

# ============================================================================
# SYSTEM INFO
# ============================================================================

def collect_system_info():
    """Collect system and environment information."""
    import transformers
    import platform

    info = {
        "hostname": socket.gethostname(),
        "container_id": os.environ.get('HOSTNAME', 'unknown'),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "transformers_version": transformers.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    # Try to get flash attention version
    try:
        import flash_attn
        info["flash_attn_version"] = flash_attn.__version__
    except ImportError:
        info["flash_attn_version"] = "N/A"

    return info

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("EXPERIMENT 0: REFERENCE BASELINE")
    print("="*80)
    print(f"Start time: {datetime.now().isoformat()}")
    print()

    # Collect system info
    print("System Information:")
    system_info = collect_system_info()
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    print()

    # Print configuration
    print("Experiment Configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Quantization: {QUANTIZATION}")
    print(f"  Attention: {ATTN_IMPLEMENTATION}")
    print(f"  Compile: {TORCH_COMPILE}")
    print(f"  Repetitions: {NUM_REPETITIONS}")
    print(f"  Decode steps: {MAX_NEW_TOKENS}")
    print(f"  PDF: {PDF_PATH}")
    print()

    # Load model
    print("="*80)
    print("LOADING MODEL")
    print("="*80)
    model, tokenizer = load_model(
        model_name=MODEL_NAME,
        quantization=QUANTIZATION,
        attn_implementation=ATTN_IMPLEMENTATION,
        torch_compile=TORCH_COMPILE,
        device_map="auto",
        cache_dir=CACHE_DIR,
    )

    # Get detailed model info
    model_info = get_model_info(model, tokenizer)
    print("\nModel Details:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    print()

    # Load prompt
    print("="*80)
    print("LOADING PROMPT")
    print("="*80)
    print(f"Loading text from: {PDF_PATH}")

    prompt, token_count = load_prompt_from_pdf(
        pdf_path=PDF_PATH,
        tokenizer=tokenizer,
        target_tokens=6000,
    )

    print(f"  Prompt tokens: {token_count}")
    print(f"  Prompt preview: {prompt[:200]}...")
    print()

    # Run inference with repetitions
    print("="*80)
    print("RUNNING INFERENCE")
    print("="*80)

    runs, is_reproducible, error_message = run_multiple_repetitions(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        num_reps=NUM_REPETITIONS,
        max_new_tokens=MAX_NEW_TOKENS,
        layer_indices=LAYER_INDICES,
        positions=POSITIONS,
        top_k_logprobs=TOP_K_LOGPROBS,
        use_cache=False,  # Disable cache for determinism
        verify_reproducibility=True,
    )

    # Create experiment writer
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    writer = ExperimentWriter(
        experiment_type=EXPERIMENT_TYPE,
        variable_tested=VARIABLE_TESTED,
        model=MODEL_NAME.split('/')[-1],  # Just model name, not full path
        layer_indices=LAYER_INDICES,
        positions=POSITIONS,
        decode_steps=MAX_NEW_TOKENS,
    )

    # Determine hardware type from GPU name
    gpu_name = system_info["gpu_name"]
    if "A100" in gpu_name:
        hardware = "A100-80GB"
    elif "H100" in gpu_name:
        hardware = "H100"
    else:
        hardware = gpu_name

    # Determine provider (check common environment variables)
    provider = "Unknown"
    if "RUNPOD" in os.environ.get('HOSTNAME', '').upper():
        provider = "RunPod"
    elif "VAST" in os.environ.get('HOSTNAME', '').upper():
        provider = "vast.ai"

    # Add configuration
    config_id = f"{hardware}_reference"
    writer.add_configuration(
        config_id=config_id,
        hardware=hardware,
        provider=provider,
        variable_value=1,  # Reference has value 1 (baseline)
        cuda_version=system_info["cuda_version"],
        torch_version=system_info["torch_version"],
        transformers_version=system_info["transformers_version"],
        flash_attn_version=system_info["flash_attn_version"],
        python_version=system_info["python_version"],
        fixed_params={
            "batch_size": 1,
            "compile": False,
            "attention_impl": ATTN_IMPLEMENTATION,
            "quantization": QUANTIZATION,
            "tp_size": 1,
            "ep_size": 1,
            "concurrent_streams": False,
        },
    )

    # Add runs
    for run in runs:
        writer.add_run(
            config_id=config_id,
            rep_id=run["rep_id"],
            run_data=run,
        )

    # Save to file
    writer.save(OUTPUT_FILE)

    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"End time: {datetime.now().isoformat()}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Reproducibility: {'✓ PASS' if is_reproducible else '✗ FAIL'}")
    if not is_reproducible:
        print(f"Error: {error_message}")
    print()

    # Calculate statistics
    runtimes = [run["runtime_seconds"] for run in runs]
    avg_runtime = sum(runtimes) / len(runtimes)
    print(f"Average runtime: {avg_runtime:.2f}s")
    print(f"Tokens/second: {MAX_NEW_TOKENS / avg_runtime:.2f}")
    print()

    if not is_reproducible:
        print("⚠ WARNING: Non-determinism detected!")
        print("Please investigate before proceeding with other experiments.")
        sys.exit(1)

    print("✓ Experiment 0 completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
