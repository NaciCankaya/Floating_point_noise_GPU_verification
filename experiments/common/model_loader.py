#!/usr/bin/env python3
"""
Model loading utilities for ablation experiments

Supports loading Qwen3-30B-A3B in various quantization formats with different configurations.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Literal
import os


def load_model(
    model_name: str = "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
    quantization: Literal["gptq", "bnb", "awq", "none"] = "gptq",
    attn_implementation: str = "flash_attention_2",
    torch_compile: bool = False,
    device_map: str = "auto",
    cache_dir: Optional[str] = None,
    torch_dtype = torch.bfloat16,
    **kwargs
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model with specified configuration.

    Args:
        model_name: HuggingFace model identifier
        quantization: Quantization method ("gptq", "bnb", "awq", "none")
        attn_implementation: Attention implementation ("flash_attention_2", "eager", "sdpa")
        torch_compile: Whether to compile the model with torch.compile()
        device_map: Device mapping strategy
        cache_dir: Cache directory for model downloads
        torch_dtype: PyTorch dtype for model weights
        **kwargs: Additional arguments passed to AutoModelForCausalLM.from_pretrained()

    Returns:
        tuple: (model, tokenizer)
    """
    # Set cache directory
    if cache_dir is None:
        cache_dir = os.environ.get('HF_HOME', '/workspace/huggingface_cache')

    # Set environment variables for HuggingFace cache
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_DATASETS_CACHE'] = cache_dir

    print(f"Loading model: {model_name}")
    print(f"  Quantization: {quantization}")
    print(f"  Attention: {attn_implementation}")
    print(f"  Torch compile: {torch_compile}")
    print(f"  Device map: {device_map}")
    print(f"  Cache dir: {cache_dir}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    # Configure model loading based on quantization
    model_kwargs = {
        "cache_dir": cache_dir,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        **kwargs
    }

    # Add attention implementation if not eager
    if attn_implementation != "eager":
        model_kwargs["attn_implementation"] = attn_implementation

    # Handle quantization-specific settings
    if quantization == "gptq":
        # GPTQ models are already quantized in the checkpoint
        print("Loading GPTQ model...")
    elif quantization == "bnb":
        # BitsAndBytes quantization loaded via model
        print("Loading BNB model...")
    elif quantization == "awq":
        # AWQ models are already quantized in the checkpoint
        print("Loading AWQ model...")
    elif quantization == "none":
        # Full precision model
        model_kwargs["torch_dtype"] = torch_dtype
        print(f"Loading full precision model ({torch_dtype})...")
    else:
        raise ValueError(f"Unknown quantization method: {quantization}")

    # Load model
    print("Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

    # Apply torch.compile if requested
    if torch_compile:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    print(f"✓ Model loaded successfully")
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  Dtype: {next(model.parameters()).dtype}")
    print()

    return model, tokenizer


def get_model_info(model, tokenizer) -> dict:
    """
    Extract metadata about the loaded model.

    Returns:
        dict: Model metadata including versions, device, dtype, etc.
    """
    import platform
    import sys

    return {
        "model_name": model.config._name_or_path if hasattr(model.config, '_name_or_path') else "unknown",
        "model_type": model.config.model_type if hasattr(model.config, 'model_type') else "unknown",
        "vocab_size": tokenizer.vocab_size,
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "platform": platform.platform(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }
