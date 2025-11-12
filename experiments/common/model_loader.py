"""Model loading utilities for various configurations."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(
    model_name: str = "QuixiAI/Qwen3-30B-A3B-AWQ",
    compile_model: bool = False,
    attention_impl: str = "flash_attention_2",
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    tp_size: Optional[int] = None,
    ep_size: Optional[int] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load Qwen3 model with specified configuration.
    
    Args:
        model_name: HuggingFace model identifier
        compile_model: Whether to use torch.compile()
        attention_impl: "flash_attention_2" or "eager"
        torch_dtype: Data type for computation (BF16 recommended)
        device_map: Device placement strategy
        tp_size: Tensor parallelism size (for multi-GPU)
        ep_size: Expert parallelism size (for MoE)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")
    logger.info(f"  Compile: {compile_model}")
    logger.info(f"  Attention: {attention_impl}")
    logger.info(f"  TP: {tp_size}, EP: {ep_size}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Base model loading arguments
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": True,
        "attn_implementation": attention_impl,
    }
    
    # Add parallelism if specified (for vLLM or tensor-parallel setups)
    # Note: For pure transformers, TP/EP requires additional setup
    # This is a placeholder - actual implementation depends on framework
    if tp_size or ep_size:
        logger.warning("TP/EP requires special setup - ensure proper environment")
        # For transformers with accelerate:
        # model_kwargs["device_map"] = custom_device_map
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Compile if requested
    if compile_model:
        logger.info("Compiling model with torch.compile()...")
        model = torch.compile(model, mode="reduce-overhead")
    
    # Set to eval mode
    model.eval()
    
    logger.info(f"Model loaded successfully")
    logger.info(f"  Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    return model, tokenizer


def prepare_prompt(tokenizer, text: str, max_length: int = 6000) -> dict:
    """
    Prepare a prompt for inference.
    
    Args:
        tokenizer: HuggingFace tokenizer
        text: Raw text content
        max_length: Target token length
    
    Returns:
        Tokenized inputs ready for model
    """
    # Use chat template with summarization task
    messages = [
        {"role": "user", "content": f"{text}\n\nProvide a detailed summary."}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )
    
    return inputs
