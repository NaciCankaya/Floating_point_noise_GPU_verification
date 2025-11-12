"""Prompt creation and chat template utilities.

These must be identical across all experiments to ensure fair comparisons.
"""

from typing import Dict, List
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Reference text - must be identical across all experiments
# In production, this should be loaded from a fixed PDF
REFERENCE_TEXT = "This is the reference text that will be used across all experiments. " * 500


def load_reference_text(pdf_path: str = None) -> str:
    """
    Load reference text for experiments.
    
    This MUST be identical across all experiments for valid comparisons.
    
    Args:
        pdf_path: Optional path to PDF file. If None, uses default text.
        
    Returns:
        Text string (~6k tokens when tokenized)
    """
    if pdf_path:
        # TODO: Implement PDF loading
        # For now, use default
        logger.warning(f"PDF loading not implemented, using default text")
    
    return REFERENCE_TEXT


def load_batch_neighbor_texts(num_neighbors: int = 3, pdf_dir: str = None) -> List[str]:
    """
    Load texts for batch neighbors (experiments with bs > 1).
    
    These should be DISTINCT from reference text and from each other.
    
    Args:
        num_neighbors: Number of additional texts needed
        pdf_dir: Directory containing PDF files
        
    Returns:
        List of text strings
    """
    if pdf_dir:
        # TODO: Implement loading from multiple PDFs
        logger.warning(f"PDF loading not implemented, using generated texts")
    
    # Generate distinct texts
    texts = []
    for i in range(num_neighbors):
        text = f"This is batch neighbor text {i+1} for larger batches. " * 500
        texts.append(text)
    
    return texts


def create_prompt_with_template(
    tokenizer,
    text: str,
    task: str = "Provide a detailed summary.",
    max_length: int = 6000
) -> Dict:
    """
    Create a prompt using the model's chat template.
    
    This ensures consistent formatting across all experiments.
    
    Args:
        tokenizer: HuggingFace tokenizer
        text: Raw text content
        task: Task instruction to append
        max_length: Maximum token length
        
    Returns:
        Dictionary with input_ids and attention_mask
    """
    # Use chat template
    messages = [
        {"role": "user", "content": f"{text}\n\n{task}"}
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
    
    actual_length = inputs.input_ids.shape[1]
    logger.info(f"Prompt created: {actual_length} tokens")
    
    return inputs


def prepare_reference_prompt(tokenizer, max_length: int = 6000) -> Dict:
    """
    Prepare the standard reference prompt used across all experiments.
    
    This is a convenience wrapper that ensures consistency.
    
    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum token length
        
    Returns:
        Tokenized inputs
    """
    text = load_reference_text()
    return create_prompt_with_template(tokenizer, text, max_length=max_length)


def prepare_batch_prompts(
    tokenizer,
    batch_size: int,
    max_length: int = 6000
) -> List[Dict]:
    """
    Prepare prompts for batch experiments.
    
    Position 0: Reference text (same as all other experiments)
    Positions 1+: Distinct neighbor texts
    
    Args:
        tokenizer: HuggingFace tokenizer
        batch_size: Total batch size
        max_length: Maximum token length per sequence
        
    Returns:
        List of tokenized inputs (length = batch_size)
    """
    prompts = []
    
    # Position 0: Reference
    reference_text = load_reference_text()
    reference_prompt = create_prompt_with_template(
        tokenizer,
        reference_text,
        max_length=max_length
    )
    prompts.append(reference_prompt)
    
    # Positions 1+: Neighbors
    if batch_size > 1:
        neighbor_texts = load_batch_neighbor_texts(num_neighbors=batch_size - 1)
        
        for neighbor_text in neighbor_texts:
            neighbor_prompt = create_prompt_with_template(
                tokenizer,
                neighbor_text,
                max_length=max_length
            )
            prompts.append(neighbor_prompt)
    
    return prompts


def verify_prompt_consistency(prompt1: Dict, prompt2: Dict) -> bool:
    """
    Verify that two prompts are identical (for reproducibility checks).
    
    Args:
        prompt1: First tokenized prompt
        prompt2: Second tokenized prompt
        
    Returns:
        True if identical, False otherwise
    """
    import torch
    
    ids_match = torch.equal(prompt1.input_ids, prompt2.input_ids)
    
    if not ids_match:
        logger.error("Prompt mismatch detected!")
        return False
    
    return True
