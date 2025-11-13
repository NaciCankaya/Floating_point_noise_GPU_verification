#!/usr/bin/env python3
"""
Inference runner utilities for ablation experiments

Orchestrates model inference with timing, signal extraction, and repetitions.
"""

import torch
import time
from typing import Dict, List, Optional
from datetime import datetime
from .extraction import extract_signals


def run_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 30,
    layer_indices: List[int] = [1, 2, 4, 12, 39],
    positions: List[int] = [-3, -2, -1],
    top_k_logprobs: int = 10,
    use_cache: bool = False,
) -> Dict:
    """
    Run a single inference with signal extraction and timing.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt text
        max_new_tokens: Number of tokens to generate
        layer_indices: Layers to extract signals from
        positions: Token positions to extract (-3, -2, -1)
        top_k_logprobs: Number of top logprobs to record
        use_cache: Whether to use KV cache

    Returns:
        dict: {
            "timestamp": ISO timestamp,
            "runtime_seconds": float,
            "prompt_text": str,
            "decode_steps": List[Dict],  # Signal data for each step
            "generated_tokens": List[int],
            "generated_text": str,
        }
    """
    # Clear cache before run
    torch.cuda.empty_cache()

    # Record start time
    start_time = time.time()
    timestamp = datetime.now().isoformat()

    # Run inference with signal extraction
    decode_steps, token_ids, token_texts = extract_signals(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        layer_indices=layer_indices,
        positions=positions,
        top_k_logprobs=top_k_logprobs,
        use_cache=use_cache,
    )

    # Record end time
    runtime = time.time() - start_time

    # Combine token texts
    generated_text = "".join(token_texts)

    return {
        "timestamp": timestamp,
        "runtime_seconds": runtime,
        "prompt_text": prompt,
        "decode_steps": decode_steps,
        "generated_tokens": token_ids,
        "generated_text": generated_text,
    }


def run_multiple_repetitions(
    model,
    tokenizer,
    prompt: str,
    num_reps: int = 3,
    max_new_tokens: int = 30,
    layer_indices: List[int] = [1, 2, 4, 12, 39],
    positions: List[int] = [-3, -2, -1],
    top_k_logprobs: int = 10,
    use_cache: bool = False,
    verify_reproducibility: bool = True,
) -> tuple[List[Dict], bool, Optional[str]]:
    """
    Run multiple repetitions of inference to verify reproducibility.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt text
        num_reps: Number of repetitions (default: 3)
        max_new_tokens: Number of tokens to generate
        layer_indices: Layers to extract from
        positions: Token positions to extract
        top_k_logprobs: Number of top logprobs
        use_cache: Whether to use KV cache
        verify_reproducibility: Whether to check bit-exact reproducibility

    Returns:
        tuple: (runs, is_reproducible, error_message)
            - runs: List of run results
            - is_reproducible: True if all runs are bit-exact identical
            - error_message: Description of differences if not reproducible
    """
    print(f"\nRunning {num_reps} repetitions...")

    runs = []
    for rep_id in range(num_reps):
        print(f"\nRepetition {rep_id + 1}/{num_reps}:")
        run_result = run_inference(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            layer_indices=layer_indices,
            positions=positions,
            top_k_logprobs=top_k_logprobs,
            use_cache=use_cache,
        )

        run_result["rep_id"] = rep_id
        runs.append(run_result)

        print(f"  Runtime: {run_result['runtime_seconds']:.2f}s")
        print(f"  Generated: {run_result['generated_text'][:50]}...")

    # Verify reproducibility if requested
    is_reproducible = True
    error_message = None

    if verify_reproducibility and num_reps > 1:
        print("\nVerifying bit-exact reproducibility...")

        # Check token sequences
        first_tokens = runs[0]["generated_tokens"]
        for i, run in enumerate(runs[1:], 1):
            if run["generated_tokens"] != first_tokens:
                is_reproducible = False
                error_message = f"Token mismatch between rep 0 and rep {i}"
                print(f"  ✗ Rep {i}: Token sequence differs")
                break

        if is_reproducible:
            print("  ✓ All token sequences identical")

            # Check signal reproducibility (sample a few)
            # Compare hidden states from last layer, last step, last position
            first_signal = runs[0]["decode_steps"][-1]["hidden_states"]
            first_layer = list(first_signal.keys())[0]  # e.g., "layer_39"
            first_pos = list(first_signal[first_layer].keys())[0]  # e.g., "pos_-1"
            first_values = first_signal[first_layer][first_pos]

            for i, run in enumerate(runs[1:], 1):
                run_values = run["decode_steps"][-1]["hidden_states"][first_layer][first_pos]

                # Check if bit-exact
                if run_values != first_values:
                    is_reproducible = False
                    error_message = f"Hidden state mismatch between rep 0 and rep {i} at {first_layer}/{first_pos}"
                    print(f"  ✗ Rep {i}: Hidden states differ")

                    # Compute L2 distance for debugging
                    import numpy as np
                    diff = np.array(run_values) - np.array(first_values)
                    l2_dist = np.linalg.norm(diff)
                    print(f"    L2 distance: {l2_dist:.2e}")
                    break

            if is_reproducible:
                print("  ✓ All signals bit-exact identical")

    if is_reproducible:
        print("\n✓ REPRODUCIBILITY VERIFIED: All runs are bit-exact identical")
    else:
        print(f"\n✗ REPRODUCIBILITY FAILED: {error_message}")
        print("  WARNING: Non-determinism detected!")

    return runs, is_reproducible, error_message


def compare_runs_l2(run1: Dict, run2: Dict, layer_idx: int = 39, position: int = -1) -> float:
    """
    Compute L2 distance between two runs at a specific layer and position.

    Args:
        run1: First run result
        run2: Second run result
        layer_idx: Layer index to compare
        position: Token position to compare

    Returns:
        float: L2 distance
    """
    import numpy as np

    layer_key = f"layer_{layer_idx}"
    pos_key = f"pos_{position}"

    # Extract hidden states from last decode step
    hidden1 = run1["decode_steps"][-1]["hidden_states"][layer_key][pos_key]
    hidden2 = run2["decode_steps"][-1]["hidden_states"][layer_key][pos_key]

    # Compute L2 distance
    diff = np.array(hidden1) - np.array(hidden2)
    l2_dist = np.linalg.norm(diff)

    return l2_dist
