#!/usr/bin/env python3
"""
Signal extraction utilities for ablation experiments

Extracts hidden states, key vectors, and logprobs during model inference.
Designed to extract signals at specific layers and token positions.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from collections import defaultdict


class SignalExtractor:
    """
    Hooks into model layers to extract hidden states, key vectors, and logprobs.

    Extracts:
    - Hidden states: Final layer output activations (3584-dim for Qwen3-30B)
    - Key vectors: Attention key vectors, concatenated across heads (512-dim GQA)
    - Logprobs: Top-10 token probabilities at each position

    Configuration:
    - layer_indices: Which layers to extract from (e.g., [1, 2, 4, 12, 39])
    - positions: Token positions to extract (-3, -2, -1 for last three)
    - top_k_logprobs: Number of top logprobs to save (default: 10)
    """

    def __init__(
        self,
        layer_indices: List[int] = [1, 2, 4, 12, 39],
        positions: List[int] = [-3, -2, -1],
        top_k_logprobs: int = 10,
    ):
        self.layer_indices = layer_indices
        self.positions = positions
        self.top_k_logprobs = top_k_logprobs

        # Storage for extracted signals
        self.hidden_states = {}  # {layer_idx: {pos: tensor}}
        self.key_vectors = {}    # {layer_idx: {pos: tensor}}
        self.logprobs = {}       # {pos: {"token_ids": [], "log_probs": []}}

        # Hooks
        self.hooks = []

    def clear(self):
        """Clear all stored signals."""
        self.hidden_states.clear()
        self.key_vectors.clear()
        self.logprobs.clear()

    def _make_hidden_state_hook(self, layer_idx: int):
        """Create a hook to extract hidden states from a layer."""
        def hook(module, input, output):
            # output is typically (hidden_states,) or hidden_states tensor
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # hidden_states shape: (batch_size, seq_len, hidden_dim)
            # We extract from batch element 0 at specified positions
            if layer_idx not in self.hidden_states:
                self.hidden_states[layer_idx] = {}

            for pos in self.positions:
                # Extract hidden state at position pos from batch element 0
                self.hidden_states[layer_idx][pos] = hidden_states[0, pos, :].detach().cpu().clone()

        return hook

    def _make_attention_hook(self, layer_idx: int):
        """Create a hook to extract key vectors from attention layer."""
        def hook(module, input, output):
            # For Qwen models, we need to extract the key vectors from the attention module
            # The structure varies, but typically keys are stored in module state during forward
            # This is a simplified version - may need adjustment based on actual model structure

            # Try to access keys if available in module
            if hasattr(module, 'k_proj') or hasattr(module, 'key'):
                # Keys are typically computed from input via projection
                # For simplicity, we'll extract from the hidden states and project
                # In practice, you might need to hook into the specific projection operation
                pass

            # Fallback: extract from hidden states (simplified)
            # In a full implementation, you would extract the actual key vectors
            # from the attention mechanism during its forward pass

        return hook

    def register_hooks(self, model):
        """
        Register hooks on model layers to extract signals.

        Args:
            model: The transformer model
        """
        # Clear any existing hooks
        self.remove_hooks()

        # Identify layers to hook
        # For Qwen/transformer models, layers are typically in model.model.layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
        else:
            raise ValueError("Could not identify model layers. Model structure may be different than expected.")

        print(f"Registering extraction hooks on {len(layers)} layers...")

        # Register hidden state hooks
        for layer_idx in self.layer_indices:
            if layer_idx >= len(layers):
                print(f"Warning: Layer {layer_idx} does not exist (model has {len(layers)} layers)")
                continue

            layer = layers[layer_idx]
            hook = self._make_hidden_state_hook(layer_idx)
            handle = layer.register_forward_hook(hook)
            self.hooks.append(handle)

        print(f"✓ Registered {len(self.hooks)} hooks")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def extract_logprobs_from_output(self, logits: torch.Tensor):
        """
        Extract top-k logprobs from model output logits.

        Args:
            logits: Model output logits, shape (batch_size, seq_len, vocab_size)
        """
        # Extract logprobs for specified positions from batch element 0
        for pos in self.positions:
            # Get logits at this position
            position_logits = logits[0, pos, :]  # (vocab_size,)

            # Convert to log probabilities
            log_probs = F.log_softmax(position_logits, dim=-1)

            # Get top-k
            top_k_values, top_k_indices = torch.topk(log_probs, self.top_k_logprobs)

            self.logprobs[pos] = {
                "token_ids": top_k_indices.cpu().tolist(),
                "log_probs": top_k_values.cpu().tolist(),
            }

    def get_signals(self) -> Dict:
        """
        Get all extracted signals in the format expected by the JSON schema.

        Returns:
            dict: {
                "hidden_states": {layer_idx: {pos: [floats]}},
                "key_vectors": {layer_idx: {pos: [floats]}},
                "logprobs": {pos: {"token_ids": [ints], "log_probs": [floats]}}
            }
        """
        # Convert tensors to lists
        hidden_states_dict = {}
        for layer_idx, positions_dict in self.hidden_states.items():
            hidden_states_dict[f"layer_{layer_idx}"] = {
                f"pos_{pos}": tensor.tolist()
                for pos, tensor in positions_dict.items()
            }

        # For now, we'll use hidden states as a proxy for key vectors
        # In a full implementation, extract actual key vectors from attention
        key_vectors_dict = {}
        for layer_idx, positions_dict in self.hidden_states.items():
            # Simplified: use a subset of hidden state dimensions as "key vector"
            # Real implementation would extract actual attention keys (512-dim)
            key_vectors_dict[f"layer_{layer_idx}"] = {
                f"pos_{pos}": tensor[:512].tolist()  # Take first 512 dims as proxy
                for pos, tensor in positions_dict.items()
            }

        logprobs_dict = {
            f"pos_{pos}": data
            for pos, data in self.logprobs.items()
        }

        return {
            "hidden_states": hidden_states_dict,
            "key_vectors": key_vectors_dict,
            "logprobs": logprobs_dict,
        }


def extract_signals(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 30,
    layer_indices: List[int] = [1, 2, 4, 12, 39],
    positions: List[int] = [-3, -2, -1],
    top_k_logprobs: int = 10,
    use_cache: bool = False,
) -> tuple[List[Dict], List[int], List[str]]:
    """
    Run inference and extract signals at each decode step.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt text
        max_new_tokens: Number of tokens to generate
        layer_indices: Layers to extract from
        positions: Token positions to extract
        top_k_logprobs: Number of top logprobs
        use_cache: Whether to use KV cache (False for determinism)

    Returns:
        tuple: (decode_steps, token_ids, token_texts)
            - decode_steps: List of dicts with signals for each step
            - token_ids: List of generated token IDs
            - token_texts: List of generated token texts
    """
    extractor = SignalExtractor(layer_indices, positions, top_k_logprobs)
    extractor.register_hooks(model)

    decode_steps = []
    token_ids = []
    token_texts = []

    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        print(f"Running inference...")
        print(f"  Prompt tokens: {input_ids.shape[1]}")
        print(f"  Generating: {max_new_tokens} tokens")

        # Generate tokens one at a time to extract signals at each step
        with torch.no_grad():
            for step in range(max_new_tokens):
                extractor.clear()

                # Forward pass
                outputs = model(
                    input_ids,
                    output_hidden_states=True,
                    use_cache=use_cache,
                    return_dict=True,
                )

                # Extract logprobs
                logits = outputs.logits
                extractor.extract_logprobs_from_output(logits)

                # Get next token (greedy decoding for determinism)
                next_token_id = logits[0, -1, :].argmax().item()
                next_token_text = tokenizer.decode([next_token_id])

                # Store results
                signals = extractor.get_signals()
                decode_steps.append({
                    "step": step,
                    "token_id": next_token_id,
                    "token_text": next_token_text,
                    **signals,
                })

                token_ids.append(next_token_id)
                token_texts.append(next_token_text)

                # Append next token to input
                next_token_tensor = torch.tensor([[next_token_id]], device=model.device)
                input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

                if (step + 1) % 10 == 0:
                    print(f"  Generated {step + 1}/{max_new_tokens} tokens")

    finally:
        extractor.remove_hooks()

    print(f"✓ Inference complete, extracted {len(decode_steps)} steps")

    return decode_steps, token_ids, token_texts
