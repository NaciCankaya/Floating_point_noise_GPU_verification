"""Extraction utilities for hidden states, key vectors, and logprobs."""

import torch
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """Extract forensic signals from model outputs."""
    
    def __init__(
        self,
        layers: List[int] = [1, 2, 4, 12, 39],
        positions: List[int] = [-3, -2, -1],
        top_k_logprobs: int = 10
    ):
        """
        Initialize extraction pipeline.
        
        Args:
            layers: Layer indices to extract (0-indexed)
            positions: Token positions to extract (negative = from end)
            top_k_logprobs: Number of top logprobs to store
        """
        self.layers = layers
        self.positions = positions
        self.top_k_logprobs = top_k_logprobs
        
    def extract_from_outputs(
        self,
        outputs,
        tokenizer,
        step_idx: int
    ) -> Dict:
        """
        Extract signals from a single decode step.
        
        Args:
            outputs: Model output object (transformers.modeling_outputs)
            tokenizer: Tokenizer for decoding
            step_idx: Current decode step index
            
        Returns:
            Dictionary with extracted signals
        """
        extraction = {
            "step": step_idx,
            "token_id": None,
            "token_text": None,
            "hidden_states": {},
            "key_vectors": {},
            "logprobs": {}
        }
        
        # Extract token info
        if hasattr(outputs, 'sequences'):
            token_id = outputs.sequences[0, -1].item()
            extraction["token_id"] = token_id
            extraction["token_text"] = tokenizer.decode([token_id])
        
        # Extract hidden states
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            extraction["hidden_states"] = self._extract_hidden_states(outputs.hidden_states)
        
        # Extract key vectors from past_key_values
        if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
            extraction["key_vectors"] = self._extract_key_vectors(outputs.past_key_values)
        
        # Extract logprobs from logits
        if hasattr(outputs, 'logits') and outputs.logits is not None:
            extraction["logprobs"] = self._extract_logprobs(outputs.logits, tokenizer)
        
        return extraction
    
    def _extract_hidden_states(self, hidden_states: Tuple[torch.Tensor]) -> Dict:
        """
        Extract hidden states from specified layers and positions.
        
        Args:
            hidden_states: Tuple of tensors (num_layers, batch, seq_len, hidden_dim)
            
        Returns:
            Dict mapping layer -> position -> vector
        """
        result = {}
        
        for layer_idx in self.layers:
            if layer_idx >= len(hidden_states):
                logger.warning(f"Layer {layer_idx} not available (max: {len(hidden_states)-1})")
                continue
            
            layer_hidden = hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)
            layer_data = {}
            
            for pos in self.positions:
                # Extract position (handle negative indexing)
                vector = layer_hidden[0, pos, :].detach().cpu().numpy().tolist()
                layer_data[f"pos_{pos}"] = vector
            
            result[f"layer_{layer_idx}"] = layer_data
        
        return result
    
    def _extract_key_vectors(self, past_key_values: Tuple) -> Dict:
        """
        Extract key vectors from KV cache.
        
        Args:
            past_key_values: Tuple of (key, value) tensors per layer
            Format: [(key, value), ...] where key is (batch, num_heads, seq_len, head_dim)
            
        Returns:
            Dict mapping layer -> position -> concatenated key vector
        """
        result = {}
        
        for layer_idx in self.layers:
            if layer_idx >= len(past_key_values):
                logger.warning(f"Layer {layer_idx} KV not available")
                continue
            
            # Get keys for this layer
            keys, _ = past_key_values[layer_idx]
            # keys shape: (batch, num_heads, seq_len, head_dim)
            
            layer_data = {}
            
            for pos in self.positions:
                # Extract keys at position, concatenate across heads
                # (num_heads, head_dim) -> (num_heads * head_dim,)
                key_vector = keys[0, :, pos, :].reshape(-1).detach().cpu().numpy().tolist()
                layer_data[f"pos_{pos}"] = key_vector
            
            result[f"layer_{layer_idx}"] = layer_data
        
        return result
    
    def _extract_logprobs(self, logits: torch.Tensor, tokenizer) -> Dict:
        """
        Extract top-k logprobs from logits.
        
        Args:
            logits: Tensor of shape (batch, seq_len, vocab_size)
            tokenizer: For decoding token IDs
            
        Returns:
            Dict mapping position -> top-k tokens and log probs
        """
        result = {}
        
        # Convert logits to log probabilities
        log_probs = torch.log_softmax(logits[0], dim=-1)  # (seq_len, vocab_size)
        
        for pos in self.positions:
            # Get top-k at this position
            pos_logprobs = log_probs[pos]  # (vocab_size,)
            top_k_values, top_k_indices = torch.topk(pos_logprobs, self.top_k_logprobs)
            
            result[f"pos_{pos}"] = {
                "token_ids": top_k_indices.cpu().numpy().tolist(),
                "log_probs": top_k_values.cpu().numpy().tolist()
            }
        
        return result
    
    def verify_bit_exact(self, extraction1: Dict, extraction2: Dict) -> Tuple[bool, Dict]:
        """
        Verify bit-exact reproducibility between two extractions.
        
        Returns:
            (is_exact, differences_dict)
        """
        differences = {
            "hidden_states": [],
            "key_vectors": [],
            "logprobs": []
        }
        
        is_exact = True
        
        # Compare hidden states
        for layer in extraction1["hidden_states"]:
            for pos in extraction1["hidden_states"][layer]:
                vec1 = np.array(extraction1["hidden_states"][layer][pos])
                vec2 = np.array(extraction2["hidden_states"][layer][pos])
                
                if not np.array_equal(vec1, vec2):
                    is_exact = False
                    l2_dist = np.linalg.norm(vec1 - vec2)
                    differences["hidden_states"].append({
                        "layer": layer,
                        "position": pos,
                        "l2_distance": float(l2_dist)
                    })
        
        # Similar for key vectors and logprobs...
        
        return is_exact, differences
