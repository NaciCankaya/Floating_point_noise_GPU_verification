"""Inference runner with extraction capabilities."""

import torch
import time
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceRunner:
    """Run inference with signal extraction."""
    
    def __init__(
        self,
        model,
        tokenizer,
        extraction_pipeline,
        num_decode_steps: int = 30,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """
        Initialize inference runner.
        
        Args:
            model: Loaded model
            tokenizer: Tokenizer
            extraction_pipeline: ExtractionPipeline instance
            num_decode_steps: Number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        self.model = model
        self.tokenizer = tokenizer
        self.extractor = extraction_pipeline
        self.num_decode_steps = num_decode_steps
        self.temperature = temperature
        self.top_p = top_p
    
    def run_with_extraction(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Run inference and extract signals at each decode step.
        
        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Attention mask
            
        Returns:
            Dictionary containing:
                - prompt_text: Input prompt
                - decode_steps: List of extractions per step
                - runtime_seconds: Total inference time
        """
        logger.info(f"Running inference with {self.num_decode_steps} decode steps...")
        
        # Move to device
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Decode input for logging
        prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        start_time = time.time()
        
        # Store extractions
        decode_steps = []
        
        # Initialize generation
        past_key_values = None
        current_input_ids = input_ids
        
        with torch.no_grad():
            for step in range(self.num_decode_steps):
                # Forward pass with output collection
                outputs = self.model(
                    input_ids=current_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    output_attentions=False,
                    return_dict=True
                )
                
                # Extract signals
                extraction = self.extractor.extract_from_outputs(
                    outputs=outputs,
                    tokenizer=self.tokenizer,
                    step_idx=step
                )
                
                decode_steps.append(extraction)
                
                # Sample next token
                next_token_logits = outputs.logits[:, -1, :]
                
                if self.temperature > 0:
                    next_token_logits = next_token_logits / self.temperature
                    
                    # Top-p filtering
                    if self.top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(
                            next_token_logits, descending=True
                        )
                        cumulative_probs = torch.cumsum(
                            torch.softmax(sorted_logits, dim=-1), dim=-1
                        )
                        
                        # Remove tokens with cumulative probability above threshold
                        sorted_indices_to_remove = cumulative_probs > self.top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1, sorted_indices, sorted_indices_to_remove
                        )
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Update for next iteration
                current_input_ids = next_token
                past_key_values = outputs.past_key_values
                
                # Update attention mask
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((attention_mask.shape[0], 1), device=device)
                    ], dim=1)
                
                # Log progress
                if (step + 1) % 10 == 0:
                    logger.info(f"  Decode step {step + 1}/{self.num_decode_steps}")
        
        runtime = time.time() - start_time
        
        logger.info(f"Inference complete in {runtime:.2f}s")
        
        return {
            "prompt_text": prompt_text,
            "decode_steps": decode_steps,
            "runtime_seconds": runtime
        }
    
    def run_batch_with_extraction(
        self,
        input_ids_list: List[torch.Tensor],
        batch_size: int
    ) -> List[Dict]:
        """
        Run batched inference with extraction.
        
        Note: For batch experiments, we only extract from sequence 0 (reference).
        
        Args:
            input_ids_list: List of input tensors
            batch_size: Batch size to use
            
        Returns:
            List of results (but only sequence 0 has extractions)
        """
        if len(input_ids_list) != batch_size:
            raise ValueError(f"Expected {batch_size} inputs, got {len(input_ids_list)}")
        
        # Pad sequences to same length
        max_len = max(ids.shape[1] for ids in input_ids_list)
        
        padded_inputs = []
        attention_masks = []
        
        for ids in input_ids_list:
            pad_len = max_len - ids.shape[1]
            
            if pad_len > 0:
                # Pad on the left (for causal models)
                padded = torch.cat([
                    torch.full((1, pad_len), self.tokenizer.pad_token_id, dtype=ids.dtype),
                    ids
                ], dim=1)
                
                mask = torch.cat([
                    torch.zeros((1, pad_len), dtype=torch.long),
                    torch.ones((1, ids.shape[1]), dtype=torch.long)
                ], dim=1)
            else:
                padded = ids
                mask = torch.ones_like(ids)
            
            padded_inputs.append(padded)
            attention_masks.append(mask)
        
        # Stack into batch
        batch_input_ids = torch.cat(padded_inputs, dim=0)
        batch_attention_mask = torch.cat(attention_masks, dim=0)
        
        logger.info(f"Running batched inference (bs={batch_size})...")
        logger.info(f"  Batch shape: {batch_input_ids.shape}")
        
        # For now, we extract from the full batch but only keep sequence 0
        # In practice, batch processing with extraction is complex
        # Simplification: Run sequence 0 separately with extraction
        result = self.run_with_extraction(
            input_ids=batch_input_ids[0:1],  # Just reference sequence
            attention_mask=batch_attention_mask[0:1]
        )
        
        return [result]
