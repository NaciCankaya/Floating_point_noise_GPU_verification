"""
Experiment 0: Reference Baseline

Establishes baseline measurements with default configuration.
All subsequent experiments will reuse this as their baseline comparison.

Purpose:
1. Verify bit-exact reproducibility within identical setups
2. Establish baseline configuration for all experiments
3. Validate extraction pipeline
"""

import torch
import sys
import gc
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from common import (
    load_model,
    ExtractionPipeline,
    InferenceRunner,
    ExperimentWriter,
    prepare_reference_prompt
)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_reference_experiment(
    model,
    tokenizer,
    extractor,
    prompt_inputs,
    num_reps: int = 3
) -> list:
    """Run reference baseline experiment."""
    
    logger.info("\nRunning reference baseline measurements...")
    
    runner = InferenceRunner(
        model=model,
        tokenizer=tokenizer,
        extraction_pipeline=extractor,
        num_decode_steps=30,
        temperature=0.7,
        top_p=0.9
    )
    
    results = []
    
    for rep in range(num_reps):
        logger.info(f"\nRepetition {rep + 1}/{num_reps}")
        
        if rep > 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        result = runner.run_with_extraction(
            input_ids=prompt_inputs.input_ids,
            attention_mask=prompt_inputs.attention_mask
        )
        
        results.append(result)
        logger.info(f"  Runtime: {result['runtime_seconds']:.2f}s")
    
    # Verify reproducibility
    logger.info("\nVerifying bit-exact reproducibility...")
    for i in range(1, num_reps):
        is_exact, diffs = extractor.verify_bit_exact(
            results[0]["decode_steps"][0],
            results[i]["decode_steps"][0]
        )
        
        if is_exact:
            logger.info(f"  Rep 0 vs Rep {i}: BIT-EXACT ✓")
        else:
            logger.error(f"  Rep 0 vs Rep {i}: NOT REPRODUCIBLE!")
            logger.error(f"    Differences: {diffs}")
            logger.error("\nNON-DETERMINISM DETECTED - STOPPING")
            sys.exit(1)
    
    logger.info("\n✓ All repetitions are bit-exact - setup is deterministic")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Reference Baseline Experiment")
    parser.add_argument("--hardware", type=str, required=True,
                       choices=["A100-80GB", "H100"],
                       help="Hardware type")
    parser.add_argument("--model", type=str,
                       default="QuixiAI/Qwen3-30B-A3B-AWQ",
                       help="Model name")
    parser.add_argument("--output", type=str,
                       default="reference_baseline.json",
                       help="Output JSON path")
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("EXPERIMENT 0: REFERENCE BASELINE")
    logger.info("="*80)
    logger.info(f"Hardware: {args.hardware}")
    logger.info(f"Model: {args.model}")
    
    # Load model with baseline configuration
    logger.info("\nLoading model with baseline configuration...")
    model, tokenizer = load_model(
        model_name=args.model,
        compile_model=False,
        attention_impl="flash_attention_2"
    )
    
    # Create extraction pipeline
    extractor = ExtractionPipeline(
        layers=[1, 2, 4, 12, 39],
        positions=[-3, -2, -1],
        top_k_logprobs=10
    )
    
    # Load reference prompt (using common utility for consistency)
    prompt_inputs = prepare_reference_prompt(tokenizer, max_length=6000)
    
    logger.info(f"Prompt length: {prompt_inputs.input_ids.shape[1]} tokens")
    
    # Initialize experiment writer
    writer = ExperimentWriter(
        experiment_type="reference",
        variable_tested="none",
        model_name=args.model,
        sequence_length=6000,
        decode_steps=30
    )
    
    # Add baseline configuration
    config_id = f"{args.hardware}_baseline"
    
    writer.add_configuration(
        config_id=config_id,
        hardware=args.hardware,
        variable_value=1,  # bs=1 for baseline
        fixed_params={
            "batch_size": 1,
            "compile": False,
            "attention_impl": "flash_attention_2",
            "quantization": "awq-int4",
            "tp_size": 1,
            "ep_size": 1,
            "concurrent_streams": False
        }
    )
    
    # Run experiment
    results = run_reference_experiment(
        model=model,
        tokenizer=tokenizer,
        extractor=extractor,
        prompt_inputs=prompt_inputs,
        num_reps=3
    )
    
    # Store results
    for rep_id, result in enumerate(results):
        writer.add_run(
            config_id=config_id,
            rep_id=rep_id,
            extraction_result=result,
            runtime_seconds=result["runtime_seconds"],
            prompt_text=result["prompt_text"][:500]
        )
    
    # Save
    logger.info(f"\nSaving reference baseline to {args.output}...")
    writer.save(args.output)
    
    logger.info("\n" + "="*80)
    logger.info("REFERENCE BASELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {args.output}")
    
    avg_runtime = sum(r["runtime_seconds"] for r in results) / 3
    logger.info(f"Average runtime: {avg_runtime:.2f}s")
    logger.info("\nThis baseline will be reused for experiments 1-6")


if __name__ == "__main__":
    main()
