"""
Experiment 1: Batch Size Detection

Tests whether different batch sizes create detectable activation differences.

Key challenge: Must use distinct prompts for batch neighbors while keeping
reference sequence (position 0) identical across batch size configurations.
"""

import torch
import sys
import gc
import argparse
from pathlib import Path

# Add common utilities to path
sys.path.append(str(Path(__file__).parent.parent))

from common import (
    load_model,
    ExtractionPipeline,
    InferenceRunner,
    ExperimentWriter,
    ExperimentReader,
    prepare_batch_prompts
)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_batch_experiment(
    model,
    tokenizer,
    extractor,
    batch_size: int,
    num_reps: int = 3
) -> list:
    """
    Run experiment for a specific batch size.
    
    Args:
        model: Loaded model
        tokenizer: Tokenizer
        extractor: ExtractionPipeline
        batch_size: Batch size to test (1, 2, or 4)
        num_reps: Number of repetitions
        
    Returns:
        List of extraction results (one per rep)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Running batch size = {batch_size}")
    logger.info(f"{'='*80}")
    
    # Prepare prompts for this batch size using common utility
    prompts = prepare_batch_prompts(tokenizer, batch_size=batch_size, max_length=6000)
    
    # Create runner
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
        
        # Clear cache between reps
        if rep > 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        # Extract input_ids from prompts
        input_ids_list = [p.input_ids for p in prompts]
        
        # Run batched inference
        # Note: We extract only from sequence 0 (reference)
        result = runner.run_batch_with_extraction(
            input_ids_list=input_ids_list,
            batch_size=batch_size
        )
        
        results.append(result[0])  # Only sequence 0 has extractions
        
        logger.info(f"  Runtime: {result[0]['runtime_seconds']:.2f}s")
    
    # Verify bit-exact reproducibility
    logger.info("\nVerifying reproducibility...")
    for i in range(1, num_reps):
        is_exact, diffs = extractor.verify_bit_exact(
            results[0]["decode_steps"][0],
            results[i]["decode_steps"][0]
        )
        
        if is_exact:
            logger.info(f"  Rep 0 vs Rep {i}: BIT-EXACT ✓")
        else:
            logger.warning(f"  Rep 0 vs Rep {i}: DIFFERS!")
            logger.warning(f"    Differences: {diffs}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch Size Experiment")
    parser.add_argument("--hardware", type=str, required=True, 
                       choices=["A100-80GB", "H100"],
                       help="Hardware type")
    parser.add_argument("--model", type=str, 
                       default="QuixiAI/Qwen3-30B-A3B-AWQ",
                       help="Model name")
    parser.add_argument("--reference-json", type=str,
                       default="reference_baseline.json",
                       help="Path to reference baseline JSON")
    parser.add_argument("--output", type=str,
                       default="batch_size_experiment.json",
                       help="Output JSON path")
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("EXPERIMENT 1: BATCH SIZE DETECTION")
    logger.info("="*80)
    logger.info(f"Hardware: {args.hardware}")
    logger.info(f"Model: {args.model}")
    
    # Load model
    logger.info("\nLoading model...")
    model, tokenizer = load_model(
        model_name=args.model,
        compile_model=False,
        attention_impl="flash_attention_2"
    )
    
    # Create extraction pipeline
    extractor = ExtractionPipeline(
        layers=[1, 2, 4, 12, 39],  # Adjust last layer based on actual model
        positions=[-3, -2, -1],
        top_k_logprobs=10
    )
    
    # Prepare prompts (will be created per batch size)
    # The prepare_batch_prompts utility ensures consistency
    # Position 0: Reference text (same across all experiments)
    # Positions 1+: Distinct neighbors (for bs > 1)
    
    # Initialize experiment writer
    writer = ExperimentWriter(
        experiment_type="batch_size",
        variable_tested="batch_size",
        model_name=args.model,
        sequence_length=6000,
        decode_steps=30
    )
    
    # Batch sizes to test
    batch_sizes = [1, 2, 4]
    
    # Run experiments for each batch size
    all_results = {}
    
    for bs in batch_sizes:
        config_id = f"{args.hardware}_bs{bs}"
        
        # Add configuration
        writer.add_configuration(
            config_id=config_id,
            hardware=args.hardware,
            variable_value=bs,
            fixed_params={
                "compile": False,
                "attention_impl": "flash_attention_2",
                "quantization": "awq-int4",
                "tp_size": 1,
                "ep_size": 1,
                "concurrent_streams": False
            }
        )
        
        # Run experiment
        results = run_batch_experiment(
            model=model,
            tokenizer=tokenizer,
            extractor=extractor,
            batch_size=bs,
            num_reps=3
        )
        
        # Store results
        for rep_id, result in enumerate(results):
            writer.add_run(
                config_id=config_id,
                rep_id=rep_id,
                extraction_result=result,
                runtime_seconds=result["runtime_seconds"],
                prompt_text=result["prompt_text"][:500]  # Truncate for storage
            )
        
        all_results[config_id] = results
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    # Merge with reference baseline if exists
    if Path(args.reference_json).exists():
        logger.info("\nMerging with reference baseline...")
        writer.merge_with_reference(args.reference_json)
    else:
        logger.warning(f"Reference baseline not found: {args.reference_json}")
        logger.warning("Proceeding without baseline merge")
    
    # Save results
    logger.info(f"\nSaving results to {args.output}...")
    writer.save(args.output)
    
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 1 COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {args.output}")
    
    # Summary
    logger.info("\nSummary:")
    for bs in batch_sizes:
        config_id = f"{args.hardware}_bs{bs}"
        avg_runtime = sum(r["runtime_seconds"] for r in all_results[config_id]) / 3
        logger.info(f"  Batch size {bs}: avg runtime = {avg_runtime:.2f}s")


if __name__ == "__main__":
    main()
