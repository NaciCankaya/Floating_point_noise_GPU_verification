"""
Experiment 2: Compilation Detection

Tests whether torch.compile() creates detectable activation differences.

This is simpler than batch size - just one additional configuration to test.
Demonstrates the common pattern for most experiments.
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
    ExperimentReader,
    prepare_reference_prompt
)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_compile_experiment(
    model,
    tokenizer,
    extractor,
    prompt_inputs,
    num_reps: int = 3
) -> list:
    """Run experiment with compiled model."""
    
    logger.info("\nRunning with compiled model...")
    
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
    parser = argparse.ArgumentParser(description="Compilation Experiment")
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
                       default="compile_experiment.json",
                       help="Output JSON path")
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("EXPERIMENT 2: COMPILATION DETECTION")
    logger.info("="*80)
    logger.info(f"Hardware: {args.hardware}")
    logger.info(f"Model: {args.model}")
    
    # Load model WITH compilation
    logger.info("\nLoading and compiling model...")
    model, tokenizer = load_model(
        model_name=args.model,
        compile_model=True,  # <-- Key difference from baseline
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
        experiment_type="compile",
        variable_tested="compile",
        model_name=args.model,
        sequence_length=6000,
        decode_steps=30
    )
    
    # Add compiled configuration
    config_id = f"{args.hardware}_compile"
    
    writer.add_configuration(
        config_id=config_id,
        hardware=args.hardware,
        variable_value=True,  # compile=True
        fixed_params={
            "batch_size": 1,
            "compile": True,
            "attention_impl": "flash_attention_2",
            "quantization": "awq-int4",
            "tp_size": 1,
            "ep_size": 1,
            "concurrent_streams": False
        }
    )
    
    # Run experiment
    results = run_compile_experiment(
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
    
    # Merge with reference baseline (compile=False)
    if Path(args.reference_json).exists():
        logger.info("\nMerging with reference baseline (compile=False)...")
        writer.merge_with_reference(args.reference_json)
    else:
        logger.error(f"Reference baseline not found: {args.reference_json}")
        logger.error("Cannot proceed without baseline for comparison")
        sys.exit(1)
    
    # Save
    logger.info(f"\nSaving results to {args.output}...")
    writer.save(args.output)
    
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 2 COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {args.output}")
    
    avg_runtime = sum(r["runtime_seconds"] for r in results) / 3
    logger.info(f"Average runtime (compiled): {avg_runtime:.2f}s")
    
    # Load baseline for comparison
    reader = ExperimentReader(args.reference_json)
    baseline_run = reader.get_baseline_run(args.hardware, rep_id=0)
    
    if baseline_run:
        baseline_runtime = baseline_run["runtime_seconds"]
        logger.info(f"Baseline runtime (eager): {baseline_runtime:.2f}s")
        logger.info(f"Speedup: {baseline_runtime / avg_runtime:.2f}x")


if __name__ == "__main__":
    main()
