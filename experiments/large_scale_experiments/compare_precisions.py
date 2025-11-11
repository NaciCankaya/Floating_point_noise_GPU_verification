#!/usr/bin/env python3
"""
Compare INT8 vs FP8 Determinism Results

IMPORTANT: This script compares REPRODUCIBILITY within each experiment,
NOT outputs between experiments.

What we compare:
- INT8: Are runs 1-10 identical to each other? (determinism test)
- FP8:  Are runs 1-10 identical to each other? (determinism test)
- Result: Which quantization method maintains determinism?

What we DO NOT compare:
- INT8 outputs vs FP8 outputs (meaningless - different quantization)
- Cross-experiment logprobs or activations

Each experiment tests: "Can we reproduce inference bit-exactly?"
The comparison tests: "Which quantization method passes this test?"
"""

import json
import sys
from pathlib import Path

def load_result(filepath):
    """Load JSON result file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_reproducibility(data, label):
    """Analyze reproducibility metrics"""
    results = data['results']
    
    print(f"\n{'='*80}")
    print(f"{label} RESULTS")
    print(f"{'='*80}")
    print(f"Model: {data['config']['model']}")
    print(f"Quantization: {data['config']['quantization']}")
    print(f"Precision: {data['config'].get('precision', 'N/A')}")
    print()
    
    print("Reproducibility:")
    print(f"  Token sequences identical: {results['tokens_identical']}")
    print(f"  Logprobs bit-exact: {results['logprobs_exact']}")
    print(f"  Distributions bit-exact: {results['distributions_exact']}")
    print(f"  Perfect reproducibility: {results['perfect_reproducibility']}")
    print()
    
    # Timing stats
    if 'timing' in data:
        timing = data['timing']['statistics']
        print("Performance:")
        print(f"  Mean time: {timing['mean_time']:.3f}s (σ={timing['std_time']:.4f}s)")
        print(f"  Tokens/sec: {timing['mean_tokens_per_sec']:.1f} (σ={timing['std_tokens_per_sec']:.2f})")
        print(f"  Time/token: {timing['mean_time_per_token_ms']:.2f}ms (σ={timing['std_time_per_token_ms']:.3f}ms)")
    
    return results

def compare_results(int8_data, fp8_data):
    """Compare INT8 vs FP8 reproducibility results
    
    NOTE: We compare whether each experiment achieved reproducibility,
    NOT the actual outputs between experiments.
    
    INT8: Did runs 1-10 produce identical outputs?
    FP8:  Did runs 1-10 produce identical outputs?
    """
    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*80}")
    print()
    print("Comparing reproducibility within each experiment:")
    print("  INT8: Are all 10 INT8 runs identical to each other?")
    print("  FP8:  Are all 10 FP8 runs identical to each other?")
    print("  (NOT comparing INT8 outputs to FP8 outputs)")
    print()
    
    int8_results = int8_data['results']
    fp8_results = fp8_data['results']
    
    # Determinism comparison
    print("Determinism Comparison:")
    print(f"  INT8 perfect reproducibility: {int8_results['perfect_reproducibility']}")
    print(f"  FP8 perfect reproducibility: {fp8_results['perfect_reproducibility']}")
    
    if int8_results['perfect_reproducibility'] != fp8_results['perfect_reproducibility']:
        print("\n  [WARNING] DIVERGENCE DETECTED!")
        if fp8_results['perfect_reproducibility'] and not int8_results['perfect_reproducibility']:
            print("    => FP8 is deterministic, INT8 is not")
            print("    => Integer quantization breaks determinism")
            print("    => Native tensor core FP8 preserves determinism")
        elif int8_results['perfect_reproducibility'] and not fp8_results['perfect_reproducibility']:
            print("    => INT8 is deterministic, FP8 is not")
            print("    => Unexpected: FP8 should be more stable")
    else:
        if int8_results['perfect_reproducibility']:
            print("\n  [PASS] Both maintain perfect reproducibility")
        else:
            print("\n  [FAIL] Both break determinism")
    
    # Performance comparison
    if 'timing' in int8_data and 'timing' in fp8_data:
        int8_timing = int8_data['timing']['statistics']
        fp8_timing = fp8_data['timing']['statistics']
        
        print("\nPerformance Comparison:")
        int8_tps = int8_timing['mean_tokens_per_sec']
        fp8_tps = fp8_timing['mean_tokens_per_sec']
        speedup = fp8_tps / int8_tps
        
        print(f"  INT8: {int8_tps:.1f} tok/s")
        print(f"  FP8:  {fp8_tps:.1f} tok/s")
        print(f"  FP8 speedup: {speedup:.2f}x")
        
        if speedup > 1.1:
            print(f"    => FP8 is {(speedup-1)*100:.1f}% faster (expected for native tensor cores)")
        elif speedup < 0.9:
            print(f"    => INT8 is {(1/speedup-1)*100:.1f}% faster (unexpected)")
        else:
            print(f"    => Similar performance")

def forensic_implications(int8_data, fp8_data):
    """Analyze forensic implications"""
    int8_results = int8_data['results']
    fp8_results = fp8_data['results']
    
    print(f"\n{'='*80}")
    print("FORENSIC IMPLICATIONS")
    print(f"{'='*80}")
    print()
    
    if int8_results['perfect_reproducibility'] and fp8_results['perfect_reproducibility']:
        print("[PASS] OPTIMAL SCENARIO")
        print("  - Both INT8 and FP8 maintain determinism")
        print("  - Quantized inference can be forensically verified")
        print("  - Choose FP8 for best performance on H100")
        print()
        print("Verification Protocol:")
        print("  1. Use same quantization format (INT8 or FP8)")
        print("  2. Reproduce inference on trusted verifier")
        print("  3. Compare activations/logprobs for bit-exact match")
        
    elif fp8_results['perfect_reproducibility'] and not int8_results['perfect_reproducibility']:
        print("[WARNING] PARTIAL SUCCESS")
        print("  - FP8 maintains determinism (native tensor cores)")
        print("  - INT8 breaks determinism (dequantization issues)")
        print()
        print("Forensic Strategy:")
        print("  1. Require FP8 quantization for verifiable inference")
        print("  2. INT8 models cannot be forensically verified")
        print("  3. OR: Accept INT8 non-determinism, use statistical detection")
        print()
        print("Root Cause:")
        print("  - INT8 requires dequantization to FP32 (introduces rounding)")
        print("  - FP8 stays in floating-point domain (tensor core native)")
        
    elif not fp8_results['perfect_reproducibility'] and not int8_results['perfect_reproducibility']:
        print("[FAIL] QUANTIZATION BREAKS DETERMINISM")
        print("  - Both INT8 and FP8 are non-deterministic")
        print("  - Likely vLLM implementation or kernel issues")
        print()
        print("Implications:")
        print("  1. Forensic verification limited to FP16/BF16")
        print("  2. Quantized inference cannot be bit-exact verified")
        print("  3. Need alternative verification protocols:")
        print("     - Statistical deviation thresholds")
        print("     - Cross-quantization detection (INT8 vs FP8 comparison)")
        print("     - Timing-based forensics")
        
    else:
        print("[WARNING] UNEXPECTED: INT8 deterministic, FP8 non-deterministic")
        print("  - Requires deeper investigation")

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_int8_fp8.py <int8_results.json> <fp8_results.json>")
        print()
        print("This script compares REPRODUCIBILITY within each experiment:")
        print("  - INT8: Are the 10 INT8 runs identical to each other?")
        print("  - FP8:  Are the 10 FP8 runs identical to each other?")
        print()
        print("We do NOT compare INT8 outputs to FP8 outputs (meaningless).")
        print("We compare whether each quantization method maintains determinism.")
        sys.exit(1)
    
    int8_file = sys.argv[1]
    fp8_file = sys.argv[2]
    
    print("="*80)
    print("INT8 vs FP8 QUANTIZATION DETERMINISM COMPARISON")
    print("="*80)
    print()
    print("WHAT THIS SCRIPT COMPARES:")
    print("  INT8 Experiment: Reproducibility across 10 runs with same config")
    print("  FP8 Experiment:  Reproducibility across 10 runs with same config")
    print("  Result:          Which quantization method is deterministic?")
    print()
    print("WHAT THIS SCRIPT DOES NOT COMPARE:")
    print("  INT8 outputs vs FP8 outputs (different quantization = different outputs)")
    print("="*80)
    
    # Load results
    print(f"\nLoading results...")
    print(f"  INT8: {int8_file}")
    print(f"  FP8:  {fp8_file}")
    
    int8_data = load_result(int8_file)
    fp8_data = load_result(fp8_file)
    
    # Analyze each
    analyze_reproducibility(int8_data, "INT8")
    analyze_reproducibility(fp8_data, "FP8")
    
    # Compare
    compare_results(int8_data, fp8_data)
    
    # Forensic implications
    forensic_implications(int8_data, fp8_data)
    
    print()
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()