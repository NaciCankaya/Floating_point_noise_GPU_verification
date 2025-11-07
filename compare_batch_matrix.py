#!/usr/bin/env python3
"""
Compare Batch Matrix Experiments

Compares two batch_matrix experiment files, showing all pairwise comparisons.

Key comparisons:
1. Same hardware, diff batch size (should MATCH)
2. Cross-hardware, same batch size (baseline - should MATCH)
3. Cross-hardware, diff batch size (key test - MATCH = undetectable evasion)

Usage:
    python compare_batch_matrix.py <file1.json> <file2.json>
    
    # Example:
    python compare_batch_matrix.py h100_batch_matrix_*.json a100_batch_matrix_*.json
"""

import json
import sys
import numpy as np
from pathlib import Path

# ============================================================================
# COMPARISON LOGIC
# ============================================================================

def compute_l2_distance(vec1, vec2):
    """Compute L2 distance between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.linalg.norm(v1 - v2))

def compare_measurements(m1, m2, layer_indices):
    """
    Compare two measurements across all layers.
    
    Returns:
        dict: {
            'match': bool,
            'layer_distances': {layer_name: distance},
            'max_distance': float,
            'max_layer': str,
            'avg_distance': float
        }
    """
    layer_distances = {}
    
    for layer_idx in layer_indices:
        layer_name = f'layer_{layer_idx}'
        
        vec1 = m1['layers'][layer_name]['key_vector']
        vec2 = m2['layers'][layer_name]['key_vector']
        
        distance = compute_l2_distance(vec1, vec2)
        layer_distances[layer_name] = distance
    
    distances = list(layer_distances.values())
    max_distance = max(distances)
    max_layer = max(layer_distances.keys(), key=lambda k: layer_distances[k])
    avg_distance = sum(distances) / len(distances)
    
    # Match if all distances are exactly 0
    match = max_distance == 0.0
    
    return {
        'match': match,
        'layer_distances': layer_distances,
        'max_distance': max_distance,
        'max_layer': max_layer,
        'avg_distance': avg_distance
    }

# ============================================================================
# MAIN COMPARISON
# ============================================================================

def main(file1=None, file2=None):
    # Support both command line and function call
    if file1 is None or file2 is None:
        if len(sys.argv) != 3:
            print("Usage: python compare_batch_matrix.py <file1.json> <file2.json>")
            print("Or in notebook: main('file1.json', 'file2.json')")
            sys.exit(1)
        file1_path = Path(sys.argv[1])
        file2_path = Path(sys.argv[2])
    else:
        file1_path = Path(file1)
        file2_path = Path(file2)
    
    if not file1_path.exists():
        print(f"Error: {file1_path} not found")
        sys.exit(1)
    
    if not file2_path.exists():
        print(f"Error: {file2_path} not found")
        sys.exit(1)
    
    # Load data
    with open(file1_path) as f:
        data1 = json.load(f)
    
    with open(file2_path) as f:
        data2 = json.load(f)
    
    hw1 = data1['metadata']['hardware']
    hw2 = data2['metadata']['hardware']
    
    print("="*80)
    print("BATCH MATRIX COMPARISON")
    print("="*80)
    print(f"\nFile 1: {file1_path.name}")
    print(f"  Hardware: {hw1.upper()}")
    print(f"  GPU: {data1['metadata']['gpu']}")
    print(f"  Measurements: {len(data1['measurements'])}")
    print()
    print(f"File 2: {file2_path.name}")
    print(f"  Hardware: {hw2.upper()}")
    print(f"  GPU: {data2['metadata']['gpu']}")
    print(f"  Measurements: {len(data2['measurements'])}")
    print()
    
    layer_indices = data1['metadata']['layer_indices']
    
    # Get all measurement names
    names1 = sorted(data1['measurements'].keys())
    names2 = sorted(data2['measurements'].keys())
    
    # Find reference sequences
    refs1 = set(name.rsplit('_bs', 1)[0] for name in names1)
    refs2 = set(name.rsplit('_bs', 1)[0] for name in names2)
    common_refs = sorted(refs1 & refs2)
    
    if not common_refs:
        print("⚠ WARNING: No common reference sequences between files!")
        print(f"  File 1 refs: {refs1}")
        print(f"  File 2 refs: {refs2}")
        print()
    
    # ========================================================================
    # COMPARISON CATEGORIES
    # ========================================================================
    
    print("="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print()
    
    if hw1 == hw2:
        print(f"Same Hardware: {hw1.upper()}")
        print("-" * 80)
        print("Testing: Different batch sizes on same hardware")
        print("Expected: DIFFER (batch size affects activations)")
        print()
        
        same_hw_distances = []
        
        for ref in common_refs:
            print(f"\n{ref}:")
            
            # Get all batch sizes for this ref
            ref_measurements = [(name, data1['measurements'][name]) for name in names1 if name.startswith(ref)]
            ref_measurements.sort(key=lambda x: x[1]['batch_size'])
            
            # Compare all pairs
            for i, (name1, m1) in enumerate(ref_measurements):
                for name2, m2 in ref_measurements[i+1:]:
                    bs1 = m1['batch_size']
                    bs2 = m2['batch_size']
                    
                    result = compare_measurements(m1, m2, layer_indices)
                    distance = result['max_distance']
                    same_hw_distances.append(distance)
                    
                    status = "✓ differ" if distance > 0 else "✗ identical"
                    print(f"  bs{bs1} vs bs{bs2}: Δ={distance:.2e} {status}")
        
        print()
        print("Summary:")
        if same_hw_distances:
            print(f"  Average distance: {np.mean(same_hw_distances):.2e}")
            print(f"  Distance range: [{min(same_hw_distances):.2e}, {max(same_hw_distances):.2e}]")
            differs = sum(1 for d in same_hw_distances if d > 0)
            print(f"  Non-zero differences: {differs}/{len(same_hw_distances)}")
        print()
    else:
        print(f"Cross-Hardware: {hw1.upper()} vs {hw2.upper()}")
        print("-" * 80)
        print()
        
        # Detectability criterion
        DETECTABILITY_RATIO = 1.5  # Key test must be 1.5x larger than baseline
        
        # ====================================================================
        # BASELINE: Same reference, same batch (cross-hardware)
        # ====================================================================
        print("1. BASELINE: Cross-hardware distance (same reference, same batch)")
        print("   Establishes magnitude of hardware-induced differences")
        print()
        
        baseline_distances = {1: [], 2: [], 4: []}  # Organize by batch size
        baseline_per_layer = {f'layer_{idx}': [] for idx in layer_indices}  # Track per-layer
        
        for ref in common_refs:
            for bs in [1, 2, 4]:
                name1 = f"{ref}_bs{bs}"
                name2 = f"{ref}_bs{bs}"
                
                if name1 in data1['measurements'] and name2 in data2['measurements']:
                    m1 = data1['measurements'][name1]
                    m2 = data2['measurements'][name2]
                    
                    result = compare_measurements(m1, m2, layer_indices)
                    baseline_distances[bs].append(result['max_distance'])
                    
                    # Collect per-layer distances
                    for layer_name, dist in result['layer_distances'].items():
                        baseline_per_layer[layer_name].append(dist)
                    
                    print(f"  {ref}_bs{bs}: Δ_max={result['max_distance']:.2e} (at {result['max_layer']})")
        
        print()
        print("  Baseline averages by batch size (using max across layers):")
        baseline_means = {}
        baseline_stds = {}
        for bs in [1, 2, 4]:
            if baseline_distances[bs]:
                mean_dist = np.mean(baseline_distances[bs])
                std_dist = np.std(baseline_distances[bs])
                baseline_means[bs] = mean_dist
                baseline_stds[bs] = std_dist
                print(f"    bs={bs}: μ={mean_dist:.2e}, σ={std_dist:.2e}")
        
        # Overall baseline (average across all batch sizes and refs)
        all_baseline = [d for dists in baseline_distances.values() for d in dists]
        baseline_mean = np.mean(all_baseline) if all_baseline else 0
        baseline_std = np.std(all_baseline) if all_baseline else 0
        print(f"    Overall: μ={baseline_mean:.2e}, σ={baseline_std:.2e}")
        print()
        
        print("  Baseline averages by layer (all batch sizes combined):")
        for layer_name in [f'layer_{idx}' for idx in layer_indices]:
            if baseline_per_layer[layer_name]:
                layer_mean = np.mean(baseline_per_layer[layer_name])
                layer_std = np.std(baseline_per_layer[layer_name])
                print(f"    {layer_name}: μ={layer_mean:.2e}, σ={layer_std:.2e}")
        print()
        
        # ====================================================================
        # KEY TEST: Same reference, different batch (cross-hardware)
        # ====================================================================
        print("2. KEY TEST: Cross-hardware + batch mismatch")
        print("   Tests if batch size mismatch adds detectable signal")
        print(f"   Detectability criterion: distance > {DETECTABILITY_RATIO}× baseline")
        print()
        
        key_test_results = []
        
        for ref in common_refs:
            print(f"  {ref}:")
            
            # Get all measurements for this ref from both files
            measures1 = [(name, data1['measurements'][name]) for name in names1 if name.startswith(ref)]
            measures2 = [(name, data2['measurements'][name]) for name in names2 if name.startswith(ref)]
            
            # Compare all cross-pairs with different batch sizes
            for name1, m1 in measures1:
                for name2, m2 in measures2:
                    bs1 = m1['batch_size']
                    bs2 = m2['batch_size']
                    
                    if bs1 == bs2:
                        continue  # Skip same batch (already in baseline)
                    
                    result = compare_measurements(m1, m2, layer_indices)
                    distance = result['max_distance']
                    
                    # Compare to baseline for this batch size (use smaller bs as reference)
                    ref_bs = min(bs1, bs2)
                    baseline_ref = baseline_means.get(ref_bs, baseline_mean)
                    
                    ratio = distance / baseline_ref if baseline_ref > 0 else float('inf')
                    
                    # Detectability assessment
                    detectable = ratio >= DETECTABILITY_RATIO
                    
                    status_icon = "✓" if detectable else "✗"
                    print(f"    {hw1}_bs{bs1} vs {hw2}_bs{bs2}: Δ={distance:.2e} (at {result['max_layer']}), ratio={ratio:.2f}× {status_icon}")
                    
                    key_test_results.append({
                        'ref': ref,
                        'bs1': bs1,
                        'bs2': bs2,
                        'distance': distance,
                        'baseline': baseline_ref,
                        'ratio': ratio,
                        'detectable': detectable
                    })
            
            print()
        
        # Key test summary
        key_test_distances = [r['distance'] for r in key_test_results]
        key_test_ratios = [r['ratio'] for r in key_test_results]
        detectable_count = sum(1 for r in key_test_results if r['detectable'])
        
        print()
        print(f"  Key test averages:")
        print(f"    Distance: μ={np.mean(key_test_distances):.2e}, σ={np.std(key_test_distances):.2e}")
        print(f"    Ratio to baseline: μ={np.mean(key_test_ratios):.2f}×, σ={np.std(key_test_ratios):.2f}×")
        print(f"    Detectable: {detectable_count}/{len(key_test_results)}")
        print()
    
    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    
    print("="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print()
    
    # Compute all pairwise distances for reference
    all_distances = []
    for name1 in names1:
        for name2 in names2:
            m1 = data1['measurements'][name1]
            m2 = data2['measurements'][name2]
            result = compare_measurements(m1, m2, layer_indices)
            all_distances.append(result['max_distance'])
    
    print(f"Total pairwise comparisons: {len(all_distances)}")
    print(f"Distance statistics:")
    print(f"  Min: {min(all_distances):.2e}")
    print(f"  Max: {max(all_distances):.2e}")
    print(f"  Mean: {np.mean(all_distances):.2e}")
    print(f"  Median: {np.median(all_distances):.2e}")
    print(f"  Std: {np.std(all_distances):.2e}")
    print()
    print("="*80)
    print("CONCLUSION")
    print("="*80)
    print()
    
    if hw1 == hw2:
        differs = sum(1 for d in same_hw_distances if d > 0)
        if differs > 0:
            print(f"✓ Batch sizes produce different outputs on same hardware ({differs}/{len(same_hw_distances)} differ)")
            print(f"  Average distance: {np.mean(same_hw_distances):.2e}")
            print("  (As expected - batch size affects activations)")
        else:
            print("✗ All batch sizes produce identical outputs on same hardware")
            print("  (Unexpected - batch size should affect computation)")
    else:
        # Cross-hardware
        print("BASELINE (Hardware-induced differences):")
        print(f"  Average distance: {baseline_mean:.2e} ± {baseline_std:.2e}")
        print(f"  Range: [{min(all_baseline):.2e}, {max(all_baseline):.2e}]")
        print()
        
        print("KEY TEST (Hardware + Batch mismatch):")
        print(f"  Average distance: {np.mean(key_test_distances):.2e} ± {np.std(key_test_distances):.2e}")
        print(f"  Average ratio to baseline: {np.mean(key_test_ratios):.2f}×")
        print()
        
        if detectable_count > len(key_test_results) * 0.5:
            # Majority detectable
            pct = (detectable_count / len(key_test_results)) * 100
            print(f"✓ BATCH SIZE MISMATCH IS DETECTABLE ({pct:.0f}% of comparisons)")
            print(f"  → {detectable_count}/{len(key_test_results)} comparisons exceed {DETECTABILITY_RATIO}× baseline")
            print("  → Malicious prover cannot hide batch size changes cross-hardware")
            print("  → Forensic verification can detect this type of throughput evasion")
        else:
            # Majority undetectable
            pct = (detectable_count / len(key_test_results)) * 100
            print(f"✗ BATCH SIZE MISMATCH IS NOT RELIABLY DETECTABLE ({pct:.0f}% detectable)")
            print(f"  → Only {detectable_count}/{len(key_test_results)} comparisons exceed {DETECTABILITY_RATIO}× baseline")
            print("  → Batch size signal is comparable to or smaller than hardware noise")
            print("  → Forensic verification cannot reliably detect this type of evasion")
    
    print()

if __name__ == "__main__":
    main()