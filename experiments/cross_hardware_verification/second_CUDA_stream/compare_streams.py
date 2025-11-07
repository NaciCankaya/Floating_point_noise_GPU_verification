#!/usr/bin/env python3
"""
Cross-Hardware CUDA Stream Comparison and Analysis

Analyzes results from two hardware runs testing parallel CUDA stream effects.
Compares across multiple dimensions:
1. Same hardware, different stream conditions (baseline vs light vs heavy)
2. Different hardware, same stream condition
3. Different hardware, different stream conditions

Usage:
    From command line:
        python compare_streams.py a100_cuda_stream_TIMESTAMP.json h100_cuda_stream_TIMESTAMP.json

    In Jupyter/IPython (auto-detects most recent files):
        %run compare_streams.py
"""

import json
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import sys

def load_results(filepath):
    """Load experimental results from JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)

def compute_l2_distance(key_vectors1, key_vectors2):
    """
    Compute L2 distance between two key vector tensors.
    Both should be shape (num_sampled_positions, key_dim)
    """
    arr1 = np.array(key_vectors1)
    arr2 = np.array(key_vectors2)

    if arr1.shape != arr2.shape:
        raise ValueError(f"Shape mismatch: {arr1.shape} vs {arr2.shape}")

    # Compute L2 per sampled position
    l2_per_position = np.linalg.norm(arr1 - arr2, axis=1)

    # Return aggregate L2 (norm across all positions and dimensions)
    return float(np.linalg.norm(arr1 - arr2))

def compare_conditions(seq1_conditions, seq2_conditions, layer_name, seq_name,
                      hw1_label, hw2_label, condition1, condition2):
    """
    Compare key vectors between two conditions (may be same or different hardware).

    Returns: L2 distance and metadata
    """
    # Get the specific condition data
    cond1_data = seq1_conditions.get(condition1)
    cond2_data = seq2_conditions.get(condition2)

    if cond1_data is None or cond2_data is None:
        return None

    # Get key vectors for this layer
    keys1 = cond1_data['layers'][layer_name]['key_vectors']
    keys2 = cond2_data['layers'][layer_name]['key_vectors']

    # Compute L2 distance
    l2 = compute_l2_distance(keys1, keys2)

    return {
        'l2_distance': l2,
        'hw1': hw1_label,
        'hw2': hw2_label,
        'condition1': condition1,
        'condition2': condition2,
        'sequence': seq_name,
        'reproducible1': cond1_data.get('reproducible', True),
        'reproducible2': cond2_data.get('reproducible', True)
    }

def main(file1, file2):
    print("="*70)
    print("CROSS-HARDWARE CUDA STREAM ANALYSIS")
    print("="*70)
    print()

    # Load results
    print(f"Loading: {file1}")
    results1 = load_results(file1)
    print(f"Loading: {file2}")
    results2 = load_results(file2)
    print()

    # Hardware info
    hw1 = results1['metadata']['hardware']
    hw2 = results2['metadata']['hardware']
    print(f"Hardware 1: {hw1} ({results1['metadata']['gpu']})")
    print(f"Hardware 2: {hw2} ({results2['metadata']['gpu']})")
    print()

    # Get common sequences
    sequences1 = set(results1['sequences'].keys())
    sequences2 = set(results2['sequences'].keys())
    common_sequences = sorted(sequences1 & sequences2)

    if len(common_sequences) == 0:
        print("ERROR: No common sequences found!")
        sys.exit(1)

    print(f"Common sequences: {len(common_sequences)}")
    print()

    # Get available conditions
    first_seq = common_sequences[0]
    conditions = sorted(results1['sequences'][first_seq]['conditions'].keys())
    print(f"Conditions tested: {conditions}")
    print()

    # Get layer names
    layer_names = list(results1['sequences'][first_seq]['conditions']['baseline']['layers'].keys())
    print(f"Layers to analyze: {layer_names}")
    print()

    # ========================================================================
    # ANALYSIS 1: Within-hardware reproducibility check
    # ========================================================================
    print("="*70)
    print("ANALYSIS 1: REPRODUCIBILITY CHECK")
    print("="*70)
    print()

    reproducibility_summary = {}
    for hw_label, hw_results in [(hw1, results1), (hw2, results2)]:
        print(f"--- {hw_label.upper()} ---")
        hw_summary = {}

        for condition in conditions:
            reproducible_count = 0
            total_count = 0

            for seq_name in common_sequences:
                seq_data = hw_results['sequences'][seq_name]
                cond_data = seq_data['conditions'][condition]
                if cond_data['reproducible']:
                    reproducible_count += 1
                total_count += 1

            hw_summary[condition] = {
                'reproducible': reproducible_count,
                'total': total_count,
                'percentage': 100 * reproducible_count / total_count if total_count > 0 else 0
            }

            symbol = "✓" if reproducible_count == total_count else "⚠"
            print(f"  {condition}: {symbol} {reproducible_count}/{total_count} sequences reproducible "
                  f"({hw_summary[condition]['percentage']:.0f}%)")

        reproducibility_summary[hw_label] = hw_summary
        print()

    # ========================================================================
    # ANALYSIS 2: Cross-hardware, same condition
    # ========================================================================
    print("="*70)
    print("ANALYSIS 2: CROSS-HARDWARE (SAME CONDITION)")
    print("="*70)
    print()
    print("Comparing same condition across different hardware")
    print()

    cross_hardware_results = {}

    for condition in conditions:
        print(f"--- {condition.upper()} ---")
        condition_results = {}

        for layer_name in layer_names:
            l2_distances = []

            for seq_name in common_sequences:
                seq1 = results1['sequences'][seq_name]
                seq2 = results2['sequences'][seq_name]

                comp = compare_conditions(
                    seq1['conditions'], seq2['conditions'],
                    layer_name, seq_name,
                    hw1, hw2, condition, condition
                )

                if comp:
                    l2_distances.append(comp['l2_distance'])

            if len(l2_distances) > 0:
                condition_results[layer_name] = {
                    'mean': float(np.mean(l2_distances)),
                    'std': float(np.std(l2_distances)),
                    'median': float(np.median(l2_distances)),
                    'min': float(np.min(l2_distances)),
                    'max': float(np.max(l2_distances)),
                    'num_sequences': len(l2_distances)
                }

                stats = condition_results[layer_name]
                print(f"  {layer_name}: L2 = {stats['mean']:.6f} ± {stats['std']:.6f} "
                      f"(range: [{stats['min']:.6f}, {stats['max']:.6f}])")

        cross_hardware_results[condition] = condition_results
        print()

    # ========================================================================
    # ANALYSIS 3: Same hardware, different conditions
    # ========================================================================
    print("="*70)
    print("ANALYSIS 3: WITHIN-HARDWARE (DIFFERENT CONDITIONS)")
    print("="*70)
    print()
    print("Comparing baseline vs parallel streams on same hardware")
    print()

    within_hardware_results = {}

    # Define condition pairs to compare
    condition_pairs = [
        ('baseline', 'light_concurrent'),
        ('baseline', 'heavy_concurrent'),
        ('light_concurrent', 'heavy_concurrent')
    ]

    for hw_label, hw_results in [(hw1, results1), (hw2, results2)]:
        print(f"--- {hw_label.upper()} ---")
        hw_comparison = {}

        for cond1, cond2 in condition_pairs:
            print(f"  {cond1} vs {cond2}:")
            pair_results = {}

            for layer_name in layer_names:
                l2_distances = []

                for seq_name in common_sequences:
                    seq = hw_results['sequences'][seq_name]

                    comp = compare_conditions(
                        seq['conditions'], seq['conditions'],
                        layer_name, seq_name,
                        hw_label, hw_label, cond1, cond2
                    )

                    if comp:
                        l2_distances.append(comp['l2_distance'])

                if len(l2_distances) > 0:
                    pair_results[layer_name] = {
                        'mean': float(np.mean(l2_distances)),
                        'std': float(np.std(l2_distances)),
                        'median': float(np.median(l2_distances)),
                        'min': float(np.min(l2_distances)),
                        'max': float(np.max(l2_distances))
                    }

                    stats = pair_results[layer_name]
                    print(f"    {layer_name}: L2 = {stats['mean']:.6f} ± {stats['std']:.6f}")

            hw_comparison[f"{cond1}_vs_{cond2}"] = pair_results
            print()

        within_hardware_results[hw_label] = hw_comparison

    # ========================================================================
    # ANALYSIS 4: Cross-hardware AND different conditions (full matrix)
    # ========================================================================
    print("="*70)
    print("ANALYSIS 4: CROSS-HARDWARE + DIFFERENT CONDITIONS")
    print("="*70)
    print()
    print("Full comparison matrix")
    print()

    cross_matrix_results = {}

    for cond1 in conditions:
        for cond2 in conditions:
            if cond1 == cond2:
                continue  # Already covered in Analysis 2

            pair_key = f"{hw1}_{cond1}_vs_{hw2}_{cond2}"
            print(f"--- {hw1.upper()}/{cond1} vs {hw2.upper()}/{cond2} ---")

            pair_results = {}

            for layer_name in layer_names:
                l2_distances = []

                for seq_name in common_sequences:
                    seq1 = results1['sequences'][seq_name]
                    seq2 = results2['sequences'][seq_name]

                    comp = compare_conditions(
                        seq1['conditions'], seq2['conditions'],
                        layer_name, seq_name,
                        hw1, hw2, cond1, cond2
                    )

                    if comp:
                        l2_distances.append(comp['l2_distance'])

                if len(l2_distances) > 0:
                    pair_results[layer_name] = {
                        'mean': float(np.mean(l2_distances)),
                        'std': float(np.std(l2_distances)),
                        'median': float(np.median(l2_distances)),
                        'min': float(np.min(l2_distances)),
                        'max': float(np.max(l2_distances))
                    }

                    stats = pair_results[layer_name]
                    print(f"  {layer_name}: L2 = {stats['mean']:.6f} ± {stats['std']:.6f}")

            cross_matrix_results[pair_key] = pair_results
            print()

    # ========================================================================
    # SUMMARY AND DETECTION THRESHOLDS
    # ========================================================================
    print("="*70)
    print("SUMMARY: DETECTION ANALYSIS")
    print("="*70)
    print()

    # Get final layer stats for key comparisons
    final_layer = layer_names[-1]

    # Baseline cross-hardware (legitimate variation)
    baseline_cross_hw = cross_hardware_results['baseline'][final_layer]
    baseline_threshold = baseline_cross_hw['mean']

    print(f"Using {final_layer} for detection thresholds")
    print()
    print(f"1. BASELINE CROSS-HARDWARE VARIATION (legitimate):")
    print(f"   L2 distance: {baseline_cross_hw['mean']:.4f} ± {baseline_cross_hw['std']:.4f}")
    print(f"   95% upper bound: {baseline_cross_hw['mean'] + 2*baseline_cross_hw['std']:.4f}")
    print()

    # Check if parallel streams are detectable
    print(f"2. PARALLEL STREAM DETECTION:")
    print()

    detection_results = {}

    for hw_label in [hw1, hw2]:
        hw_within = within_hardware_results[hw_label]

        print(f"   {hw_label.upper()}:")

        for comparison, comp_data in hw_within.items():
            if final_layer in comp_data:
                stats = comp_data[final_layer]
                ratio = stats['mean'] / baseline_threshold

                detectable = ratio > 2.0  # More than 2x baseline
                symbol = "✓" if detectable else "✗"

                print(f"     {comparison}: L2 = {stats['mean']:.4f} "
                      f"({ratio:.1f}× baseline) {symbol}")

                detection_results[f"{hw_label}_{comparison}"] = {
                    'l2': stats['mean'],
                    'ratio_to_baseline': float(ratio),
                    'detectable': detectable
                }
        print()

    # Cross-hardware + different conditions
    print(f"3. CROSS-HARDWARE + PARALLEL STREAMS:")
    print()

    for pair_key, pair_data in cross_matrix_results.items():
        if final_layer in pair_data:
            stats = pair_data[final_layer]
            ratio = stats['mean'] / baseline_threshold

            detectable = ratio > 2.0
            symbol = "✓" if detectable else "✗"

            print(f"   {pair_key}:")
            print(f"     L2 = {stats['mean']:.4f} ({ratio:.1f}× baseline) {symbol}")

            detection_results[pair_key] = {
                'l2': stats['mean'],
                'ratio_to_baseline': float(ratio),
                'detectable': detectable
            }

    print()
    print("="*70)
    print("FORENSIC VERDICT")
    print("="*70)
    print()

    # Count detectable cases
    total_parallel_tests = sum(1 for k in detection_results if 'concurrent' in k)
    detectable_parallel = sum(1 for k, v in detection_results.items()
                             if 'concurrent' in k and v['detectable'])

    if detectable_parallel == total_parallel_tests:
        print("✓ PARALLEL STREAMS ARE DETECTABLE")
        print(f"  All {total_parallel_tests}/{total_parallel_tests} parallel stream scenarios detected")
        print(f"  FP forensics successfully identifies concurrent workloads")
        print(f"  Hardware matching NOT required for detection")
    elif detectable_parallel > 0:
        print(f"⚠ PARTIAL DETECTION")
        print(f"  {detectable_parallel}/{total_parallel_tests} parallel stream scenarios detected")
        print(f"  Some concurrent workloads may evade detection")
    else:
        print(f"✗ DETECTION FAILED")
        print(f"  Parallel streams not distinguishable from hardware variation")
        print(f"  FP forensics insufficient for this threat model")

    # ========================================================================
    # Save comprehensive results
    # ========================================================================
    output_dir = Path(file1).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"stream_analysis_{hw1}_vs_{hw2}_{timestamp}.json"

    analysis_results = {
        'metadata': {
            'hardware1': hw1,
            'hardware2': hw2,
            'gpu1': results1['metadata']['gpu'],
            'gpu2': results2['metadata']['gpu'],
            'num_sequences': len(common_sequences),
            'conditions': conditions,
            'layers': layer_names,
            'timestamp': datetime.now().isoformat()
        },
        'reproducibility_analysis': reproducibility_summary,
        'cross_hardware_same_condition': cross_hardware_results,
        'within_hardware_different_conditions': within_hardware_results,
        'cross_hardware_different_conditions': cross_matrix_results,
        'detection_analysis': {
            'baseline_threshold': baseline_threshold,
            'detection_results': detection_results,
            'verdict': {
                'total_parallel_tests': total_parallel_tests,
                'detectable_count': detectable_parallel,
                'detection_rate': detectable_parallel / total_parallel_tests if total_parallel_tests > 0 else 0
            }
        }
    }

    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)

    file_size_kb = output_file.stat().st_size / 1024

    print()
    print("="*70)
    print(f"✓ Analysis saved to: {output_file}")
    print(f"  File size: {file_size_kb:.1f} KB")
    print("="*70)

if __name__ == "__main__":
    # Detect if running in notebook or command-line
    try:
        get_ipython()
        in_notebook = True
    except NameError:
        in_notebook = False

    if in_notebook:
        # Running in notebook - auto-find most recent stream files
        import glob
        import os

        # Check both /workspace/experiments/cross_hardware_verification/second_CUDA_stream and current directory
        search_dirs = [
            '/workspace/experiments/cross_hardware_verification/second_CUDA_stream',
            os.getcwd()
        ]
        stream_files = []

        for dir_path in search_dirs:
            if os.path.exists(dir_path):
                stream_files.extend(glob.glob(f'{dir_path}/*_cuda_stream_*.json'))

        if len(stream_files) < 2:
            print("ERROR: Need at least 2 cuda_stream files to compare")
            print(f"Found {len(stream_files)} cuda_stream files")
            for f in stream_files:
                print(f"  - {f}")
            print("\nRun collect_keys.ipynb on two different hardware types first!")
            sys.exit(1)

        # Sort by modification time (most recent first)
        stream_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)

        # Use two most recent files
        file1 = stream_files[0]
        file2 = stream_files[1]

        print("Auto-detected cuda_stream files (2 most recent):")
        print(f"  File 1: {Path(file1).name}")
        print(f"  File 2: {Path(file2).name}")
        print()
        print("(To compare specific files, run from command line with file arguments)")
        print()

        main(file1, file2)
    else:
        # Running from command line - use argparse
        parser = argparse.ArgumentParser(description='Analyze cross-hardware CUDA stream results')
        parser.add_argument('file1', type=str, help='First cuda_stream results file')
        parser.add_argument('file2', type=str, help='Second cuda_stream results file')
        args = parser.parse_args()

        main(args.file1, args.file2)
