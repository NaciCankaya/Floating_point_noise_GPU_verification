#!/usr/bin/env python3
"""
Cross-Hardware Baseline Comparison and Analysis

Analyzes results from two hardware runs to establish deviation baseline.
Works with sparse key vector data (~120 sampled positions per sequence).

Key analyses:
1. Per-position L2 distances (using sampled positions)
2. Convergence analysis (3, 5, 7, 10 sample means)
3. Position-dependent effects (early vs late in sequence)
4. Variance across examples

Usage:
    In Jupyter/IPython (auto-detects most recent files):
        %run compare_baseline.py
    
    From command line:
        python compare_baseline.py h100_baseline_TIMESTAMP.json a100_baseline_TIMESTAMP.json
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
    # axis=1 means compute norm across key_dim for each position
    l2_per_position = np.linalg.norm(arr1 - arr2, axis=1)
    
    return l2_per_position

def convergence_analysis(distances_by_seq, layer_name):
    """
    Analyze how mean distance converges as more examples are added.
    
    distances_by_seq: list of arrays, each is L2 distances per position for one sequence
    """
    n_sequences = len(distances_by_seq)
    
    # For each subsequence length (3, 5, 7, 10)
    subsequence_lengths = [3, 5, 7, 10] if n_sequences >= 10 else list(range(3, n_sequences + 1))
    
    convergence_results = {}
    
    for n in subsequence_lengths:
        if n > n_sequences:
            continue
        
        # Take first n sequences
        subset = distances_by_seq[:n]
        
        # Concatenate all token positions from these sequences
        all_distances = np.concatenate(subset)
        
        convergence_results[f'n_{n}'] = {
            'mean': float(np.mean(all_distances)),
            'std': float(np.std(all_distances)),
            'median': float(np.median(all_distances)),
            'num_tokens': int(len(all_distances))
        }
    
    return convergence_results

def position_analysis(distances_by_seq, sampled_positions_by_seq, seq_lengths):
    """
    Analyze if deviation varies by position in sequence.
    
    Since we sample every 5th token starting from END going backwards,
    we'll look at:
    - Quartile 4 (latest tokens, 75-100% into sequence)
    - Quartile 3 (50-75%)
    - Quartile 2 (25-50%)
    - Quartile 1 (earliest tokens, 0-25%)
    """
    quartile_stats = {
        'q1': [],  # 0-25% (earliest in sequence)
        'q2': [],  # 25-50%
        'q3': [],  # 50-75%
        'q4': []   # 75-100% (latest in sequence)
    }
    
    for distances, positions, seq_len in zip(distances_by_seq, sampled_positions_by_seq, seq_lengths):
        # positions are indices into the original sequence
        # Assign each distance to a quartile based on its position
        for dist, pos in zip(distances, positions):
            relative_pos = pos / seq_len  # 0.0 to 1.0
            
            if relative_pos < 0.25:
                quartile_stats['q1'].append(dist)
            elif relative_pos < 0.5:
                quartile_stats['q2'].append(dist)
            elif relative_pos < 0.75:
                quartile_stats['q3'].append(dist)
            else:
                quartile_stats['q4'].append(dist)
    
    # Compute statistics for each quartile
    results = {}
    for quartile, values in quartile_stats.items():
        if len(values) > 0:
            results[quartile] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'num_samples': len(values)
            }
    
    return results

def main(file1, file2):
    print("="*70)
    print("CROSS-HARDWARE BASELINE ANALYSIS")
    print("="*70)
    print()
    
    # Load results
    print(f"Loading: {file1}")
    results1 = load_results(file1)
    print(f"Loading: {file2}")
    results2 = load_results(file2)
    print()
    
    # Verify compatibility
    hw1 = results1['metadata']['hardware']
    hw2 = results2['metadata']['hardware']
    print(f"Hardware 1: {hw1} ({results1['metadata']['gpu']})")
    print(f"Hardware 2: {hw2} ({results2['metadata']['gpu']})")
    print()
    
    sequences1 = set(results1['sequences'].keys())
    sequences2 = set(results2['sequences'].keys())
    common_sequences = sequences1 & sequences2
    
    if len(common_sequences) == 0:
        print("ERROR: No common sequences found!")
        sys.exit(1)
    
    print(f"Common sequences: {len(common_sequences)}")
    
    # Print sequence summary with prompt lengths
    print()
    print("Sequence Summary:")
    first_seq_name = list(common_sequences)[0]
    for seq_name in sorted(common_sequences):
        seq_data = results1['sequences'][seq_name]
        prompt_len = seq_data.get('prompt_length_tokens', seq_data['sequence_length'])
        num_sampled = seq_data.get('num_sampled_positions', len(seq_data.get('sampled_positions', [])))
        print(f"  {seq_name}: {prompt_len} tokens → {num_sampled} sampled positions")
    print()
    
    # Get layer names
    first_seq = list(common_sequences)[0]
    layer_names = list(results1['sequences'][first_seq]['layers'].keys())
    
    print(f"Layers to analyze: {layer_names}")
    print()
    
    # Main analysis loop
    analysis_results = {
        'metadata': {
            'hardware1': hw1,
            'hardware2': hw2,
            'gpu1': results1['metadata']['gpu'],
            'gpu2': results2['metadata']['gpu'],
            'num_sequences': len(common_sequences),
            'timestamp': datetime.now().isoformat()
        },
        'per_layer_analysis': {}
    }
    
    for layer_name in layer_names:
        print("="*70)
        print(f"LAYER: {layer_name}")
        print("="*70)
        
        distances_by_seq = []
        sampled_positions_by_seq = []
        seq_lengths = []
        per_sequence_stats = {}
        
        # Compute distances for each sequence
        for seq_name in sorted(common_sequences):
            seq1 = results1['sequences'][seq_name]
            seq2 = results2['sequences'][seq_name]
            
            # Verify sequence lengths and sampling match
            if seq1['sequence_length'] != seq2['sequence_length']:
                print(f"  WARNING: Length mismatch for {seq_name}, skipping")
                continue
            
            if seq1['sampled_positions'] != seq2['sampled_positions']:
                print(f"  WARNING: Sampled positions differ for {seq_name}, skipping")
                continue
            
            seq_len = seq1['sequence_length']
            prompt_length = seq1.get('prompt_length_tokens', seq_len)  # Fallback if old format
            sampled_positions = seq1['sampled_positions']
            seq_lengths.append(seq_len)
            sampled_positions_by_seq.append(sampled_positions)
            
            # Get key vectors
            keys1 = seq1['layers'][layer_name]['key_vectors']
            keys2 = seq2['layers'][layer_name]['key_vectors']
            
            # Compute L2 per sampled position
            l2_per_position = compute_l2_distance(keys1, keys2)
            distances_by_seq.append(l2_per_position)
            
            # Per-sequence statistics
            per_sequence_stats[seq_name] = {
                'prompt_length_tokens': prompt_length,
                'sequence_length': seq_len,
                'num_sampled': len(sampled_positions),
                'mean_l2': float(np.mean(l2_per_position)),
                'std_l2': float(np.std(l2_per_position)),
                'median_l2': float(np.median(l2_per_position)),
                'min_l2': float(np.min(l2_per_position)),
                'max_l2': float(np.max(l2_per_position))
            }
        
        # Aggregate statistics (all sampled positions from all sequences)
        all_distances = np.concatenate(distances_by_seq)
        total_samples = len(all_distances)
        
        aggregate_stats = {
            'total_samples': int(total_samples),
            'mean': float(np.mean(all_distances)),
            'std': float(np.std(all_distances)),
            'median': float(np.median(all_distances)),
            'min': float(np.min(all_distances)),
            'max': float(np.max(all_distances)),
            'percentiles': {
                'p05': float(np.percentile(all_distances, 5)),
                'p25': float(np.percentile(all_distances, 25)),
                'p75': float(np.percentile(all_distances, 75)),
                'p95': float(np.percentile(all_distances, 95)),
                'p99': float(np.percentile(all_distances, 99))
            }
        }
        
        # Convergence analysis
        convergence = convergence_analysis(distances_by_seq, layer_name)
        
        # Position analysis
        position_stats = position_analysis(distances_by_seq, sampled_positions_by_seq, seq_lengths)
        
        # Variance across examples (using per-sequence means)
        sequence_means = [per_sequence_stats[s]['mean_l2'] for s in sorted(common_sequences) if s in per_sequence_stats]
        cross_sequence_variance = {
            'mean_of_means': float(np.mean(sequence_means)),
            'std_of_means': float(np.std(sequence_means)),
            'min_mean': float(np.min(sequence_means)),
            'max_mean': float(np.max(sequence_means)),
            'coefficient_of_variation': float(np.std(sequence_means) / np.mean(sequence_means))
        }
        
        # Store all results for this layer
        analysis_results['per_layer_analysis'][layer_name] = {
            'aggregate_statistics': aggregate_stats,
            'convergence_analysis': convergence,
            'position_analysis': position_stats,
            'cross_sequence_variance': cross_sequence_variance,
            'per_sequence_statistics': per_sequence_stats
        }
        
        # Print summary
        print(f"\nAggregate Statistics (all {total_samples} sampled positions):")
        print(f"  Mean L2:    {aggregate_stats['mean']:.6f} ± {aggregate_stats['std']:.6f}")
        print(f"  Median L2:  {aggregate_stats['median']:.6f}")
        print(f"  Range:      [{aggregate_stats['min']:.6f}, {aggregate_stats['max']:.6f}]")
        print(f"  95% CI:     [{aggregate_stats['percentiles']['p05']:.6f}, {aggregate_stats['percentiles']['p95']:.6f}]")
        
        print(f"\nConvergence Analysis:")
        for n_key in sorted(convergence.keys()):
            conv = convergence[n_key]
            n = int(n_key.split('_')[1])
            print(f"  {n} sequences ({conv['num_tokens']} samples): "
                  f"mean={conv['mean']:.6f} ± {conv['std']:.6f}")
        
        # Check convergence quality
        if 'n_10' in convergence and 'n_7' in convergence:
            mean_diff = abs(convergence['n_10']['mean'] - convergence['n_7']['mean'])
            relative_change = mean_diff / convergence['n_10']['mean']
            print(f"\n  Convergence quality (7→10 sequences):")
            print(f"    Absolute change: {mean_diff:.6f}")
            print(f"    Relative change: {relative_change*100:.2f}%")
            if relative_change < 0.05:
                print(f"    ✓ Good convergence (<5% change)")
            else:
                print(f"    ⚠ May need more sequences (>{5}% change)")
        
        print(f"\nPosition Analysis (quartiles):")
        for q in ['q1', 'q2', 'q3', 'q4']:
            if q in position_stats:
                q_stats = position_stats[q]
                q_label = {
                    'q1': '0-25% (early)',
                    'q2': '25-50%',
                    'q3': '50-75%',
                    'q4': '75-100% (late)'
                }[q]
                print(f"  {q.upper()} {q_label}: mean={q_stats['mean']:.6f} ± {q_stats['std']:.6f} "
                      f"({q_stats['num_samples']} samples)")
        
        # Check if position matters
        if all(q in position_stats for q in ['q1', 'q2', 'q3', 'q4']):
            q_means = [position_stats[q]['mean'] for q in ['q1', 'q2', 'q3', 'q4']]
            max_diff = max(q_means) - min(q_means)
            relative_diff = max_diff / np.mean(q_means)
            print(f"\n  Position effect:")
            print(f"    Max quartile difference: {max_diff:.6f}")
            print(f"    Relative difference: {relative_diff*100:.2f}%")
            if relative_diff < 0.1:
                print(f"    ✓ Position-independent (<10% variation)")
            else:
                print(f"    ⚠ Position-dependent (>{10}% variation)")
        
        print(f"\nCross-Sequence Variance:")
        print(f"  Mean of per-sequence means: {cross_sequence_variance['mean_of_means']:.6f}")
        print(f"  Std of per-sequence means:  {cross_sequence_variance['std_of_means']:.6f}")
        print(f"  Coefficient of variation:   {cross_sequence_variance['coefficient_of_variation']:.4f}")
        print(f"  Range of means: [{cross_sequence_variance['min_mean']:.6f}, "
              f"{cross_sequence_variance['max_mean']:.6f}]")
        
        if cross_sequence_variance['coefficient_of_variation'] < 0.2:
            print(f"  ✓ Low variance across examples (CV < 0.2)")
        else:
            print(f"  ⚠ High variance across examples (CV > 0.2)")
        
        print()
    
    # Save comprehensive results
    output_dir = Path(file1).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"baseline_analysis_{hw1}_vs_{hw2}_{timestamp}.json"
    
    # Generate console summary text for JSON
    console_summary = []
    console_summary.append("="*70)
    console_summary.append("CROSS-HARDWARE BASELINE ANALYSIS SUMMARY")
    console_summary.append("="*70)
    console_summary.append(f"Hardware 1: {hw1} ({results1['metadata']['gpu']})")
    console_summary.append(f"Hardware 2: {hw2} ({results2['metadata']['gpu']})")
    console_summary.append(f"Sequences analyzed: {len(common_sequences)}")
    console_summary.append(f"Layers analyzed: {layer_names}")
    console_summary.append("")
    
    # Add layer summaries
    for layer_name in layer_names:
        layer_stats = analysis_results['per_layer_analysis'][layer_name]
        agg = layer_stats['aggregate_statistics']
        conv = layer_stats['convergence_analysis']
        var = layer_stats['cross_sequence_variance']
        
        console_summary.append(f"--- {layer_name.upper()} ---")
        console_summary.append(f"  Mean L2: {agg['mean']:.6f} ± {agg['std']:.6f}")
        console_summary.append(f"  Median:  {agg['median']:.6f}")
        console_summary.append(f"  95% CI:  [{agg['percentiles']['p05']:.6f}, {agg['percentiles']['p95']:.6f}]")
        
        # Convergence verdict
        if 'n_10' in conv and 'n_7' in conv:
            rel_change = abs(conv['n_10']['mean'] - conv['n_7']['mean']) / conv['n_10']['mean']
            if rel_change < 0.05:
                console_summary.append(f"  Convergence: ✓ Good ({rel_change*100:.1f}% change)")
            else:
                console_summary.append(f"  Convergence: ⚠ Marginal ({rel_change*100:.1f}% change)")
        
        # Variance verdict
        if var['coefficient_of_variation'] < 0.2:
            console_summary.append(f"  Variance: ✓ Low (CV={var['coefficient_of_variation']:.4f})")
        else:
            console_summary.append(f"  Variance: ⚠ High (CV={var['coefficient_of_variation']:.4f})")
        
        console_summary.append("")
    
    # Final recommendations
    last_layer = layer_names[-1]
    last_stats = analysis_results['per_layer_analysis'][last_layer]['aggregate_statistics']
    threshold = last_stats['mean'] + 2 * last_stats['std']
    
    console_summary.append("="*70)
    console_summary.append("RECOMMENDED DETECTION THRESHOLD")
    console_summary.append("="*70)
    console_summary.append(f"Final layer ({last_layer}):")
    console_summary.append(f"  Baseline mean: {last_stats['mean']:.4f}")
    console_summary.append(f"  Baseline std:  {last_stats['std']:.4f}")
    console_summary.append(f"  Threshold (mean + 2σ): {threshold:.4f}")
    console_summary.append(f"  95th percentile: {last_stats['percentiles']['p95']:.4f}")
    console_summary.append("")
    console_summary.append("Detection rule:")
    console_summary.append(f"  L2 < {threshold:.2f} → Legitimate cross-hardware variation")
    console_summary.append(f"  L2 > {threshold:.2f} → Suspicious (investigate)")
    console_summary.append("")
    
    # Add to results
    analysis_results['console_summary'] = "\n".join(console_summary)
    
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    file_size_kb = output_file.stat().st_size / 1024
    
    print("="*70)
    print(f"✓ Analysis saved to: {output_file}")
    print(f"  File size: {file_size_kb:.1f} KB")
    print(f"  (JSON includes this console summary at the top for easy reference)")
    print("="*70)
    print()
    
    # Final summary
    print("SUMMARY RECOMMENDATIONS:")
    print()
    
    last_layer = layer_names[-1]
    last_stats = analysis_results['per_layer_analysis'][last_layer]['aggregate_statistics']
    last_conv = analysis_results['per_layer_analysis'][last_layer]['convergence_analysis']
    
    mean_l2 = last_stats['mean']
    std_l2 = last_stats['std']
    p95 = last_stats['percentiles']['p95']
    
    print(f"For {last_layer} (final layer):")
    print(f"  Baseline deviation: {mean_l2:.4f} ± {std_l2:.4f}")
    print(f"  95th percentile: {p95:.4f}")
    print(f"  Recommended threshold (mean + 2σ): {mean_l2 + 2*std_l2:.4f}")
    print()
    
    # Check if 10 sequences was enough
    if 'n_10' in last_conv and 'n_7' in last_conv:
        relative_change = abs(last_conv['n_10']['mean'] - last_conv['n_7']['mean']) / last_conv['n_10']['mean']
        if relative_change < 0.05:
            print(f"✓ 10 sequences provides good convergence")
            print(f"  Can proceed with batch size mismatch experiments")
        else:
            print(f"⚠ Consider collecting more sequences for better statistics")
            print(f"  Convergence not fully stable (7→10: {relative_change*100:.1f}% change)")

if __name__ == "__main__":
    # Detect if running in notebook or command-line
    try:
        get_ipython()
        in_notebook = True
    except NameError:
        in_notebook = False
    
    if in_notebook:
        # Running in notebook - auto-find most recent baseline files
        import glob
        import os
        
        # Check both /workspace/experiments and current directory
        search_dirs = ['/workspace/experiments', os.getcwd()]
        baseline_files = []
        
        for dir_path in search_dirs:
            if os.path.exists(dir_path):
                baseline_files.extend(glob.glob(f'{dir_path}/*_baseline_*.json'))
        
        if len(baseline_files) < 2:
            print("ERROR: Need at least 2 baseline files to compare")
            print(f"Found {len(baseline_files)} baseline files")
            for f in baseline_files:
                print(f"  - {f}")
            print("\nRun cross_hardware_baseline.py on two different hardware types first!")
            sys.exit(1)
        
        # Sort by modification time (most recent first)
        baseline_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
        
        # Use two most recent files
        file1 = baseline_files[0]
        file2 = baseline_files[1]
        
        print("Auto-detected baseline files (2 most recent):")
        print(f"  File 1: {Path(file1).name}")
        print(f"  File 2: {Path(file2).name}")
        print()
        print("(To compare specific files, run from command line with file arguments)")
        print()
        
        main(file1, file2)
    else:
        # Running from command line - use argparse
        parser = argparse.ArgumentParser(description='Analyze cross-hardware baseline')
        parser.add_argument('file1', type=str, help='First baseline results file')
        parser.add_argument('file2', type=str, help='Second baseline results file')
        args = parser.parse_args()
        
        main(args.file1, args.file2)