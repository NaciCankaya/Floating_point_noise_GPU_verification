#!/usr/bin/env python3
"""
Compare logprobs across all runs from vLLM Kimi K2 experiment
Shows which runs are identical vs different
"""

import json
import numpy as np
import sys
from pathlib import Path

def load_experiment_data(json_path):
    """Load experiment data from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def compare_logprobs(data):
    """Compare logprobs across all runs"""
    
    # Extract data
    logprobs_vectors = data['logprobs_vectors']
    token_sequences = data['token_sequences']
    num_runs = len(logprobs_vectors)
    
    print("=" * 80)
    print("LOGPROBS COMPARISON")
    print("=" * 80)
    print()
    print(f"Number of runs: {num_runs}")
    print(f"Tokens per run: {len(logprobs_vectors[0])}")
    print()
    
    # Convert to numpy arrays for easier comparison
    logprobs_arrays = [np.array(lp) for lp in logprobs_vectors]
    
    # Check token sequences first
    print("TOKEN SEQUENCES:")
    print("-" * 80)
    token_groups = {}  # Group runs by their token sequences
    
    for i in range(num_runs):
        seq_tuple = tuple(token_sequences[i])
        if seq_tuple not in token_groups:
            token_groups[seq_tuple] = []
        token_groups[seq_tuple].append(i)
    
    print(f"Found {len(token_groups)} distinct token sequences:")
    for group_id, (seq, runs) in enumerate(token_groups.items(), 1):
        print(f"  Group {group_id}: Runs {runs}")
    print()
    
    # Check logprobs
    print("LOGPROBS COMPARISON:")
    print("-" * 80)
    
    # Build comparison matrix
    comparison_matrix = np.zeros((num_runs, num_runs))
    
    for i in range(num_runs):
        for j in range(num_runs):
            if i == j:
                comparison_matrix[i, j] = 0.0
            else:
                l2_dist = np.linalg.norm(logprobs_arrays[i] - logprobs_arrays[j])
                comparison_matrix[i, j] = l2_dist
    
    # Find groups of identical logprobs (L2 < 1e-10)
    EPSILON = 1e-10
    logprob_groups = {}
    assigned = set()
    
    for i in range(num_runs):
        if i in assigned:
            continue
        
        # Find all runs identical to run i
        group = [i]
        for j in range(i + 1, num_runs):
            if j not in assigned and comparison_matrix[i, j] < EPSILON:
                group.append(j)
                assigned.add(j)
        
        assigned.add(i)
        logprob_groups[i] = group
    
    print(f"Found {len(logprob_groups)} distinct logprob groups:")
    for group_id, (rep_run, runs) in enumerate(logprob_groups.items(), 1):
        print(f"  Group {group_id}: Runs {runs}")
    print()
    
    # Show pairwise L2 distances
    print("PAIRWISE L2 DISTANCES:")
    print("-" * 80)
    
    # Show as matrix
    print("     ", end="")
    for i in range(num_runs):
        print(f"Run{i:2d}  ", end="")
    print()
    
    for i in range(num_runs):
        print(f"Run{i:2d}", end="  ")
        for j in range(num_runs):
            if i == j:
                print("   -    ", end="")
            else:
                dist = comparison_matrix[i, j]
                if dist < EPSILON:
                    print(" EXACT  ", end="")
                elif dist < 1e-6:
                    print(f"{dist:7.1e}", end="")
                else:
                    print(f"{dist:7.4f}", end="")
        print()
    print()
    
    # Summary statistics
    print("STATISTICS:")
    print("-" * 80)
    
    # Get all pairwise distances (upper triangle, excluding diagonal)
    all_distances = []
    for i in range(num_runs):
        for j in range(i + 1, num_runs):
            all_distances.append(comparison_matrix[i, j])
    
    all_distances = np.array(all_distances)
    
    # Count exact matches vs differences
    num_exact = np.sum(all_distances < EPSILON)
    num_different = len(all_distances) - num_exact
    
    print(f"Total comparisons: {len(all_distances)}")
    print(f"Exact matches: {num_exact} ({100*num_exact/len(all_distances):.1f}%)")
    print(f"Different: {num_different} ({100*num_different/len(all_distances):.1f}%)")
    print()
    
    if num_different > 0:
        non_zero_distances = all_distances[all_distances >= EPSILON]
        print(f"Non-zero distance statistics:")
        print(f"  Min:    {non_zero_distances.min():.6e}")
        print(f"  Max:    {non_zero_distances.max():.6e}")
        print(f"  Mean:   {non_zero_distances.mean():.6e}")
        print(f"  Median: {np.median(non_zero_distances):.6e}")
        print(f"  Std:    {non_zero_distances.std():.6e}")
        print()
    
    # Per-token statistics
    print("PER-TOKEN VARIANCE:")
    print("-" * 80)
    all_logprobs = np.array(logprobs_arrays)
    std_per_token = all_logprobs.std(axis=0)
    
    print(f"Standard deviation across runs for each token:")
    print(f"  Mean:   {std_per_token.mean():.6e}")
    print(f"  Max:    {std_per_token.max():.6e}")
    print(f"  Median: {np.median(std_per_token):.6e}")
    print(f"  Min:    {std_per_token.min():.6e}")
    print()
    
    # Show tokens with highest variance
    max_var_indices = np.argsort(std_per_token)[-5:][::-1]
    print(f"Top 5 tokens with highest variance:")
    for idx in max_var_indices:
        print(f"  Token {idx}: std = {std_per_token[idx]:.6e}")
    print()
    
    return comparison_matrix, token_groups, logprob_groups

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_h200_logprobs.py <json_file>")
        print()
        print("Example:")
        print("  python compare_h200_logprobs.py vllm_kimi_k2_test_20241110_120000.json")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    if not Path(json_path).exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    
    print(f"Loading: {json_path}")
    print()
    
    data = load_experiment_data(json_path)
    
    # Show experiment info
    print("=" * 80)
    print("EXPERIMENT INFO")
    print("=" * 80)
    print()
    print(f"Experiment: {data['experiment']}")
    print(f"Timestamp: {data['timestamp']}")
    print(f"Model: {data['config']['model']}")
    print(f"Tensor parallel: {data['config']['tensor_parallel']}")
    print(f"Repetitions: {data['config']['repetitions']}")
    print(f"Temperature: {data['config']['temperature']}")
    print(f"Max tokens: {data['config']['max_tokens']}")
    print()
    
    # Compare logprobs
    compare_logprobs(data)
    
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
