#!/usr/bin/env python3
"""
Check for exact logprob matches at each token position across runs
"""

import json
import numpy as np
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_position_matches.py <json_file>")
        sys.exit(1)
    
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)
    
    logprobs_arrays = [np.array(lp) for lp in data['logprobs_vectors']]
    num_runs = len(logprobs_arrays)
    num_tokens = len(logprobs_arrays[0])
    
    EPSILON = 1e-10
    
    print("=" * 80)
    print("EXACT MATCHES BY TOKEN POSITION")
    print("=" * 80)
    print()
    
    for pos in range(num_tokens):
        # Extract logprobs at this position from all runs
        logprobs_at_pos = [lp[pos] for lp in logprobs_arrays]
        
        # Find exact matches
        matches = []
        for i in range(num_runs):
            for j in range(i + 1, num_runs):
                if abs(logprobs_at_pos[i] - logprobs_at_pos[j]) < EPSILON:
                    matches.append((i, j))
        
        # Calculate std
        std = np.std(logprobs_at_pos)
        
        # Show results
        if matches:
            print(f"Token {pos:2d}: std={std:.6e}, {len(matches)} exact matches")
            if len(matches) <= 10:
                print(f"         Matches: {matches}")
            else:
                print(f"         First 10: {matches[:10]}")
        else:
            print(f"Token {pos:2d}: std={std:.6e}, NO exact matches")
    
    print()
    print("=" * 80)
    
    # Summary
    total_positions = num_tokens
    positions_with_matches = sum(1 for pos in range(num_tokens) 
                                  if any(abs(logprobs_arrays[i][pos] - logprobs_arrays[j][pos]) < EPSILON
                                        for i in range(num_runs)
                                        for j in range(i + 1, num_runs)))
    
    print(f"Positions with at least one exact match: {positions_with_matches}/{total_positions}")
    print(f"Positions with NO exact matches: {total_positions - positions_with_matches}/{total_positions}")

if __name__ == "__main__":
    main()
