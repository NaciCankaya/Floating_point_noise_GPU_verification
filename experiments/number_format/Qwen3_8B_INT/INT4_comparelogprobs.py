#!/usr/bin/env python3
"""
Compare logprobs across different quantization formats

Reads JSON result files from unified_quantization_experiment.py and compares
the logprobs to identify which formats produce identical vs different outputs.
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys


def load_results(directory: str) -> Dict[str, np.ndarray]:
    """Load logprobs from all JSON result files in directory"""
    results = {}

    json_files = list(Path(directory).glob("*_investigation_*.json"))

    if not json_files:
        print(f"No result files found in {directory}")
        print("Looking for files matching pattern: *_investigation_*.json")
        sys.exit(1)

    print(f"Found {len(json_files)} result file(s)")
    print()

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            variant = data.get('variant', 'unknown')

            # Get first repetition's logprobs from baseline
            if 'baseline' in data and 'logprobs' in data['baseline']:
                logprobs = np.array(data['baseline']['logprobs'][0])
                results[variant] = logprobs
                print(f"✓ Loaded {variant}: {len(logprobs)} tokens")
            else:
                print(f"⚠ Skipping {variant}: no baseline logprobs found")

        except Exception as e:
            print(f"✗ Error loading {json_file}: {e}")

    print()
    return results


def compare_logprobs(results: Dict[str, np.ndarray], tolerance: float = 1e-10) -> Tuple[List, List]:
    """Compare all pairs of logprobs and categorize as identical or different"""

    variants = sorted(results.keys())
    identical_pairs = []
    different_pairs = []

    for i, variant1 in enumerate(variants):
        for j, variant2 in enumerate(variants):
            if i >= j:  # Skip self-comparison and duplicates
                continue

            logprobs1 = results[variant1]
            logprobs2 = results[variant2]

            # Check if same length
            if len(logprobs1) != len(logprobs2):
                l2_dist = float('inf')
                max_diff = float('inf')
                identical = False
            else:
                # Check if identical (within tolerance)
                identical = np.allclose(logprobs1, logprobs2, rtol=0, atol=tolerance)

                # Compute L2 distance and max difference
                l2_dist = np.linalg.norm(logprobs1 - logprobs2)
                max_diff = np.max(np.abs(logprobs1 - logprobs2))

            comparison = {
                'pair': (variant1, variant2),
                'identical': identical,
                'l2_distance': l2_dist,
                'max_difference': max_diff
            }

            if identical:
                identical_pairs.append(comparison)
            else:
                different_pairs.append(comparison)

    return identical_pairs, different_pairs


def print_results(identical_pairs: List, different_pairs: List):
    """Print comparison results in a readable format"""

    print("=" * 80)
    print("LOGPROB COMPARISON RESULTS")
    print("=" * 80)
    print()

    # Identical pairs
    if identical_pairs:
        print(f"✓ IDENTICAL LOGPROBS ({len(identical_pairs)} pair(s)):")
        print("-" * 80)
        for comp in identical_pairs:
            v1, v2 = comp['pair']
            print(f"  {v1:20} ≡ {v2:20}")
            print(f"    L2 distance: {comp['l2_distance']:.2e}")
            print(f"    Max diff:    {comp['max_difference']:.2e}")
            print()
    else:
        print("✓ IDENTICAL LOGPROBS: None")
        print()

    # Different pairs
    if different_pairs:
        print(f"✗ DIFFERENT LOGPROBS ({len(different_pairs)} pair(s)):")
        print("-" * 80)

        # Sort by L2 distance (smallest first)
        different_pairs_sorted = sorted(different_pairs, key=lambda x: x['l2_distance'])

        for comp in different_pairs_sorted:
            v1, v2 = comp['pair']
            print(f"  {v1:20} ≠ {v2:20}")
            print(f"    L2 distance: {comp['l2_distance']:.6e}")
            print(f"    Max diff:    {comp['max_difference']:.6e}")
            print()
    else:
        print("✗ DIFFERENT LOGPROBS: None")
        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Identical pairs: {len(identical_pairs)}")
    print(f"Different pairs: {len(different_pairs)}")
    print()


def print_matrix(results: Dict[str, np.ndarray], tolerance: float = 1e-10):
    """Print a comparison matrix showing all pairwise comparisons"""

    variants = sorted(results.keys())
    n = len(variants)

    print("=" * 80)
    print("COMPARISON MATRIX")
    print("=" * 80)
    print()
    print("Legend: ✓ = identical, ✗ = different (L2 distance shown)")
    print()

    # Print header
    print(f"{'':20}", end='')
    for v in variants:
        print(f"{v[:15]:>15}", end=' ')
    print()
    print("-" * (20 + 16 * n))

    # Print matrix
    for i, v1 in enumerate(variants):
        print(f"{v1:20}", end='')
        for j, v2 in enumerate(variants):
            if i == j:
                print(f"{'—':>15}", end=' ')
            elif i > j:
                # Lower triangle: show symbol only
                logprobs1 = results[v1]
                logprobs2 = results[v2]

                if len(logprobs1) == len(logprobs2):
                    identical = np.allclose(logprobs1, logprobs2, rtol=0, atol=tolerance)
                    symbol = '✓' if identical else '✗'
                else:
                    symbol = '✗'

                print(f"{symbol:>15}", end=' ')
            else:
                # Upper triangle: show L2 distance
                logprobs1 = results[v1]
                logprobs2 = results[v2]

                if len(logprobs1) == len(logprobs2):
                    l2_dist = np.linalg.norm(logprobs1 - logprobs2)
                    print(f"{l2_dist:>15.2e}", end=' ')
                else:
                    print(f"{'length≠':>15}", end=' ')
        print()

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare logprobs across different quantization formats"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory containing JSON result files (default: current directory)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-10,
        help="Tolerance for considering logprobs identical (default: 1e-10)"
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Also print comparison matrix"
    )

    args = parser.parse_args()

    # Load results
    results = load_results(args.directory)

    if len(results) < 2:
        print("Need at least 2 result files to compare")
        sys.exit(1)

    # Compare
    identical_pairs, different_pairs = compare_logprobs(results, tolerance=args.tolerance)

    # Print results
    print_results(identical_pairs, different_pairs)

    # Print matrix if requested
    if args.matrix:
        print_matrix(results, tolerance=args.tolerance)


if __name__ == "__main__":
    main()