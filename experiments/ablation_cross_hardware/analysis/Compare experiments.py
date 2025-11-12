#!/usr/bin/env python3
"""
Compare Forensics Experiments

Analyzes experiment JSON files to determine detectability of inference modifications.
Adapted from legacy batch_matrix comparison script to work with new JSON format.

Key analyses:
1. Reproducibility: Verify bit-exact results within identical setups
2. Baseline: Cross-hardware distance (same config, different GPUs)  
3. Signal: Within-hardware distance (different configs, same GPU)
4. Detectability: Can we detect config changes across different hardware?

Usage:
    # Analyze single experiment
    python compare_experiments.py batch_size_experiment.json
    
    # Analyze all experiments and generate summary
    python compare_experiments.py --all --experiment-dir .
"""

import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

# Add parent directory to path to import common utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import ExperimentReader

# ============================================================================
# DISTANCE COMPUTATION
# ============================================================================

def compute_l2_distance(vec1: List[float], vec2: List[float]) -> float:
    """Compute L2 distance between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.linalg.norm(v1 - v2))


def compare_runs(
    run1: Dict,
    run2: Dict,
    signal_type: str = "key_vectors",
    layer: str = "layer_39"
) -> Dict:
    """
    Compare two runs across all decode steps and positions.
    
    Returns max distance found across all measurements.
    """
    distances = []
    
    for step1, step2 in zip(run1["decode_steps"], run2["decode_steps"]):
        for pos in ["pos_-3", "pos_-2", "pos_-1"]:
            try:
                if signal_type == "logprobs":
                    vec1 = step1[signal_type][pos]["log_probs"]
                    vec2 = step2[signal_type][pos]["log_probs"]
                else:
                    vec1 = step1[signal_type][layer][pos]
                    vec2 = step2[signal_type][layer][pos]
                
                distance = compute_l2_distance(vec1, vec2)
                distances.append(distance)
            except (KeyError, IndexError):
                continue
    
    return {
        'mean': np.mean(distances) if distances else 0.0,
        'max': np.max(distances) if distances else 0.0,
        'median': np.median(distances) if distances else 0.0,
        'std': np.std(distances) if distances else 0.0
    }


# ============================================================================
# ANALYSIS: REPRODUCIBILITY
# ============================================================================

def analyze_reproducibility(reader: ExperimentReader, signal_type: str = "key_vectors") -> Dict:
    """
    Check bit-exact reproducibility within identical setups.
    
    Expected: All repetitions should be identical (distance = 0).
    """
    print("\n" + "="*80)
    print("1. REPRODUCIBILITY CHECK")
    print("="*80)
    print("Testing: Multiple repetitions of identical setup")
    print("Expected: BIT-EXACT (distance = 0)\n")
    
    data = reader.extract_all_data()
    results = {}
    all_exact = True
    
    for config in data["configurations"]:
        config_id = config["config_id"]
        runs = reader.get_runs(config_id=config_id)
        
        if len(runs) < 2:
            continue
        
        print(f"{config_id}:")
        
        # Compare all pairs
        max_dist = 0.0
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                comparison = compare_runs(runs[i], runs[j], signal_type=signal_type)
                dist = comparison['max']
                max_dist = max(max_dist, dist)
                
                status = "✓ exact" if dist == 0.0 else f"✗ DIFFERS (Δ={dist:.2e})"
                print(f"  Rep {i} vs Rep {j}: {status}")
                
                if dist > 0:
                    all_exact = False
        
        results[config_id] = {
            'is_exact': max_dist == 0.0,
            'max_distance': max_dist
        }
        print()
    
    if all_exact:
        print("✓ ALL CONFIGURATIONS ARE DETERMINISTIC\n")
    else:
        print("✗ NON-DETERMINISM DETECTED - INVESTIGATION REQUIRED\n")
    
    return results


# ============================================================================
# ANALYSIS: CROSS-HARDWARE BASELINE
# ============================================================================

def analyze_baseline(reader: ExperimentReader, signal_type: str = "key_vectors") -> Dict:
    """
    Establish baseline: cross-hardware distance for same configuration.
    
    This measures the "noise floor" from hardware differences alone.
    """
    print("\n" + "="*80)
    print("2. CROSS-HARDWARE BASELINE")
    print("="*80)
    print("Testing: Same configuration on different hardware")
    print("Expected: Small systematic deviation (hardware fingerprint)\n")
    
    data = reader.extract_all_data()
    
    # Get unique hardware types
    hardware_types = list(set(c["hardware"] for c in data["configurations"]))
    
    if len(hardware_types) < 2:
        print("⚠ Only one hardware type - cannot compute baseline\n")
        return {}
    
    hw1, hw2 = sorted(hardware_types)[:2]
    print(f"Comparing: {hw1} vs {hw2}\n")
    
    results = {}
    baseline_distances = []
    
    # Find configs with matching variable values
    configs_hw1 = reader.get_configs_by_hardware(hw1)
    configs_hw2 = reader.get_configs_by_hardware(hw2)
    
    for config1 in configs_hw1:
        value = config1["variable_value"]
        
        # Find matching config on other hardware
        matching = [c for c in configs_hw2 if c["variable_value"] == value]
        if not matching:
            continue
        
        config2 = matching[0]
        
        # Get first rep from each
        runs1 = reader.get_runs(config_id=config1["config_id"], rep_id=0)
        runs2 = reader.get_runs(config_id=config2["config_id"], rep_id=0)
        
        if not runs1 or not runs2:
            continue
        
        comparison = compare_runs(runs1[0], runs2[0], signal_type=signal_type)
        distance = comparison['max']
        
        print(f"Variable value = {value}:")
        print(f"  {config1['config_id']} vs {config2['config_id']}")
        print(f"  Distance: {distance:.2e}\n")
        
        results[value] = distance
        baseline_distances.append(distance)
    
    if baseline_distances:
        mean_baseline = np.mean(baseline_distances)
        std_baseline = np.std(baseline_distances)
        
        print("Baseline Summary:")
        print(f"  Mean: {mean_baseline:.2e}")
        print(f"  Std:  {std_baseline:.2e}")
        print(f"  Range: [{min(baseline_distances):.2e}, {max(baseline_distances):.2e}]")
        print()
        
        return {
            'per_value': results,
            'mean': mean_baseline,
            'std': std_baseline,
            'values': baseline_distances
        }
    
    return {}


# ============================================================================
# ANALYSIS: WITHIN-HARDWARE DETECTABILITY
# ============================================================================

def analyze_within_hardware(reader: ExperimentReader, signal_type: str = "key_vectors") -> Dict:
    """
    Test detectability within same hardware.
    
    Measures: Can we detect variable changes on same hardware?
    """
    print("\n" + "="*80)
    print("3. WITHIN-HARDWARE DETECTABILITY")
    print("="*80)
    print("Testing: Variable changes on same hardware")
    print("Expected: Clear signal (distance > 0)\n")
    
    data = reader.extract_all_data()
    variable = data["experiment_metadata"]["variable_tested"]
    
    # Get unique hardware types
    hardware_types = list(set(c["hardware"] for c in data["configurations"]))
    
    results = {}
    
    for hardware in hardware_types:
        print(f"{hardware}:")
        print("-" * 40)
        
        configs = reader.get_configs_by_hardware(hardware)
        configs.sort(key=lambda c: c["variable_value"])
        
        if len(configs) < 2:
            print("  Only one configuration\n")
            continue
        
        # Use smallest value as baseline
        baseline_config = configs[0]
        baseline_runs = reader.get_runs(config_id=baseline_config["config_id"], rep_id=0)
        
        if not baseline_runs:
            continue
        
        baseline_run = baseline_runs[0]
        baseline_value = baseline_config["variable_value"]
        
        print(f"  Baseline: {baseline_config['config_id']} (value={baseline_value})\n")
        
        hw_results = {}
        
        for config in configs[1:]:
            test_runs = reader.get_runs(config_id=config["config_id"], rep_id=0)
            
            if not test_runs:
                continue
            
            test_value = config["variable_value"]
            
            comparison = compare_runs(baseline_run, test_runs[0], signal_type=signal_type)
            distance = comparison['max']
            
            print(f"  {baseline_value} → {test_value}: Δ={distance:.2e}")
            
            hw_results[test_value] = distance
        
        results[hardware] = hw_results
        print()
    
    return results


# ============================================================================
# ANALYSIS: CROSS-HARDWARE DETECTABILITY (KEY TEST)
# ============================================================================

def analyze_cross_hardware_detectability(
    reader: ExperimentReader,
    baseline: Dict,
    signal_type: str = "key_vectors",
    detectability_ratio: float = 1.5
) -> Dict:
    """
    KEY TEST: Can we detect variable changes across different hardware?
    
    Compares: (cross-hardware + variable mismatch) vs (cross-hardware baseline)
    
    Criterion: Distance must be >{detectability_ratio}× baseline to be detectable.
    """
    print("\n" + "="*80)
    print("4. CROSS-HARDWARE DETECTABILITY (KEY TEST)")
    print("="*80)
    print("Testing: Variable changes across different hardware")
    print(f"Criterion: Distance > {detectability_ratio}× baseline\n")
    
    data = reader.extract_all_data()
    
    # Get unique hardware types
    hardware_types = list(set(c["hardware"] for c in data["configurations"]))
    
    if len(hardware_types) < 2:
        print("⚠ Only one hardware type\n")
        return {}
    
    hw1, hw2 = sorted(hardware_types)[:2]
    
    baseline_mean = baseline.get('mean', 0)
    
    if baseline_mean == 0:
        print("⚠ No baseline available\n")
        return {}
    
    print(f"Reference baseline: {baseline_mean:.2e}\n")
    
    results = {}
    detectable_count = 0
    total_count = 0
    
    configs_hw1 = reader.get_configs_by_hardware(hw1)
    configs_hw2 = reader.get_configs_by_hardware(hw2)
    
    # Compare all cross-pairs with DIFFERENT variable values
    for config1 in configs_hw1:
        runs1 = reader.get_runs(config_id=config1["config_id"], rep_id=0)
        if not runs1:
            continue
        
        for config2 in configs_hw2:
            value1 = config1["variable_value"]
            value2 = config2["variable_value"]
            
            # Skip same values (already in baseline)
            if value1 == value2:
                continue
            
            runs2 = reader.get_runs(config_id=config2["config_id"], rep_id=0)
            if not runs2:
                continue
            
            comparison = compare_runs(runs1[0], runs2[0], signal_type=signal_type)
            distance = comparison['max']
            
            ratio = distance / baseline_mean
            detectable = ratio >= detectability_ratio
            
            status = "✓ DETECTABLE" if detectable else "✗ not detectable"
            
            print(f"{hw1}={value1} vs {hw2}={value2}:")
            print(f"  Distance: {distance:.2e}")
            print(f"  Ratio: {ratio:.2f}×")
            print(f"  {status}\n")
            
            results[(value1, value2)] = {
                'distance': distance,
                'ratio': ratio,
                'detectable': detectable
            }
            
            total_count += 1
            if detectable:
                detectable_count += 1
    
    # Summary
    if total_count > 0:
        pct = (detectable_count / total_count) * 100
        avg_ratio = np.mean([r['ratio'] for r in results.values()])
        
        print("="*80)
        print("KEY TEST SUMMARY")
        print("="*80)
        print(f"Detectable: {detectable_count}/{total_count} ({pct:.1f}%)")
        print(f"Average ratio: {avg_ratio:.2f}×\n")
        
        if pct > 50:
            variable = data["experiment_metadata"]["variable_tested"]
            print(f"✓ {variable.upper()} IS DETECTABLE CROSS-HARDWARE")
            print("  → Forensic verification can detect this type of evasion")
        else:
            variable = data["experiment_metadata"]["variable_tested"]
            print(f"✗ {variable.upper()} IS NOT RELIABLY DETECTABLE CROSS-HARDWARE")
            print("  → Signal is comparable to or smaller than hardware noise")
        print()
    
    return {
        'comparisons': results,
        'detectable_count': detectable_count,
        'total_count': total_count,
        'percentage': pct if total_count > 0 else 0
    }


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_experiment(json_path: str, signal_type: str = "key_vectors") -> Dict:
    """Run complete analysis on single experiment."""
    print("\n" + "#"*80)
    print(f"# ANALYZING: {Path(json_path).name}")
    print("#"*80)
    
    reader = ExperimentReader(json_path)
    data = reader.extract_all_data()
    
    exp_type = data["experiment_metadata"]["experiment_type"]
    variable = data["experiment_metadata"]["variable_tested"]
    
    print(f"\nExperiment: {exp_type}")
    print(f"Variable: {variable}")
    print(f"Configurations: {len(data['configurations'])}")
    print(f"Runs: {len(data['runs'])}")
    
    report = {
        'experiment_type': exp_type,
        'variable': variable,
        'signal_type': signal_type
    }
    
    # 1. Reproducibility
    report['reproducibility'] = analyze_reproducibility(reader, signal_type)
    
    # 2. Baseline
    report['baseline'] = analyze_baseline(reader, signal_type)
    
    # 3. Within-hardware
    report['within_hardware'] = analyze_within_hardware(reader, signal_type)
    
    # 4. Cross-hardware detectability
    report['cross_hardware'] = analyze_cross_hardware_detectability(
        reader,
        report['baseline'],
        signal_type
    )
    
    return report


def analyze_all_experiments(experiment_dir: str = ".") -> Tuple[List[Dict], Dict]:
    """Analyze all experiment files and generate summary."""
    experiment_files = [
        "batch_size_experiment.json",
        "compile_experiment.json",
        "quantization_experiment.json",
        "attention_experiment.json",
        "concurrent_streams_experiment.json",
        "tensor_parallel_experiment.json",
        "expert_parallel_experiment.json",
        "cuda_version_experiment.json"
    ]
    
    reports = []
    
    for filename in experiment_files:
        filepath = Path(experiment_dir) / filename
        
        if not filepath.exists():
            print(f"\n⚠ Skipping {filename} (not found)")
            continue
        
        try:
            report = analyze_experiment(str(filepath))
            reports.append(report)
        except Exception as e:
            print(f"\n✗ Error analyzing {filename}: {e}")
            continue
    
    # Generate summary table
    summary = generate_summary_table(reports)
    
    return reports, summary


def generate_summary_table(reports: List[Dict]) -> Dict:
    """Generate summary table across all experiments."""
    print("\n\n" + "="*100)
    print("OVERALL SUMMARY TABLE")
    print("="*100)
    
    print(f"{'Experiment':<25} {'Variable':<20} {'Baseline':<12} {'Detectable':<12} {'Status':<15}")
    print("-"*100)
    
    summary_data = []
    
    for report in reports:
        exp = report['experiment_type']
        var = report['variable']
        
        # Baseline
        baseline = report.get('baseline', {})
        baseline_mean = baseline.get('mean', 0)
        
        # Detectability
        cross_hw = report.get('cross_hardware', {})
        pct = cross_hw.get('percentage', 0)
        
        # Classification
        if pct > 75:
            status = "✓ Strong"
        elif pct > 50:
            status = "~ Moderate"
        else:
            status = "✗ Weak"
        
        print(f"{exp:<25} {var:<20} {baseline_mean:<12.2e} {pct:<11.1f}% {status:<15}")
        
        summary_data.append({
            'experiment': exp,
            'variable': var,
            'baseline': baseline_mean,
            'detectability_pct': pct,
            'status': status
        })
    
    print("="*100)
    print()
    
    return {'experiments': summary_data}


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze forensics experiments")
    parser.add_argument(
        "file",
        nargs="?",
        help="Single experiment JSON to analyze"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all experiments in directory"
    )
    parser.add_argument(
        "--experiment-dir",
        default=".",
        help="Directory containing experiment JSONs"
    )
    parser.add_argument(
        "--signal",
        default="key_vectors",
        choices=["hidden_states", "key_vectors", "logprobs"],
        help="Signal type to analyze"
    )
    parser.add_argument(
        "--output",
        default="analysis_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    if args.all:
        reports, summary = analyze_all_experiments(args.experiment_dir)
        
        # Save results
        output_data = {
            'summary': summary,
            'detailed_reports': reports
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {args.output}")
        
    elif args.file:
        report = analyze_experiment(args.file, signal_type=args.signal)
        
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nResults saved to: {args.output}")
        
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
