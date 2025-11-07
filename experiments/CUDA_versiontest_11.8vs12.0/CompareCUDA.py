#!/usr/bin/env python3
"""
Compare CUDA version experiment results
Analyzes activation differences between cu118 and cu120
"""

import json
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

def load_results(filepath):
    """Load experiment results from JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_comparison_results(output_path, comparison_data):
    """Save comparison results to JSON"""
    with open(output_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\n📁 Results saved to: {output_path}")

def compare_cuda_versions(file1, file2, save_output=True):
    """Compare two CUDA version experiment results"""
    
    print("="*60)
    print("CUDA VERSION COMPARISON ANALYSIS")
    print("="*60)
    
    # Load results
    results1 = load_results(file1)
    results2 = load_results(file2)
    
    cuda1 = results1['software']['cuda_build']
    cuda2 = results2['software']['cuda_build']
    
    print(f"\nComparing: {cuda1} vs {cuda2}")
    print(f"  File 1: {file1}")
    print(f"  File 2: {file2}")
    
    # Initialize output structure
    comparison_output = {
        "timestamp": datetime.now().isoformat(),
        "files": {
            "file1": str(file1),
            "file2": str(file2)
        },
        "cuda_versions": {
            "cuda1": cuda1,
            "cuda2": cuda2
        }
    }
    
    # Verify same experimental setup
    print("\n" + "-"*60)
    print("SETUP VERIFICATION")
    print("-"*60)
    
    checks = {
        "GPU": (results1['hardware']['gpu'], results2['hardware']['gpu']),
        "PyTorch": (results1['software']['pytorch_version'].split('+')[0], 
                   results2['software']['pytorch_version'].split('+')[0]),
        "Model": (results1['model'], results2['model']),
        "Attention": (results1['software']['attention_implementation'],
                     results2['software']['attention_implementation']),
        "Precision": (results1['config']['dtype'], results2['config']['dtype']),
        "Prompt tokens": (results1['config']['prompt_tokens'], 
                         results2['config']['prompt_tokens'])
    }
    
    setup_verification = {}
    all_match = True
    for key, (val1, val2) in checks.items():
        match = val1 == val2
        symbol = "✓" if match else "✗"
        print(f"  {symbol} {key}: {val1} {'==' if match else '!='} {val2}")
        setup_verification[key] = {
            "value1": val1,
            "value2": val2,
            "match": match
        }
        if not match and key != "PyTorch":  # PyTorch full version differs by CUDA
            all_match = False
    
    comparison_output["setup_verification"] = {
        "checks": setup_verification,
        "all_match": all_match
    }
    
    if not all_match:
        print("\n⚠ WARNING: Experimental setups differ!")
        if save_output:
            output_path = f"comparison_{cuda1}_vs_{cuda2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            save_comparison_results(output_path, comparison_output)
        return
    
    # Get layers
    layers = sorted(results1['config']['layers_sampled'])
    
    print("\n" + "-"*60)
    print("WITHIN-CUDA REPRODUCIBILITY")
    print("-"*60)
    
    reproducibility_info = {
        cuda1: results1['reproducibility']['all_repetitions_identical'],
        cuda2: results2['reproducibility']['all_repetitions_identical']
    }
    print(f"{cuda1}: {reproducibility_info[cuda1]}")
    print(f"{cuda2}: {reproducibility_info[cuda2]}")
    
    comparison_output["reproducibility"] = reproducibility_info
    
    # Compare activations layer by layer
    print("\n" + "="*60)
    print("LAYER-BY-LAYER DEVIATION ANALYSIS")
    print("="*60)
    
    print(f"\nComparing: {cuda1} vs {cuda2}")
    print(f"Layers analyzed: {layers}\n")
    
    deviations = {}
    
    for layer_idx in layers:
        layer_name = f"layer_{layer_idx}"
        
        # Get mean activations across repetitions
        acts1 = np.array(results1['raw_activations'][layer_name])  # shape: (reps, dim)
        acts2 = np.array(results2['raw_activations'][layer_name])
        
        mean1 = acts1.mean(axis=0)
        mean2 = acts2.mean(axis=0)
        
        # Compute L2 distance
        l2_dist = np.linalg.norm(mean1 - mean2)
        norm1 = np.linalg.norm(mean1)
        norm2 = np.linalg.norm(mean2)
        relative_diff = l2_dist / norm1 if norm1 > 0 else 0
        
        # Compute element-wise differences
        diff = np.abs(mean1 - mean2)
        max_diff = diff.max()
        dims_affected = (diff > 0.01).sum()
        dims_total = len(diff)
        
        deviations[layer_idx] = {
            "l2_distance": float(l2_dist),
            "relative_diff": float(relative_diff),
            "max_diff": float(max_diff),
            "dims_affected": int(dims_affected),
            "dims_total": int(dims_total),
            "norm1": float(norm1),
            "norm2": float(norm2)
        }
        
        print(f"Layer {layer_idx}:")
        print(f"  L2 distance: {l2_dist:.6f}")
        print(f"  Relative diff: {relative_diff:.6f} ({relative_diff*100:.3f}%)")
        print(f"  Max |diff|: {max_diff:.6f}")
        print(f"  Dims with |diff| > 0.01: {dims_affected}/{dims_total}")
        print(f"  {cuda1} norm: {norm1:.2f}")
        print(f"  {cuda2} norm: {norm2:.2f}")
        print()
    
    comparison_output["layer_deviations"] = deviations
    
    # Growth analysis
    print("="*60)
    print("ERROR PROPAGATION ANALYSIS")
    print("="*60)
    
    first_layer = layers[0]
    last_layer = layers[-1]
    
    first_l2 = deviations[first_layer]["l2_distance"]
    last_l2 = deviations[last_layer]["l2_distance"]
    growth_factor = last_l2 / first_l2 if first_l2 > 0 else float('inf')
    
    print(f"\nL2 distance progression:")
    for layer_idx in layers:
        l2 = deviations[layer_idx]["l2_distance"]
        print(f"  Layer {layer_idx}: {l2:.6f}")
    
    print(f"\nRelative difference progression:")
    for layer_idx in layers:
        rel = deviations[layer_idx]["relative_diff"]
        print(f"  Layer {layer_idx}: {rel:.6f} ({rel*100:.3f}%)")
    
    error_propagation = {
        "first_layer": first_layer,
        "last_layer": last_layer,
        "first_l2": first_l2,
        "last_l2": last_l2,
        "growth_factor": float(growth_factor),
        "grows": last_l2 > first_l2
    }
    comparison_output["error_propagation"] = error_propagation
    
    print(f"\n{'='*60}")
    print("DIAGNOSIS")
    print("="*60)
    
    diagnosis = {}
    
    if last_l2 > first_l2:
        print(f"✓ ERROR PROPAGATION CONFIRMED")
        print(f"  Deviation grows across layers: True")
        print(f"  Growth factor (first→last): {growth_factor:.2f}x")
        print(f"  First layer L2: {first_l2:.6f}")
        print(f"  Last layer L2: {last_l2:.6f}")
        diagnosis["error_propagation_confirmed"] = True
    else:
        print(f"? UNEXPECTED: Deviation does not grow")
        print(f"  First layer L2: {first_l2:.6f}")
        print(f"  Last layer L2: {last_l2:.6f}")
        diagnosis["error_propagation_confirmed"] = False
    
    print(f"\n{'='*60}")
    print("VERIFICATION VIABILITY")
    print("="*60)
    
    # Assess detectability
    if last_l2 > 10:
        signal_strength = "EXCELLENT"
        print(f"📊 EXCELLENT SIGNAL: L2={last_l2:.1f}")
        print(f"  This deviation is easily detectable for forensics")
    elif last_l2 > 1:
        signal_strength = "STRONG"
        print(f"✓ STRONG SIGNAL: L2={last_l2:.2f}")
        print(f"  CUDA version is forensically distinguishable")
    elif last_l2 > 0.1:
        signal_strength = "WEAK"
        print(f"⚠ WEAK SIGNAL: L2={last_l2:.3f}")
        print(f"  May be detectable with careful analysis")
    else:
        signal_strength = "NOT_DETECTABLE"
        print(f"✗ NOT DETECTABLE: L2={last_l2:.6f}")
        print(f"  Difference is below practical threshold")
    
    diagnosis["signal_strength"] = signal_strength
    diagnosis["last_l2_distance"] = last_l2
    
    # Check reproducibility
    perfect_reproducibility = (
        results1['reproducibility']['all_repetitions_identical'] and 
        results2['reproducibility']['all_repetitions_identical']
    )
    
    if perfect_reproducibility:
        print(f"\n✓ Perfect within-CUDA reproducibility")
        print(f"  Systematic deviation dominates over statistical noise")
        print(f"  This is ideal for forensics applications")
    
    diagnosis["perfect_within_cuda_reproducibility"] = perfect_reproducibility
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    if last_l2 > 1:
        conclusion = "DISTINGUISHABLE"
        print(f"\n✓ CUDA {cuda1} vs {cuda2} create distinguishable signatures")
        print(f"  L2 distance: {last_l2:.2f}")
        print(f"  Forensic verification is viable")
    elif last_l2 > 0.1:
        conclusion = "MEASURABLE"
        print(f"\n⚠ CUDA versions create small but measurable differences")
        print(f"  L2 distance: {last_l2:.3f}")
        print(f"  May require multiple samples for reliable detection")
    else:
        conclusion = "IDENTICAL"
        print(f"\n✗ CUDA versions produce nearly identical results")
        print(f"  L2 distance: {last_l2:.6f}")
        print(f"  Not forensically distinguishable")
    
    diagnosis["conclusion"] = conclusion
    comparison_output["diagnosis"] = diagnosis
    
    print("\n" + "="*60)
    
    # Save results
    if save_output:
        output_path = f"comparison_{cuda1}_vs_{cuda2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_comparison_results(output_path, comparison_output)
    
    return comparison_output

if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        print("Usage: python compare_cuda_results.py <file_cu118.json> <file_cu121.json> [--no-save]")
        print("\nOr find files automatically:")
        
        # Try to find files automatically
        workspace = Path(".")
        cu118_files = list(workspace.glob("*_cu118_*.json"))
        cu121_files = list(workspace.glob("*_cu121_*.json"))
        
        if cu118_files and cu121_files:
            print(f"\nFound files:")
            print(f"  cu118: {cu118_files[0]}")
            print(f"  cu121: {cu121_files[0]}")
            print(f"\nRunning comparison...")
            compare_cuda_versions(str(cu118_files[0]), str(cu121_files[0]))
        else:
            print(f"\nNo matching files found in current directory")
            sys.exit(1)
    else:
        save_output = "--no-save" not in sys.argv
        compare_cuda_versions(sys.argv[1], sys.argv[2], save_output)