#!/usr/bin/env python3
"""
Cross-Provider Comparison Script

Compares baseline results from two different providers to identify:
1. Configuration differences (firmware, driver, CUDA versions)
2. FP reproducibility (bit-exact or deviations)
3. Potential confounding variables

Usage:
    python compare_providers.py <provider1_results.json> <provider2_results.json>
"""

import json
import sys
import numpy as np
from pathlib import Path

def load_results(filepath):
    """Load experimental results from JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)

def compare_attestations(attest1, attest2):
    """Compare system attestations to find differences"""
    
    print("="*70)
    print("ATTESTATION COMPARISON")
    print("="*70)
    
    differences = []
    
    # Provider names
    print(f"\nProviders:")
    print(f"  Provider 1: {attest1.get('provider', 'unknown')}")
    print(f"  Provider 2: {attest2.get('provider', 'unknown')}")
    
    # GPU info
    print(f"\nGPU:")
    gpu1 = attest1.get('gpu', {})
    gpu2 = attest2.get('gpu', {})
    
    if gpu1.get('name') != gpu2.get('name'):
        print(f"  [WARN] NAME DIFFERS:")
        print(f"    Provider 1: {gpu1.get('name')}")
        print(f"    Provider 2: {gpu2.get('name')}")
        differences.append("GPU name")
    else:
        print(f"  [OK] Name: {gpu1.get('name')}")
    
    if gpu1.get('capability') != gpu2.get('capability'):
        print(f"  [WARN] CAPABILITY DIFFERS:")
        print(f"    Provider 1: {gpu1.get('capability')}")
        print(f"    Provider 2: {gpu2.get('capability')}")
        differences.append("GPU compute capability")
    else:
        print(f"  [OK] Capability: {gpu1.get('capability')}")
    
    # Detailed GPU info
    print(f"\nGPU Details:")
    gpu_det1 = attest1.get('gpu_detailed', {})
    gpu_det2 = attest2.get('gpu_detailed', {})
    
    for key in ['driver_version', 'memory_total_mb', 'pcie_gen_max', 'pcie_width_max']:
        val1 = gpu_det1.get(key)
        val2 = gpu_det2.get(key)
        
        if val1 != val2:
            print(f"  [WARN] {key.upper()} DIFFERS:")
            print(f"    Provider 1: {val1}")
            print(f"    Provider 2: {val2}")
            differences.append(key)
        else:
            print(f"  [OK] {key}: {val1}")
    
    # Firmware
    print(f"\nFirmware:")
    fw1 = attest1.get('gpu_firmware', {})
    fw2 = attest2.get('gpu_firmware', {})
    
    if fw1 and fw2:
        for key in fw1.keys():
            if key in fw2:
                if fw1[key] != fw2[key]:
                    print(f"  [WARN] {key} DIFFERS:")
                    print(f"    Provider 1: {fw1[key]}")
                    print(f"    Provider 2: {fw2[key]}")
                    differences.append(f"firmware_{key}")
                else:
                    print(f"  [OK] {key}: {fw1[key]}")
    elif not fw1 or not fw2:
        print(f"  [WARN] Firmware info missing for one provider")
        differences.append("firmware_unavailable")
    
    # PyTorch/CUDA
    print(f"\nPyTorch/CUDA:")
    pt1 = attest1.get('pytorch', {})
    pt2 = attest2.get('pytorch', {})
    
    for key in ['version', 'cuda_version', 'cudnn_version']:
        val1 = pt1.get(key)
        val2 = pt2.get(key)
        
        if val1 != val2:
            print(f"  [WARN] {key.upper()} DIFFERS:")
            print(f"    Provider 1: {val1}")
            print(f"    Provider 2: {val2}")
            differences.append(f"pytorch_{key}")
        else:
            print(f"  [OK] {key}: {val1}")
    
    # Compute/persistence modes
    print(f"\nGPU Modes:")
    for key in ['compute_mode', 'persistence_mode']:
        val1 = attest1.get(key)
        val2 = attest2.get(key)
        
        if val1 != val2:
            print(f"  [WARN] {key.upper()} DIFFERS:")
            print(f"    Provider 1: {val1}")
            print(f"    Provider 2: {val2}")
            differences.append(key)
        else:
            print(f"  [OK] {key}: {val1}")
    
    # MIG status
    mig1 = attest1.get('mig_status', {})
    mig2 = attest2.get('mig_status', {})
    
    if mig1.get('is_mig') or mig2.get('is_mig'):
        print(f"\n[WARN] MIG STATUS:")
        print(f"  Provider 1: {'ENABLED' if mig1.get('is_mig') else 'Disabled'}")
        print(f"  Provider 2: {'ENABLED' if mig2.get('is_mig') else 'Disabled'}")
        if mig1.get('is_mig') != mig2.get('is_mig'):
            differences.append("mig_mode")
    
    return differences

def compare_activations(results1, results2):
    """Compare hidden states and key vectors for FP reproducibility"""
    
    print("\n" + "="*70)
    print("ACTIVATION COMPARISON")
    print("="*70)
    
    # Use first run from each
    run1 = results1['runs'][0]
    run2 = results2['runs'][0]
    
    hidden1 = run1['hidden_states']
    hidden2 = run2['hidden_states']
    keys1 = run1['key_vectors']
    keys2 = run2['key_vectors']
    
    print("\nHidden States:")
    hidden_comparison = {}
    
    for layer_name in hidden1.keys():
        if layer_name not in hidden2:
            print(f"  {layer_name}: [WARN] MISSING in provider 2")
            continue
        
        arr1 = np.array(hidden1[layer_name])
        arr2 = np.array(hidden2[layer_name])
        
        # Check bit-exactness
        bit_exact = np.array_equal(arr1, arr2)
        
        if bit_exact:
            print(f"  {layer_name}: [OK] BIT-EXACT")
            hidden_comparison[layer_name] = {
                "bit_exact": True,
                "l2_distance": 0.0,
                "max_abs_diff": 0.0,
                "relative_diff": 0.0
            }
        else:
            # Compute differences
            l2_dist = np.linalg.norm(arr1 - arr2)
            max_abs = np.abs(arr1 - arr2).max()
            norm1 = np.linalg.norm(arr1)
            rel_diff = l2_dist / norm1 if norm1 > 0 else 0
            
            print(f"  {layer_name}: [WARN] DIFFERS")
            print(f"    L2 distance: {l2_dist:.6f}")
            print(f"    Max |diff|: {max_abs:.6f}")
            print(f"    Relative: {rel_diff:.6f} ({rel_diff*100:.4f}%)")
            
            hidden_comparison[layer_name] = {
                "bit_exact": False,
                "l2_distance": float(l2_dist),
                "max_abs_diff": float(max_abs),
                "relative_diff": float(rel_diff)
            }
    
    print("\nKey Vectors:")
    key_comparison = {}
    
    for layer_name in keys1.keys():
        if layer_name not in keys2:
            print(f"  {layer_name}: [WARN] MISSING in provider 2")
            continue
        
        arr1 = np.array(keys1[layer_name])
        arr2 = np.array(keys2[layer_name])
        
        bit_exact = np.array_equal(arr1, arr2)
        
        if bit_exact:
            print(f"  {layer_name}: [OK] BIT-EXACT")
            key_comparison[layer_name] = {
                "bit_exact": True,
                "l2_distance": 0.0,
                "max_abs_diff": 0.0,
                "relative_diff": 0.0
            }
        else:
            l2_dist = np.linalg.norm(arr1 - arr2)
            max_abs = np.abs(arr1 - arr2).max()
            norm1 = np.linalg.norm(arr1)
            rel_diff = l2_dist / norm1 if norm1 > 0 else 0
            
            print(f"  {layer_name}: [WARN] DIFFERS")
            print(f"    L2 distance: {l2_dist:.6f}")
            print(f"    Max |diff|: {max_abs:.6f}")
            print(f"    Relative: {rel_diff:.6f} ({rel_diff*100:.4f}%)")
            
            key_comparison[layer_name] = {
                "bit_exact": False,
                "l2_distance": float(l2_dist),
                "max_abs_diff": float(max_abs),
                "relative_diff": float(rel_diff)
            }
    
    return {
        "hidden_states": hidden_comparison,
        "key_vectors": key_comparison
    }

def find_baseline_files(directory='.'):
    """Auto-discover baseline experiment JSON files"""
    path = Path(directory)
    pattern = '*_baseline_*.json'
    files = sorted(path.glob(pattern))
    return files

def compare_two_files(file1, file2):
    """Compare two experiment result files"""
    if not file1.exists():
        print(f"Error: {file1} not found")
        return None
    
    if not file2.exists():
        print(f"Error: {file2} not found")
        return None
    
    print("="*70)
    print("CROSS-PROVIDER COMPARISON")
    print("="*70)
    print(f"File 1: {file1.name}")
    print(f"File 2: {file2.name}")
    
    # Load results
    results1 = load_results(file1)
    results2 = load_results(file2)
    
    # Compare attestations
    config_diffs = compare_attestations(
        results1['attestation'],
        results2['attestation']
    )
    
    # Compare activations
    activation_comparison = compare_activations(results1, results2)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nConfiguration differences found: {len(config_diffs)}")
    if config_diffs:
        print("  Differs in:")
        for diff in config_diffs:
            print(f"    - {diff}")
    else:
        print("  [OK] Identical configurations")
    
    # Count bit-exact matches
    hidden_exact = sum(1 for v in activation_comparison['hidden_states'].values() if v['bit_exact'])
    hidden_total = len(activation_comparison['hidden_states'])
    
    keys_exact = sum(1 for v in activation_comparison['key_vectors'].values() if v['bit_exact'])
    keys_total = len(activation_comparison['key_vectors'])
    
    print(f"\nActivation comparison:")
    print(f"  Hidden states: {hidden_exact}/{hidden_total} bit-exact")
    print(f"  Key vectors: {keys_exact}/{keys_total} bit-exact")
    
    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    if hidden_exact == hidden_total and keys_exact == keys_total:
        print("[OK] PERFECT MATCH: Bit-exact across providers")
        print("  FP behavior is identical despite provider differences")
        
        if config_diffs:
            print(f"\n  Configuration differences do NOT affect FP results:")
            for diff in config_diffs:
                print(f"    - {diff}")
        
        print("\n  [OK] Verification approach is ROBUST to provider choice")
    
    elif hidden_exact > 0 or keys_exact > 0:
        print("[WARN] PARTIAL MATCH: Some layers bit-exact, others differ")
        print(f"  Hidden: {hidden_exact}/{hidden_total} exact")
        print(f"  Keys: {keys_exact}/{keys_total} exact")
        
        # Find worst deviation
        max_hidden_l2 = max(
            (v['l2_distance'] for v in activation_comparison['hidden_states'].values() if not v['bit_exact']),
            default=0
        )
        max_key_l2 = max(
            (v['l2_distance'] for v in activation_comparison['key_vectors'].values() if not v['bit_exact']),
            default=0
        )
        
        print(f"\n  Worst hidden state deviation: L2 = {max_hidden_l2:.6f}")
        print(f"  Worst key vector deviation: L2 = {max_key_l2:.6f}")
        
        print("\n  Possible causes:")
        print("    - Firmware differences affecting FP behavior")
        print("    - Different CUDA/kernel implementations")
        print("    - Thermal or power management differences (unlikely)")
        
        if config_diffs:
            print(f"\n  Suspect variables:")
            for diff in config_diffs:
                print(f"    - {diff}")
        
        print("\n  [WARN] Further investigation needed")
    
    else:
        print("[ERROR] NO MATCHES: All activations differ")
        print("  Significant FP behavior differences between providers")
        
        # Show magnitude of differences
        avg_hidden_l2 = np.mean([
            v['l2_distance'] for v in activation_comparison['hidden_states'].values()
        ])
        avg_key_l2 = np.mean([
            v['l2_distance'] for v in activation_comparison['key_vectors'].values()
        ])
        
        print(f"\n  Average hidden state L2: {avg_hidden_l2:.6f}")
        print(f"  Average key vector L2: {avg_key_l2:.6f}")
        
        print("\n  Critical: These differences are LARGER than expected")
        print("  from known perturbations (batch size, CUDA version, etc.)")
        
        if config_diffs:
            print(f"\n  Likely culprits:")
            priority_diffs = [d for d in config_diffs if 'firmware' in d or 'driver' in d]
            if priority_diffs:
                for diff in priority_diffs:
                    print(f"    - {diff} (HIGH PRIORITY)")
            else:
                for diff in config_diffs:
                    print(f"    - {diff}")
        
        print("\n  [ERROR] Verification approach may NOT be robust across providers")
        print("  [WARN] Consider requiring specific firmware/driver versions")
    
    # Prepare verdict details
    verdict = {}
    if hidden_exact == hidden_total and keys_exact == keys_total:
        verdict["status"] = "PERFECT_MATCH"
        verdict["message"] = "Bit-exact across providers"
        verdict["interpretation"] = "FP behavior is identical despite provider differences"
        verdict["robustness"] = "Verification approach is ROBUST to provider choice"
        verdict["config_diffs_affect_fp"] = False
    elif hidden_exact > 0 or keys_exact > 0:
        verdict["status"] = "PARTIAL_MATCH"
        verdict["message"] = "Some layers bit-exact, others differ"
        max_hidden_l2 = max(
            (v['l2_distance'] for v in activation_comparison['hidden_states'].values() if not v['bit_exact']),
            default=0
        )
        max_key_l2 = max(
            (v['l2_distance'] for v in activation_comparison['key_vectors'].values() if not v['bit_exact']),
            default=0
        )
        verdict["worst_hidden_l2"] = float(max_hidden_l2)
        verdict["worst_key_l2"] = float(max_key_l2)
        verdict["possible_causes"] = [
            "Firmware differences affecting FP behavior",
            "Different CUDA/kernel implementations",
            "Thermal or power management differences (unlikely)"
        ]
        verdict["suspect_variables"] = config_diffs
        verdict["recommendation"] = "Further investigation needed"
    else:
        verdict["status"] = "NO_MATCH"
        verdict["message"] = "All activations differ"
        avg_hidden_l2 = np.mean([
            v['l2_distance'] for v in activation_comparison['hidden_states'].values()
        ])
        avg_key_l2 = np.mean([
            v['l2_distance'] for v in activation_comparison['key_vectors'].values()
        ])
        verdict["avg_hidden_l2"] = float(avg_hidden_l2)
        verdict["avg_key_l2"] = float(avg_key_l2)
        verdict["severity"] = "CRITICAL - differences LARGER than expected from known perturbations"
        
        priority_diffs = [d for d in config_diffs if 'firmware' in d or 'driver' in d]
        verdict["likely_culprits"] = priority_diffs if priority_diffs else config_diffs
        verdict["robustness"] = "Verification approach may NOT be robust across providers"
        verdict["recommendation"] = "Consider requiring specific firmware/driver versions"
    
    # Prepare detailed configuration comparison
    detailed_config = {
        "provider": {
            "provider1": results1['attestation'].get('provider', 'unknown'),
            "provider2": results2['attestation'].get('provider', 'unknown')
        },
        "gpu": {},
        "gpu_detailed": {},
        "firmware": {},
        "pytorch": {},
        "modes": {},
        "mig_status": {}
    }
    
    # GPU comparison
    gpu1 = results1['attestation'].get('gpu', {})
    gpu2 = results2['attestation'].get('gpu', {})
    for key in ['name', 'capability']:
        detailed_config["gpu"][key] = {
            "provider1": gpu1.get(key),
            "provider2": gpu2.get(key),
            "differs": gpu1.get(key) != gpu2.get(key)
        }
    
    # GPU detailed comparison
    gpu_det1 = results1['attestation'].get('gpu_detailed', {})
    gpu_det2 = results2['attestation'].get('gpu_detailed', {})
    for key in ['driver_version', 'memory_total_mb', 'pcie_gen_max', 'pcie_width_max']:
        detailed_config["gpu_detailed"][key] = {
            "provider1": gpu_det1.get(key),
            "provider2": gpu_det2.get(key),
            "differs": gpu_det1.get(key) != gpu_det2.get(key)
        }
    
    # Firmware comparison
    fw1 = results1['attestation'].get('gpu_firmware', {})
    fw2 = results2['attestation'].get('gpu_firmware', {})
    for key in fw1.keys():
        if key in fw2:
            detailed_config["firmware"][key] = {
                "provider1": fw1[key],
                "provider2": fw2[key],
                "differs": fw1[key] != fw2[key]
            }
    
    # PyTorch/CUDA comparison
    pt1 = results1['attestation'].get('pytorch', {})
    pt2 = results2['attestation'].get('pytorch', {})
    for key in ['version', 'cuda_version', 'cudnn_version']:
        detailed_config["pytorch"][key] = {
            "provider1": pt1.get(key),
            "provider2": pt2.get(key),
            "differs": pt1.get(key) != pt2.get(key)
        }
    
    # Modes comparison
    for key in ['compute_mode', 'persistence_mode']:
        detailed_config["modes"][key] = {
            "provider1": results1['attestation'].get(key),
            "provider2": results2['attestation'].get(key),
            "differs": results1['attestation'].get(key) != results2['attestation'].get(key)
        }
    
    # MIG comparison
    mig1 = results1['attestation'].get('mig_status', {})
    mig2 = results2['attestation'].get('mig_status', {})
    detailed_config["mig_status"] = {
        "provider1_enabled": mig1.get('is_mig', False),
        "provider2_enabled": mig2.get('is_mig', False),
        "differs": mig1.get('is_mig') != mig2.get('is_mig')
    }
    
    # Save comparison report
    comparison_report = {
        "file1": str(file1),
        "file2": str(file2),
        "provider1": results1['attestation']['provider'],
        "provider2": results2['attestation']['provider'],
        "configuration_differences": config_diffs,
        "detailed_configuration_comparison": detailed_config,
        "activation_comparison": activation_comparison,
        "summary": {
            "hidden_states_exact": hidden_exact,
            "hidden_states_total": hidden_total,
            "key_vectors_exact": keys_exact,
            "key_vectors_total": keys_total,
            "perfect_match": hidden_exact == hidden_total and keys_exact == keys_total
        },
        "verdict": verdict
    }
    
    output_file = f"comparison_{results1['attestation']['provider']}_{results2['attestation']['provider']}.json"
    with open(output_file, 'w') as f:
        json.dump(comparison_report, f, indent=2)
    
    print(f"\n[OK] Comparison report saved to: {output_file}")
    
    return comparison_report

def main():
    # Check if files specified manually
    if len(sys.argv) >= 3:
        # Manual specification
        file1 = Path(sys.argv[1])
        file2 = Path(sys.argv[2])
        compare_two_files(file1, file2)
    else:
        # Auto-discovery
        print("="*70)
        print("AUTO-DISCOVERING BASELINE EXPERIMENT FILES")
        print("="*70)
        
        # Look in current directory and common locations
        search_dirs = ['.', '/workspace']
        all_files = []
        
        for dir_path in search_dirs:
            if Path(dir_path).exists():
                found = find_baseline_files(dir_path)
                all_files.extend(found)
        
        # Remove duplicates
        all_files = list(set(all_files))
        all_files.sort(key=lambda x: x.stat().st_mtime)
        
        if len(all_files) == 0:
            print("\n[ERROR] No baseline experiment files found")
            print("Looking for files matching: *_baseline_*.json")
            print("\nUsage:")
            print("  Auto-discovery: python compare_providers.py")
            print("  Manual: python compare_providers.py <file1.json> <file2.json>")
            sys.exit(1)
        
        print(f"\nFound {len(all_files)} experiment file(s):")
        for i, f in enumerate(all_files, 1):
            # Extract provider name from attestation if possible
            try:
                data = load_results(f)
                provider = data['attestation']['provider']
                timestamp = data['attestation']['timestamp']
                print(f"  {i}. {f.name}")
                print(f"     Provider: {provider}, Time: {timestamp}")
            except:
                print(f"  {i}. {f.name}")
        
        if len(all_files) == 1:
            print("\n[ERROR] Only one file found - need at least 2 for comparison")
            sys.exit(1)
        
        elif len(all_files) == 2:
            # Exactly two files - compare them
            print(f"\n[OK] Comparing the 2 files found...")
            print()
            compare_two_files(all_files[0], all_files[1])
        
        else:
            # Multiple files - do pairwise comparisons
            print(f"\n[INFO] Found {len(all_files)} files - performing pairwise comparisons")
            print()
            
            comparisons_done = 0
            for i in range(len(all_files)):
                for j in range(i + 1, len(all_files)):
                    comparisons_done += 1
                    print("\n" + "="*70)
                    print(f"COMPARISON {comparisons_done}: File {i+1} vs File {j+1}")
                    print("="*70)
                    print()
                    compare_two_files(all_files[i], all_files[j])
                    print()
            
            print(f"\n[OK] Completed {comparisons_done} pairwise comparisons")

if __name__ == "__main__":
    main()