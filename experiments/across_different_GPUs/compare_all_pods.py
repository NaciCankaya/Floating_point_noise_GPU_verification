import json
import numpy as np
import sys
from itertools import combinations

def load_activations(json_path):
    """Load raw activations from experiment JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def compare_all_pods(json_paths):
    """Compare activations across multiple pods"""
    
    # Load all pods
    pods = [load_activations(path) for path in json_paths]
    n_pods = len(pods)
    
    print("="*70)
    print(f"MULTI-POD REPRODUCIBILITY ANALYSIS ({n_pods} pods)")
    print("="*70)
    
    # Show all pod info
    for i, pod in enumerate(pods):
        print(f"\nPod {i+1}: {pod['hardware']['gpu']}")
        print(f"  Timestamp: {pod['timestamp']}")
        print(f"  PyTorch: {pod['hardware']['pytorch']}")
        print(f"  CUDA: {pod['hardware']['cuda']}")
        print(f"  Hostname: {pod['hardware'].get('hostname', 'N/A')}")
        print(f"  Container ID: {pod['hardware'].get('container_id', 'N/A')}")
    
    # Check version consistency
    print("\n" + "="*70)
    print("SETUP CONSISTENCY CHECK")
    print("="*70)
    
    pytorch_versions = [pod['hardware']['pytorch'] for pod in pods]
    cuda_versions = [pod['hardware']['cuda'] for pod in pods]
    gpu_models = [pod['hardware']['gpu'] for pod in pods]
    hostnames = [pod['hardware'].get('hostname', 'N/A') for pod in pods]
    container_ids = [pod['hardware'].get('container_id', 'N/A') for pod in pods]
    
    pytorch_consistent = len(set(pytorch_versions)) == 1
    cuda_consistent = len(set(cuda_versions)) == 1
    gpu_consistent = len(set(gpu_models)) == 1
    hostname_unique = len(set(hostnames)) == len(hostnames)
    container_unique = len(set(container_ids)) == len(container_ids)
    
    if pytorch_consistent:
        print(f"✓ PyTorch version consistent: {pytorch_versions[0]}")
    else:
        print(f"⚠ PyTorch versions differ: {set(pytorch_versions)}")
    
    if cuda_consistent:
        print(f"✓ CUDA version consistent: {cuda_versions[0]}")
    else:
        print(f"⚠ CUDA versions differ: {set(cuda_versions)}")
    
    if gpu_consistent:
        print(f"✓ GPU model consistent: {gpu_models[0]}")
    else:
        print(f"⚠ GPU models differ: {len(set(gpu_models))} different types")
    
    if hostname_unique:
        print(f"✓ Hostnames unique: {len(set(hostnames))} different hosts")
    else:
        print(f"⚠ Hostnames NOT unique - may be same physical machine!")
        print(f"  Hostnames: {hostnames}")
    
    if container_unique:
        print(f"✓ Container IDs unique: All different containers")
    else:
        print(f"⚠ Container IDs NOT unique")
        print(f"  Container IDs: {container_ids}")
    
    # Get batch sizes
    batch_sizes = pods[0]['config']['batch_sizes']
    
    # Pairwise cross-pod comparisons
    print("\n" + "="*70)
    print("PAIRWISE CROSS-POD L2 DISTANCES")
    print("="*70)
    
    cross_pod_distances = {bs: [] for bs in batch_sizes}
    
    for bs in batch_sizes:
        print(f"\nBatch size {bs}:")
        key = f"batch_size_{bs}"
        
        for i, j in combinations(range(n_pods), 2):
            # Compare first repetition from each pod
            act_i = np.array(pods[i]['raw_activations'][key][0])
            act_j = np.array(pods[j]['raw_activations'][key][0])
            
            l2_dist = np.linalg.norm(act_i - act_j)
            rel_diff = l2_dist / np.linalg.norm(act_i)
            
            cross_pod_distances[bs].append(l2_dist)
            
            print(f"  Pod {i+1} vs Pod {j+1}: L2={l2_dist:.6f}, relative={rel_diff:.6f}")
    
    # Within-pod variance (should be ~0)
    print("\n" + "="*70)
    print("WITHIN-POD VARIANCE (rep 0 vs rep 1)")
    print("="*70)
    
    for i, pod in enumerate(pods):
        print(f"\nPod {i+1}:")
        for bs in batch_sizes:
            key = f"batch_size_{bs}"
            
            act_0 = np.array(pod['raw_activations'][key][0])
            act_1 = np.array(pod['raw_activations'][key][1])
            
            l2_dist = np.linalg.norm(act_0 - act_1)
            
            print(f"  bs={bs}: L2={l2_dist:.6f}")
    
    # Systematic deviations (within each pod)
    print("\n" + "="*70)
    print("SYSTEMATIC DEVIATIONS (bs1 vs bs2, within each pod)")
    print("="*70)
    
    for i, pod in enumerate(pods):
        print(f"\nPod {i+1}:")
        for bs1, bs2 in [(batch_sizes[0], batch_sizes[1])]:
            key1 = f"batch_size_{bs1}"
            key2 = f"batch_size_{bs2}"
            
            act1_mean = np.array(pod['raw_activations'][key1]).mean(axis=0)
            act2_mean = np.array(pod['raw_activations'][key2]).mean(axis=0)
            
            l2_dist = np.linalg.norm(act1_mean - act2_mean)
            
            print(f"  bs{bs1} vs bs{bs2}: L2={l2_dist:.6f}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    for bs in batch_sizes:
        distances = cross_pod_distances[bs]
        print(f"\nBatch size {bs}:")
        print(f"  Cross-pod L2 - mean: {np.mean(distances):.6f}")
        print(f"  Cross-pod L2 - std:  {np.std(distances):.6f}")
        print(f"  Cross-pod L2 - max:  {np.max(distances):.6f}")
    
    # Verification viability
    print("\n" + "="*70)
    print("VERIFICATION VIABILITY")
    print("="*70)
    
    max_cross_pod = max(np.max(cross_pod_distances[bs]) for bs in batch_sizes)
    min_systematic = pods[0]['systematic_deviations']['bs1_vs_bs2']
    
    print(f"Max cross-pod noise: {max_cross_pod:.6f}")
    print(f"Min systematic deviation: {min_systematic:.6f}")
    print(f"Signal-to-noise ratio: {min_systematic / max_cross_pod:.2f}x")
    
    if max_cross_pod < 0.01:
        print("\n✓ EXCELLENT: Cross-pod reproducibility is near-perfect")
    elif max_cross_pod < min_systematic / 2:
        print("\n✓ GOOD: Cross-pod noise << systematic deviations")
        print("  Verification highly viable")
    elif max_cross_pod < min_systematic:
        print("\n⚠ MARGINAL: Cross-pod noise detectable but < signal")
        print("  Verification possible with care")
    else:
        print("\n✗ PROBLEMATIC: Cross-pod noise ~ systematic deviations")
        print("  May need determinism flags")

if __name__ == "__main__":
    import os
    import glob
    
    # Auto-detect JSON files if no arguments provided
    if len(sys.argv) < 2:
        json_files = glob.glob("*.json")
        
        # Group by model size
        models_30b = [f for f in json_files if '30b' in f.lower()]
        models_7b = [f for f in json_files if '7b' in f.lower()]
        
        print(f"Found {len(models_30b)} 30B experiments")
        print(f"Found {len(models_7b)} 7B experiments\n")
        
        # Compare 30B models
        if len(models_30b) >= 2:
            print("="*70)
            print("COMPARING 30B MODELS")
            print("="*70)
            compare_all_pods(models_30b)
        
        # Compare 7B models
        if len(models_7b) >= 2:
            print("\n\n")
            print("="*70)
            print("COMPARING 7B MODELS")
            print("="*70)
            compare_all_pods(models_7b)
            
    else:
        compare_all_pods(sys.argv[1:])