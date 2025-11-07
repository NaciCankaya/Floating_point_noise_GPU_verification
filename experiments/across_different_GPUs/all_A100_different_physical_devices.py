# compare_a100_devices.py
"""
Compare activations across different physical A100 devices.
Auto-detects A100 experiment files and reports raw L2 distances.
"""
import json
import numpy as np
import sys
from pathlib import Path
from itertools import combinations

def load_activation(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def compare_a100_devices(file_paths=None):
    """Compare A100 devices"""
    
    # Auto-detect A100 files if none provided
    if file_paths is None:
        file_paths = list(Path('.').glob('NVIDIA_A100*.json'))
        if not file_paths:
            file_paths = list(Path('.').glob('*A100*.json'))
    
    if len(file_paths) < 2:
        print("ERROR: Need at least 2 A100 experiment files")
        print("Usage: python compare_a100_devices.py [file1.json file2.json ...]")
        sys.exit(1)
    
    # Load data
    data = [(str(f), load_activation(f)) for f in file_paths]
    
    print("="*70)
    print("A100 CROSS-DEVICE COMPARISON")
    print("="*70)
    print(f"\nFiles: {len(data)}")
    
    # Verify all A100
    non_a100 = [d[0] for d in data if 'A100' not in d[1]['hardware']['gpu']]
    if non_a100:
        print(f"\nERROR: Non-A100 files detected:")
        for f in non_a100:
            print(f"  {f}")
        sys.exit(1)
    
    # Verify identical setup
    models = set(d[1]['model'] for d in data)
    dtypes = set(d[1]['config']['dtype'] for d in data)
    prompt_tokens = set(d[1]['config']['prompt_tokens'] for d in data)
    hidden_dims = set(d[1]['config']['hidden_dim'] for d in data)
    pytorch_versions = set(d[1]['hardware']['pytorch'] for d in data)
    cuda_versions = set(d[1]['hardware']['cuda'] for d in data)
    
    print("\nSetup verification:")
    print(f"  Models: {models}")
    print(f"  Dtypes: {dtypes}")
    print(f"  Prompt tokens: {prompt_tokens}")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  PyTorch: {pytorch_versions}")
    print(f"  CUDA: {cuda_versions}")
    
    if (len(models) > 1 or len(dtypes) > 1 or len(prompt_tokens) > 1 or 
        len(hidden_dims) > 1 or len(pytorch_versions) > 1 or len(cuda_versions) > 1):
        print("\nERROR: Setup mismatch - cannot compare")
        sys.exit(1)
    
    # Device identification
    print("\n" + "="*70)
    print("DEVICE IDENTIFICATION")
    print("="*70)
    
    hostnames = []
    containers = []
    for i, (f, d) in enumerate(data):
        hostname = d['hardware'].get('hostname', 'N/A')
        container = d['hardware'].get('container_id', 'N/A')
        hostnames.append(hostname)
        containers.append(container)
        
        print(f"\nDevice {i+1}:")
        print(f"  File: {Path(f).name}")
        print(f"  GPU: {d['hardware']['gpu']}")
        print(f"  Hostname: {hostname}")
        print(f"  Container: {container}")
        print(f"  Timestamp: {d['timestamp']}")
    
    # Check uniqueness
    unique_hosts = len(set(h for h in hostnames if h != 'N/A'))
    unique_containers = len(set(c for c in containers if c != 'N/A'))
    
    print(f"\nUnique hostnames: {unique_hosts}")
    print(f"Unique containers: {unique_containers}")
    
    if unique_hosts < len(data) and 'N/A' not in hostnames:
        print("WARNING: Not all hostnames unique - may include same device")
    
    # Cross-device L2 distances
    print("\n" + "="*70)
    print("CROSS-DEVICE L2 DISTANCES")
    print("="*70)
    
    batch_sizes = data[0][1]['config']['batch_sizes']
    
    for bs in batch_sizes:
        print(f"\nBatch size {bs}:")
        key = f"batch_size_{bs}"
        
        distances = []
        for (i, (f1, d1)), (j, (f2, d2)) in combinations(enumerate(data), 2):
            act1 = np.array(d1['raw_activations'][key][0])
            act2 = np.array(d2['raw_activations'][key][0])
            
            l2_dist = np.linalg.norm(act1 - act2)
            distances.append(l2_dist)
            
            print(f"  Device {i+1} vs Device {j+1}: {l2_dist:.6f}")
        
        print(f"  Mean: {np.mean(distances):.6f}")
        print(f"  Max:  {np.max(distances):.6f}")
        print(f"  Min:  {np.min(distances):.6f}")
    
    # Within-device statistical noise
    print("\n" + "="*70)
    print("WITHIN-DEVICE STATISTICAL NOISE")
    print("="*70)
    
    for i, (f, d) in enumerate(data):
        print(f"\nDevice {i+1}:")
        for bs in batch_sizes:
            noise = d['statistical_noise'][f'batch_size_{bs}']
            print(f"  Batch size {bs}: mean={noise['mean']:.6f}, std={noise['std']:.6f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        compare_a100_devices([Path(f) for f in sys.argv[1:]])
    else:
        compare_a100_devices()