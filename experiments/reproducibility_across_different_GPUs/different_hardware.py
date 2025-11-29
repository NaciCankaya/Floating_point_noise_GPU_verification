# compare_hardware_types.py
"""
Compare activations across different GPU architectures.
Reports raw L2 distances only.
"""
import json
import numpy as np
import sys
from pathlib import Path

def load_activation(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def compare_hardware_types(file_paths):
    """Compare different GPU architectures"""
    
    data = [load_activation(f) for f in file_paths]
    
    print("="*70)
    print("HARDWARE TYPE COMPARISON")
    print("="*70)
    
    # Verify identical setup
    models = set(d['model'] for d in data)
    dtypes = set(d['config']['dtype'] for d in data)
    prompt_tokens = set(d['config']['prompt_tokens'] for d in data)
    hidden_dims = set(d['config']['hidden_dim'] for d in data)
    
    print("\nSetup verification:")
    print(f"  Models: {models}")
    print(f"  Dtypes: {dtypes}")
    print(f"  Prompt tokens: {prompt_tokens}")
    print(f"  Hidden dims: {hidden_dims}")
    
    if len(models) > 1 or len(dtypes) > 1 or len(prompt_tokens) > 1 or len(hidden_dims) > 1:
        print("\nERROR: Setup mismatch - cannot compare")
        return
    
    # Group by GPU type
    gpu_types = {}
    for f, d in zip(file_paths, data):
        gpu = d['hardware']['gpu']
        if gpu not in gpu_types:
            gpu_types[gpu] = []
        gpu_types[gpu].append((f, d))
    
    print(f"\nGPU types found: {len(gpu_types)}")
    for gpu, files in gpu_types.items():
        print(f"  {gpu}: {len(files)} file(s)")
    
    # Get one representative from each GPU type
    representatives = {}
    for gpu, files in gpu_types.items():
        representatives[gpu] = files[0][1]
    
    # Compare all pairs
    print("\n" + "="*70)
    print("CROSS-HARDWARE L2 DISTANCES")
    print("="*70)
    
    batch_sizes = data[0]['config']['batch_sizes']
    gpu_names = list(representatives.keys())
    
    for bs in batch_sizes:
        print(f"\nBatch size {bs}:")
        key = f"batch_size_{bs}"
        
        for i in range(len(gpu_names)):
            for j in range(i+1, len(gpu_names)):
                gpu1, gpu2 = gpu_names[i], gpu_names[j]
                
                act1 = np.array(representatives[gpu1]['raw_activations'][key][0])
                act2 = np.array(representatives[gpu2]['raw_activations'][key][0])
                
                l2_dist = np.linalg.norm(act1 - act2)
                
                print(f"  {gpu1} vs {gpu2}")
                print(f"    L2 distance: {l2_dist:.6f}")
    
    # Statistical noise within each hardware
    print("\n" + "="*70)
    print("STATISTICAL NOISE (within-hardware, across runs)")
    print("="*70)
    
    for gpu, d in representatives.items():
        print(f"\n{gpu}:")
        print(f"  Reported statistical noise: {d['statistical_noise']}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        compare_hardware_types(sys.argv[1:])
    else:
        # Auto-detect
        files = list(Path('.').glob('*7b*.json'))
        if len(files) >= 2:
            compare_hardware_types([str(f) for f in files])
        else:
            print("Usage: python compare_hardware_types.py <file1.json> <file2.json> ...")