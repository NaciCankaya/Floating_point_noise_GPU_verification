# **Mega-Plan: Floating-Point Forensics Ablation Experiments**

**Purpose:** Systematic testing of detection capabilities across hardware platforms (A100, H100) for various inference setup modifications.

**Research Question:** For each variable that affects inference execution, can we detect it via floating-point forensics (activation/key/logprob differences)? How does detectability compare to cross-hardware baseline deviation?

---

## **Executive Summary**

**Total Experiments:** 8  
 - **Hardware Platforms:** A100-80GB, H100  
 - **Model:** Qwen3-30B-A3B-AWQ-Int4 (\~15GB, MoE architecture) QuixiAI/Qwen3-30B-A3B-AWQ  
 - **Sequence Configuration:** \~6k token prompt (prefill)l \+ 30 tokens decode  
 - **Repetitions per Config:** 3 (establishes within-setup reproducibility)  
 - **Pod Requirements:** 4 instances: 

(\[(1 instance for experiments 1 to 6 with single GPU) \+ (1 instance for experiments 7 and 8 using four SXM GPUs)\]  2, since we are doing this for both A100 SXM and H100 SXM, respectively

For most of the experiments seen here, I already have implementations ready in my repo. But they do not fit the standardized format used here, with identical model, prompt, and json format. This is why we are repeating them in one go. But still, if you are not sure what to do, look into the repo.

### **Experiment 0: Reference Baseline**

**Configuration (all experiments reuse this as baseline):**

* `batch_size`: 1  
* `compile`: False  
* `quant_method`: AWQ  
* `attention_impl`: flash\_attention\_2  
* `concurrent_work`: False (default CUDA stream)  
* `cuda_version`: Pod default (\~12.8)  
* `tp_size`: 1  
* `ep_size`: 1

---

| Experiment | Variable Tested | num configs |
| ----- | ----- | :---: |
| 0\. Reference | None.  | 1 |
| 1\. Batch Size | batch\_size: 2, 4 | 2 |
| 2\. Compilation | compile: True | 1 |
| 3\. Quantization | quant\_method: gptq, bnbQuixiAI/Qwen3-30B-A3B-AWQunsloth/Qwen3-30B-A3B-bnb-4bit | 2 |
| 4\. Attention | attention\_impl: eager | 1 |
| 5\. Concurrent Streams | concurrent\_work: True | 1 |
| 6\. CUDA Version | cuda\_version: 11.8, 12.1 (or cu118, cu 121\) | 2 |
| 7\. Tensor Parallelism | tp\_size: 2, 4 | 2 |
| 8\. Expert Parallelism | ep\_size: 2, 4 | 2 |

Regarding implementation, you can take heavy inspiration from the experiment files in my repo. BUT: STICK TO THE TEMPLATE. Experiments 1-5 may be a single python script, if you prefer. Choose whatever architecture makes the most sense to you, though.

## **Experimental Variables**

---

## **Signal Extraction Specification**

### **Layers to Extract**

* **Layers:** 1, 2, 4, 12, last (beware indexing)  
* **Rationale:** Sample across depth for propagation analysis

### **Token Positions**

* **Positions:** \-3, \-2, \-1 (final three tokens at each decode step)  
* **Total measurements per sequence:** 30 decode steps × 3 positions \= 90 timepoints. Only store the measurement for the last repetition in json IF all three repetitions were bit-identical.

### **Signal Types**

1. **Hidden States:** 3584-dimensional vectors (5 layers × 3 positions)  
2. **Key Vectors:** 512-dimensional GQA keys (concatenated) (5 layers × 3 positions)  
3. **Logprobs:** Top-10 token probabilities (3 positions)

---

## **Uniform JSON Schema**

All 8 experiment files follow this identical structure:
```
{  
  "experiment\_metadata": {  
    "experiment\_type": "batch\_size",  
    "variable\_tested": "batch\_size",  
    "model": "Qwen3-30B-A3B-GPTQ-Int4",  
    "model\_size": "30B",  
    "architecture": "MoE",  
    "sequence\_length": 8192,  
    "decode\_steps": 30,  
    "extraction\_config": {  
      "layers": \[1, 2, 4, 12, 39\],  
      "positions": \[-3, \-2, \-1\],  
      "hidden\_dim": 3584,  
      "key\_dim": 512,  
      "top\_k\_logprobs": 10  
    },  
    "date\_created": "YYYY-MM-DD HH:MM:SS"  
  },  
    
  "configurations": \[  
    {  
      "config\_id": "A100\_bs1",  
      "hardware": "A100-80GB",  
      "provider": "RunPod",  
      "variable\_value": 1,  
      "cuda\_version": "12.8",  
      "torch\_version": "2.x.x",  
      "transformers\_version": "4.x.x",  
      "flash\_attn\_version": "2.x.x",  
      "python\_version": "3.x",  
      "fixed\_params": {  
        "compile": false,  
        "attention\_impl": "flash\_attention\_2",  
        "quantization": "gptq-int4",  
        "tp\_size": 1,  
        "ep\_size": 1,  
        "concurrent\_streams": false  
      }  
    }  
  \],  
    
  "runs": \[  
    {  
      "config\_id": "A100\_bs1",  
      "rep\_id": 0,  
      "timestamp": "YYYY-MM-DD HH:MM:SS",  
      "runtime\_seconds": 123.45,  
      "prompt\_text": "...",  
        
      "decode\_steps": \[  
        {  
          "step": 0,  
          "token\_id": 12345,  
          "token\_text": "Hello",  
            
          "hidden\_states": {  
            "layer\_1": {  
              "pos\_-3": \[3584 floats\],  
              "pos\_-2": \[3584 floats\],  
              "pos\_-1": \[3584 floats\]  
            },  
            "layer\_2": {},  
            "layer\_4": {},  
            "layer\_12": {},  
            "layer\_39": {}  
          },  
            
          "key\_vectors": {  
            "layer\_1": {  
              "pos\_-3": \[512 floats\],  
              "pos\_-2": \[512 floats\],  
              "pos\_-1": \[512 floats\]  
            },  
            "layer\_2": {},  
            "layer\_4": {},  
            "layer\_12": {},  
            "layer\_39": {}  
          },  
            
          "logprobs": {  
            "pos\_-3": {  
              "token\_ids": \[10 ints\],  
              "log\_probs": \[10 floats\]  
            },  
            "pos\_-2": {},  
            "pos\_-1": {}  
          }  
        }  
      \]  
    }  
  \]  
}
```
**Key Schema Features:**

* Self-documenting: All versions and configurations embedded  
* Uniform across experiments: Single comparison script will work for all experiment jsons  
* Linkable: config\_id, variable\_tested connect runs to configurations  
* Complete: Includes metadata for reproducibility verification

## **Pod Execution Plan**

### **Pod 1: Single GPU (A100-SXM or H100-SXM)**

**Experiments:** 0, 1, 2, 3, 4, 5, 6

**Execution Sequence:**

1. **Setup** (\~20 min)  
   * **Setup:** Install dependencies (just test them by running the experiment code and install whatever the error print wants you to pip install)  
   * **Experiment 1 \- Batch Size:** Download base model: `QuixiAI/Qwen3-30B-A3B-AWQ`  
   * Download quantization variants: GPTQ, BNB versions  
   * Verify installations and model loading  
2. **Experiment 0: Reference Baseline**  
   * Run 3 reps with baseline config  
   * Verify bit-exact reproducibility  
   * Save to `reference_baseline.json` (for documentation)  
   * This data will be reused as baseline for Experiments 1-6  
3. **Experiment 1: Batch Size**  
   * Load AWQ model  
   * Run bs1, bs2, bs4 configs. Make sure to use distinct token sequences, all \~6k tokens long, same as experiment 0\. Beware padding tokens to not measure the wrong thing.   
   * Run bs=2 (3 reps)  
   * Run bs=4 (3 reps)  
   * Fetch json from Experiment 0 bs=1 data as baseline  
   * Save all three’s results to `batch_size_experiment.json`  
4. **Experiment 2: Compilation**  
   * Load AWQ model with `torch.compile(model)`  
   * Run compile=True (3 reps)  
   * Reuse Experiment 0 compile=False data as baseline  
   * Save to `compile_experiment.json`  
5. **Experiment 3: Quantization**  
   * Load GPTQ model variant (3 reps)  
   * Load BNB model variant (3 reps)  
   * Reuse Experiment 0 AWQ data as baseline  
   * Save to `quantization_experiment.json`  
   * Delete the other two quants, keep awq  
6. **Experiment 4: Attention**  
   * Load AWQ model with `attn_implementation="eager"`  
   * Run eager attention (3 reps)  
   * Reuse Experiment 0 FA2 data as baseline  
   * Save to `attention_experiment.json`  
7. **Experiment 5: Concurrent Streams**  
   * Load AWQ model  
   * Launch concurrent workload on stream 1 (\~50% GPU util)  
   * Run inference on stream 0 (3 reps)  
   * make sure both overlap (check out the profiler script in my repo)  
   * Reuse Experiment 0 single-stream data as baseline  
   * Save to `concurrent_streams_experiment.json`  
8. **CUDA Version Switch to cu118** (\~30 min)

```bash

  *\# Install CUDA 11.8*

   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local\_installers/cuda\_11.8.0\_520.61.05\_linux.run

   sudo sh cuda\_11.8.0\_520.61.05\_linux.run \--silent \--toolkit

   export PATH\=/usr/local/cuda-11.8/bin:$PATH

   export LD\_LIBRARY\_PATH=/usr/local/cuda-11.8/lib64:$LD\_LIBRARY\_PATH

   

   *\# Reinstall PyTorch for cu118*

   pip uninstall torch \-y

   pip install torch torchvision torchaudio \--index-url https://download.pytorch.org/whl/cu118

   

   *\# Verify CUDA version*

   python \-c "import torch; print(torch.version.cuda)"
```

9. **Experiment 6a: CUDA 11.8**  
   * Load AWQ model  
   * Run cu118 config (3 reps)  
10. **CUDA Version Switch to cu121**

```bash

   *\# Install CUDA 12.1*

    wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local\_installers/cuda\_12.1.0\_530.30.02\_linux.run

    sudo sh cuda\_12.1.0\_530.30.02\_linux.run \--silent \--toolkit

    export PATH\=/usr/local/cuda-12.1/bin:$PATH

    export LD\_LIBRARY\_PATH=/usr/local/cuda-12.1/lib64:$LD\_LIBRARY\_PATH

    

    *\# Reinstall PyTorch for cu121*

    pip uninstall torch \-y

    pip install torch torchvision torchaudio \--index-url https://download.pytorch.org/whl/cu121
```
11. **Experiment 6b: CUDA 12.1**  
    * Load AWQ model  
    * Run cu121 config (3 reps)  
    * Combine with cu118 data and Experiment 0 (cu128) as baseline  
    * Save to `cuda_version_experiment.json`

**Output Files:**

* `reference_baseline.json` (optional documentation)  
* `batch_size_experiment.json`  
* `compile_experiment.json`  
* `quantization_experiment.json`  
* `attention_experiment.json`  
* `concurrent_streams_experiment.json`  
* `cuda_version_experiment.json`

---

### **Pod 2: Multi-GPU (4×A100-SXM or 4×H100-SXM)**  **Experiments:** 7, 8

**Execution Sequence:**

1. **Setup** (\~30 min)  
   * Download model to shared storage (NVMe recommended)  
   * Verify multi-GPU communication (NCCL)  
2. **Experiment 7: Tensor Parallelism**  
   * **TP=2 Config:**  
     * Load model with `tensor_parallel_size=2`  
     * Run 3 reps  
   * **TP=4 Config:**  
     * Load model with `tensor_parallel_size=4`  
     * Run 3 reps  
   * **Include Experiment 0 TP=1 data** from Pod 1 as baseline  
   * Save to `tensor_parallel_experiment.json`  
3. **Experiment 8: Expert Parallelism**  
   * **EP=2 Config:**  
     * Load model with `expert_parallel_size=2`  
     * Run 3 reps  
   * **EP=4 Config:**  
     * Load model with `expert_parallel_size=4`  
     * Run 3 reps  
   * **Include Experiment 0 EP=1 data** from Pod 1 as baseline  
   * Save to `expert_parallel_experiment.json`

**Output Files:**

* `tensor_parallel_experiment.json`  
* `expert_parallel_experiment.json`

---

## **Experiment-Specific Details**

### **Experiment 0: Reference Baseline**

**Variable:** None (establishes baseline)  
 **Purpose:**

1. Verify bit-exact reproducibility within identical setups  
2. Like all experiments, run it on both A100 and H100.  
3. Provide reference configuration for all subsequent experiments  
4. Validate extraction pipeline and data collection

**Configuration:**

* All parameters at "default" values  
* Will be reused as baseline comparison for Experiments 1-6

**Runs:**

* 3 repetitions per hardware  
* Verify L2=0 across repetitions (bit-exact reproducibility)  
* signals only from 3rd repetition for storage

## **Python Code Structure**

experiments/  
├── common/  
│   ├── \_\_init\_\_.py  
│   ├── model\_loader.py      \# Load Qwen3 with various configs  
│   ├── extraction.py         \# Extract hidden states, keys, logprobs  
│   ├── runner.py             \# Run inference with extraction  
│   └── json\_writer.py        \# Write to experiment JSON format  
│   └── json\_reader.py        \# Used to fetch reference measurement and store all the variant’s data in \#fused json. Yes, this is redundant across experiments, but makes later comparisons simpler.  
├── ablation_cross_hardware/  
│   ├── exp0\_reference.py 
│   ├── exp1\_batch\_size.py  
│   ├── exp2\_compile.py  
│   ├── exp3\_quantization.py  
│   ├── exp4\_attention.py  
│   ├── exp5\_concurrent\_streams.py  
│   ├── exp6\_tensor\_parallel.py  
│   ├── exp7\_expert\_parallel.py  
│   └── exp8\_cuda\_version.py  
└── analysis/  
    └── compare\_experiments.py

## **Analysis Framework**

### **Single Script for All Experiments**

\# analysis/compare\_experiments.py
```python
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
```

## **Critical Implementation Notes**

### **1\. Reproducibility Verification**

Every experiment must verify bit-exact reproducibility within identical setups:

* Compare 3 reps within each config  
* If L2 \> 0, investigate non-determinism source  
* Document any non-deterministic behaviors, immediately stop and report back to me.

### **2\. CUDA Stream Isolation (Experiment 5\)**

\# Concurrent stream implementation  
```
stream0 \= torch.cuda.Stream()  \# Main inference  
stream1 \= torch.cuda.Stream()  \# Concurrent work

with torch.cuda.stream(stream0):  
    \# Run model inference here  
    outputs \= model.generate(...)

with torch.cuda.stream(stream1):  
    \# Run synthetic workload  
    \# Must be substantial: target \~50% GPU utilization  
    for \_ in range(large\_number):  
        dummy \= torch.randn(8192, 8192, device='cuda') @ torch.randn(8192, 8192, device='cuda')

torch.cuda.synchronize()
```

### **3\. Prompt Construction**

All experiments use identical prompt to ensure comparability, except batch\_size, where we add batch neighbours to the reference sequence0 (same as in exp0):

I recommend using Qwen’s standard chat template. Prompt should be a text pulled from a long pdf, cut to 8k token length. Same with batch neighbours, but use different pdfs. This text is simply combined with the prompt: “Provide a summary” or whatever.
Actually, just check out prompts.py from commons.

### **4\. Memory Management**

Between runs:

torch.cuda.empty\_cache()  
import gc  
gc.collect()

### **5\. Timing Verification**

While not primary metric, track runtime:
```python
import time  
start \= time.time()  
\# ... run inference ...  
runtime \= time.time() \- start
```
Include in JSON for timing forensics analysis. Average of three repetitions, and variance from average.

---

## **Validation Checklist**

Before running production experiments:

* \[ \] Model loads correctly on target hardware  
* \[ \] Extraction pipeline captures all required signals  
* \[ \] JSON schema validates against specification  
* \[ \] Bit-exact reproducibility confirmed (3 reps, L2=0)  
* \[ \] Prompt generates exactly 6k tokens from the pdf, plus the summarization request  
* \[ \] CUDA version switching works (Experiment 5\)  
* \[ \] Multi-GPU setup functional (Experiments 7-8)  
* \[ \] Storage capacity sufficient (when downloading three models, the pod’s disk space needs to suffice)
