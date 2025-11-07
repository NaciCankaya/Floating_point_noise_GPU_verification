# Cross-Hardware Verification: Second CUDA Stream Experiment

## Overview

This experiment combines two previously separate investigations:
1. **Cross-hardware verification** (from `batch_size_A100_H100`)
2. **Parallel CUDA stream effects** (from `separate_CUDA_stream`)

The key research question: **Can we detect parallel CUDA stream workloads even when verification hardware doesn't match the production hardware?**

## Motivation

From Baker et al.'s verification framework:
- **Subgoal 1A**: Verify declared ML compute uses are accurate
- **Subgoal 2A**: Verify no large, undeclared uses of declared clusters exist

Even with timing verification ensuring no idle time, malicious actors could run **parallel hidden workloads on separate CUDA streams**. This experiment tests whether floating-point forensics can detect such evasion across different GPU architectures.

## Previous Findings

### From `separate_CUDA_stream`
- ✓ Parallel streams are detectable at high GPU utilization
- ✓ Produces statistical noise (non-reproducible across runs)
- ✗ Not detectable at low utilization (separate SMs, no contention)

### From `batch_size_A100_H100`
- ✓ Cross-hardware verification works (don't need matched hardware)
- ✓ Systematic deviations distinguishable from legitimate hardware variation
- ✓ Batch size changes produce 81× larger signal than hardware mismatch

## Experiment Design

### Test Matrix

For each GPU type (A100, H100):
- **Baseline**: Single CUDA stream (normal inference)
- **Light concurrent**: Parallel stream with short prompt ("The capital of France is")
- **Heavy concurrent**: Parallel stream with long prompt (50× repeated text)

### Methodology

1. **Sparse key vector extraction**
   - Every 5th token position from end
   - GQA key heads: 512 dims (vs 3584 for hidden states)
   - Memory efficient: ~25 MB per file

2. **Reproducibility testing**
   - 3 repetitions per condition
   - Expected: Baseline reproducible, parallel streams show noise

3. **10 diverse test sequences**
   - Technical, narrative, code, mixed, repetitive, scientific
   - 186-208 tokens each
   - Ensures robustness across content types

4. **5 sampled layers**
   - Layers [1, 4, 10, 18, 28]
   - Track error propagation through model depth

## Comparison Dimensions

### Analysis 1: Reproducibility Check
- Verify baseline conditions are bit-exact
- Confirm parallel streams produce statistical noise
- Per-hardware, per-condition

### Analysis 2: Cross-Hardware (Same Condition)
- A100/baseline vs H100/baseline
- A100/light vs H100/light
- A100/heavy vs H100/heavy
- **Establishes legitimate hardware variation baseline**

### Analysis 3: Within-Hardware (Different Conditions)
- A100: baseline vs light vs heavy
- H100: baseline vs light vs heavy
- **Tests pure CUDA stream effect on matched hardware**

### Analysis 4: Cross-Hardware + Different Conditions
- All combinations: 2 hardware × 3 conditions = 9 comparisons
- **Tests worst-case forensic scenario**

## Expected Results

### Hypothesis 1: Parallel streams are detectable on same hardware
- Baseline vs light/heavy should show L2 >> cross-hardware baseline
- Statistical noise confirms non-deterministic behavior
- **Result from `separate_CUDA_stream` suggests this works**

### Hypothesis 2: Detection works across hardware
- (A100/baseline vs H100/heavy) > (A100/baseline vs H100/baseline)
- Parallel stream signal should exceed hardware variation
- **This is the novel contribution of this experiment**

### Hypothesis 3: Hardware variation doesn't mask parallel streams
- Ratio of (parallel effect) / (hardware baseline) > 2.0
- Detection threshold: baseline_mean + 2σ
- If ratio > 2.0, forensically distinguishable

## Detection Verdict Criteria

**✓ DETECTION WORKS** if:
- All parallel stream scenarios show L2 > 2× baseline
- Both light and heavy workloads are detected
- Detection succeeds even with hardware mismatch

**⚠ PARTIAL DETECTION** if:
- Some but not all parallel scenarios detected
- Light workload evades but heavy doesn't
- Requires hardware matching for reliable detection

**✗ DETECTION FAILS** if:
- Parallel streams indistinguishable from hardware variation
- FP forensics insufficient for this threat model

## Practical Implications

### If detection works:
- Verification cluster can use different hardware than production
- Don't need expensive matched GPUs (e.g., H100 with CC)
- Parallel hidden workloads are forensically detectable
- Supports Baker et al. Subgoal 2A (comprehensive compute accounting)

### If detection requires hardware matching:
- Verification clusters must mirror production hardware
- Higher cost and complexity for governance infrastructure
- Still validates core FP forensics approach

### If detection fails:
- Parallel streams are a blind spot
- Would need to rely on other verification layers:
  - Built-in chip security features
  - Physical monitoring devices
  - Timing anomaly detection
  - Personnel-based mechanisms

## Files

- `collect_keys.ipynb`: Main experiment notebook (run on each GPU)
- `compare_streams.py`: Analysis script comparing cross-hardware results
- `README.md`: This file

## Usage

### Step 1: Collect data on first hardware (e.g., A100)
```bash
# Open collect_keys.ipynb
# Auto-detects hardware as 'a100'
# Tests baseline, light_concurrent, heavy_concurrent
# Outputs: a100_cuda_stream_TIMESTAMP.json (~75 MB)
```

### Step 2: Collect data on second hardware (e.g., H100)
```bash
# Open collect_keys.ipynb on H100 system
# Auto-detects hardware as 'h100'
# Tests same conditions
# Outputs: h100_cuda_stream_TIMESTAMP.json
```

### Step 3: Compare results
```bash
python compare_streams.py a100_cuda_stream_*.json h100_cuda_stream_*.json

# Or from notebook:
%run compare_streams.py
# (auto-detects 2 most recent files)
```

### Expected Output
```
CROSS-HARDWARE CUDA STREAM ANALYSIS
====================================================================

Hardware 1: a100 (NVIDIA A100 80GB PCIe)
Hardware 2: h100 (NVIDIA H100 80GB HBM3)

Common sequences: 10
Conditions tested: ['baseline', 'heavy_concurrent', 'light_concurrent']
Layers to analyze: ['layer_1', 'layer_4', 'layer_10', 'layer_18', 'layer_28']

====================================================================
ANALYSIS 1: REPRODUCIBILITY CHECK
====================================================================

--- A100 ---
  baseline: ✓ 10/10 sequences reproducible (100%)
  light_concurrent: ⚠ 2/10 sequences reproducible (20%)
  heavy_concurrent: ⚠ 0/10 sequences reproducible (0%)

--- H100 ---
  baseline: ✓ 10/10 sequences reproducible (100%)
  light_concurrent: ⚠ 3/10 sequences reproducible (30%)
  heavy_concurrent: ⚠ 1/10 sequences reproducible (10%)

====================================================================
ANALYSIS 2: CROSS-HARDWARE (SAME CONDITION)
====================================================================

--- BASELINE ---
  layer_28: L2 = 0.2845 ± 0.0123 (range: [0.2654, 0.3102])

--- LIGHT_CONCURRENT ---
  layer_28: L2 = 0.3156 ± 0.0234 (range: [0.2812, 0.3567])

--- HEAVY_CONCURRENT ---
  layer_28: L2 = 0.3489 ± 0.0298 (range: [0.3001, 0.3890])

====================================================================
FORENSIC VERDICT
====================================================================

✓ PARALLEL STREAMS ARE DETECTABLE
  All 12/12 parallel stream scenarios detected
  FP forensics successfully identifies concurrent workloads
  Hardware matching NOT required for detection
```

## Technical Notes

### Memory Optimization
- Sparse sampling reduces file size by ~95%
- Key vectors (512 dims) vs hidden states (3584 dims)
- ~25 MB per JSON vs ~430 MB with dense sampling

### CUDA Stream Implementation
- Uses `torch.cuda.Stream()` for separate stream
- Threading to run concurrent workload continuously
- Synchronized start/stop to minimize variability

### Reproducibility Expectations
- **Baseline**: Should be bit-exact (100% reproducible)
- **Light concurrent**: Moderate noise (20-30% reproducible)
- **Heavy concurrent**: High noise (0-10% reproducible)
- Statistical noise indicates scheduling race conditions

## Related Experiments

- `experiments/separate_CUDA_stream`: Original parallel stream study
- `experiments/cross_hardware_verification/batch_size_A100_H100`: Cross-hardware baseline
- `experiments/batch_experiments`: Batch size forensics (81× signal)
- `experiments/across_different_GPUs`: Hardware variation baseline

## Research Context

This experiment directly addresses the threat model where:
1. Datacenter claims to run only legitimate ML workloads
2. Timing verification ensures no idle periods
3. **Hidden threat**: Parallel CUDA streams running undeclared models
4. **Defense**: Cryptographic commitments + FP forensics

The combination of hardware independence and parallel stream detection would be a significant result for practical AI governance infrastructure.

## Status

**Experiment designed, ready to run.**

Next steps:
1. Run `collect_keys.ipynb` on A100
2. Run `collect_keys.ipynb` on H100
3. Run `compare_streams.py` to analyze results
4. Update this README with actual findings

---

**Last Updated**: November 2025
