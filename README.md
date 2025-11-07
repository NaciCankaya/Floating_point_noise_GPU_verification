# Inference Verification via Floating-Point Forensics

**Research Goal:** Determine if malicious datacenters can hide unauthorized computations by examining whether different inference setups produce distinguishable activation patterns.

## Motivation

In future AI governance scenarios, datacenters may be required to report all computations performed on their hardware. A malicious operator could evade monitoring by:
1. Improving the efficiency of declared computations (freeing up "dark capacity")
2. Running unauthorized work in the freed capacity

**Our approach:** Use floating-point forensics to verify that claimed outputs were produced by the claimed computational setup. If different setups produce distinguishable activation patterns, verifiers can detect evasion attempts.

## Core Concept

GPU computations exhibit two types of variation:

- **Statistical noise:** Random, non-reproducible variation from scheduling, atomic operations, etc.
- **Systematic deviation:** Reproducible differences from computational setup choices (batch size, precision, kernels, parallelism)

**Key insight:** If statistical noise is low (or can be characterized), systematic deviations become detectable and attributable to specific setup changes.

## Verification Strategy

A verifier with access to:
- Claimed model weights
- Input prompts  
- Output activations/logits
- Claimed inference setup

Can attempt reproduction on a trusted reference system. Deviations between claimed and reproduced outputs indicate setup differences, potentially revealing:
- Hidden batch size increases
- Parallel unauthorized work
- Different model architectures
- Computational optimizations

## Experimental Methodology

### Setup
- **Model:** Qwen2.5-7B-Instruct (BF16)
- **Hardware:** NVIDIA A100 80GB PCIe, H100 NVL
- **Extraction:** Key vectors from attention layers at multiple positions
- **Layers sampled:** 1, 4, 10, 18, 28 (capturing early, middle, and late representations)
- **Measurement:** L2 distance between key vectors

### Baseline Establishment
Cross-hardware reproducibility baseline (A100 ↔ H100, identical setup):
- **Layer 1:** μ = 0.196, σ = 0.212
- **Layer 4:** μ = 0.210, σ = 0.068  
- **Layer 10:** μ = 0.235, σ = 0.038
- **Layer 18:** μ = 0.335, σ = 0.061
- **Layer 28:** μ = 0.425, σ = 0.310

**Key finding:** Hardware differences create order-of-magnitude 10^-1 deviations, but these are consistent and can serve as a baseline for detecting additional changes.

## Experiments and Results

### 1. Repetition Reproducibility ✓
**Test:** Multiple runs of identical setup  
**Result:** Bit-exact reproduction on single CUDA stream  
**Implication:** Production inference (single stream) is deterministic

### 2. Hardware Variation ✓
**Test:** Same setup on different GPUs
- Same GPU type, different device
- Same GPU type, different cloud provider  
- Different GPU architectures (A100 vs H100)

**Result:** Systematic deviations detected (~0.2-0.4 range)  
**Implication:** Hardware differences are detectable but small enough to serve as baseline

### 3. Batch Size Changes ✓ **HIGHLY DETECTABLE**
**Test:** Cross-hardware with different batch sizes (bs1, bs2, bs4)  
**Result:**
- Baseline (same batch): 0.32-0.62 (order 10^-1)
- Batch mismatch: ~28 (order 10^1)  
- **Ratio: 81× larger than baseline**
- **Detectability: 89% of comparisons exceed 1.5× threshold**

**Implication:** Batch size changes create massive activation differences. Malicious prover **cannot** hide batch size manipulation cross-hardware.

### 4. Batch Composition (Fixed Size) ✓
**Test:** Different sequences in batch positions 1-3, same reference at position 0  
**Result:** Zero deviation in reference sequence activations  
**Implication:** Batch composition doesn't affect individual sequence processing (as expected from causal attention)

### 5. Compilation (torch.compile) ✓
**Test:** Compiled vs eager mode  
**Result:** Systematic deviations detected  
**Implication:** Compilation optimization detectable

### 6. Attention Implementation (Eager vs FlashAttention2) ✓
**Test:** Different attention kernels  
**Result:** Systematic deviations detected  
**Implication:** Kernel choice is detectable

### 7. CUDA Version Changes ✓
**Test:** Different CUDA toolkit versions  
**Result:** Systematic deviations detected  
**Implication:** Minor software updates can be detected (may need allowlisting)

### 8. Pipeline Parallelism Ranks ✓
**Test:** Different PP ranks computing same layer  
**Result:** Small systematic deviations  
**Implication:** PP rank location creates detectable differences

### 9. Parallel CUDA Streams ⚠️ **CONTEXT-DEPENDENT**
**Test:** Concurrent work on separate CUDA stream  
**Result:** 
- **Low GPU utilization:** Bit-exact reproduction (streams on different SMs)
- **High GPU utilization:** Statistical noise detected (scheduling competition)

**Implication:** Detection depends on utilization level. At realistic production utilization (~60-80%), separate streams may execute independently → undetectable.

### 10. Multiple Concurrent Streams (Detailed)
**Experiment:** Realistic parallel work at different intensities

**Light concurrent work:**
- Layer 1, 4, 10: Bit-exact (L2 = 0)
- Layer 18: L2 = 96.8  
- Layer 28: L2 = 653.6

**Heavy concurrent work:**
- All layers affected
- Layer 1: L2 = 9.8
- Layer 28: L2 = 508.0

**Implication:** Heavy concurrent loads create detectable interference, but light loads may only affect deeper layers. Verification must check all layers, not just final output.

## Summary of Detectability

### Highly Detectable (Systematic Deviations > 10× Baseline)
✓ **Batch size changes** - 81× baseline  
✓ **Different attention kernels** (FA2 vs Eager)  
✓ **Compilation mode** (torch.compile)  
✓ **Different GPU architectures** (but this IS the baseline)

### Detectable (Systematic Deviations > 1.5× Baseline)  
✓ **CUDA version changes**  
✓ **Pipeline parallelism rank location**  
✓ **Heavy concurrent CUDA stream work**

### Potentially Undetectable (Within Baseline Noise)
⚠️ **Minor software version updates** (those not changing kernels)  
⚠️ **Batch composition/reshuffling** (fixed size)  
⚠️ **GPU device location** (same architecture)  
⚠️ **Light concurrent CUDA stream work** (at low utilization)

## Implications for AI Governance

### What Verification Can Detect
1. **Hidden batch size increases** → Strongly detectable
2. **Different model architectures** → Detectable  
3. **Kernel/precision changes** → Detectable
4. **Heavy parallel unauthorized work** → Detectable via interference patterns

### Verification Challenges
1. **Light concurrent work at low utilization** → May execute independently on different SMs
2. **Legitimate software updates** → May trigger false positives (need allowlisting)
3. **Layer-dependent effects** → Must verify multiple layers, not just output
4. **Statistical noise at high utilization** → Complicates threshold setting

### Design Recommendations for Verification Systems
1. **Multi-layer verification:** Check activations at multiple depths (early, middle, late layers)
2. **Batch size as primary signal:** Strongest indicator of capacity manipulation
3. **Per-layer thresholds:** Different layers show different baseline variations
4. **Utilization monitoring:** High utilization increases detectability of parallel work
5. **Software version registry:** Maintain allowlist of legitimate version changes
6. **Continuous monitoring:** Single-point checks insufficient; need temporal consistency

## Open Questions

### Requiring Further Experimentation
1. **Precision changes (FP8/FP4):** Do lower precisions increase or decrease detectability?
2. **Scaling with model size:** How do forensics scale to 70B+ models?  
3. **Production-realistic batching:** Continuous batching, speculative decoding impacts?
4. **Long-context scenarios:** Do deviations accumulate or saturate with sequence length?
5. **Quantization:** INT8/INT4 inference detectability?

### Theoretical Questions  
1. **Optimal layer selection:** Which layers are most forensically informative?
2. **Statistical power:** How many samples needed for reliable detection?
3. **Adversarial robustness:** Can sophisticated attackers craft undetectable evasions?

## Repository Structure

```
/experiments/           # Experiment scripts
  batch_matrix_experiment.py      # Batch size variation testing
  cuda_stream_experiment.py       # Parallel stream testing (future)
  cross_hardware_baseline.py      # Hardware comparison baseline
  
/comparison/           # Analysis scripts  
  compare_batch_matrix.py         # Batch size detectability analysis
  compare_baseline.py             # Cross-hardware baseline analysis
  
/results/              # Experimental data (JSON)
  *_baseline_*.json               # Cross-hardware baselines
  *_batch_matrix_*.json           # Batch size experiments
  *_parallel_*.json               # CUDA stream experiments
  
/analysis/             # Summary reports
  baseline_analysis_*.json        # Statistical summaries
```

## Technical Details

### Why Key Vectors?
- **Attention keys** capture the model's internal representations before output transformation
- More sensitive to computational setup than final logits
- Available in cached inference (past_key_values), no extra computation needed
- Distributed across all layers, enabling depth-profile analysis

### Why Last Token Position?
- Most recent computation, least opportunity for error accumulation to saturate
- Relevant for autoregressive generation (next token prediction)
- Simplifies padding concerns in batched inference

### Detectability Criterion
**Threshold:** 1.5× baseline distance  
**Rationale:** Conservative threshold allowing for measurement uncertainty while detecting meaningful deviations

**Alternative thresholds under consideration:**
- Statistical: baseline_mean + 2σ (95% confidence)
- Percentile-based: 95th percentile of baseline  
- Layer-specific: Different thresholds per layer depth

## Running the Experiments

### Prerequisites
```bash
pip install torch transformers numpy
# Requires CUDA-capable GPU
```

### Quick Start
```python
# Establish cross-hardware baseline
python cross_hardware_baseline.py --hardware a100
python cross_hardware_baseline.py --hardware h100

# Run batch size detectability test  
python batch_matrix_experiment.py  # Runs all batch sizes (1, 2, 4)

# Compare results
python compare_batch_matrix.py a100_batch_matrix_*.json h100_batch_matrix_*.json
```

### Interpreting Results
- **L2 distance < 0.1:** Likely same computational setup
- **0.1 < L2 < 1.0:** Hardware differences or minor software variations  
- **L2 > 1.0:** Significant setup differences (batch size, kernels, etc.)
- **L2 > 10:** Major computational differences or parallel interference

## Contributing

This research is in active development. Key areas for contribution:
1. Testing additional evasion vectors (FP8, speculative decoding, etc.)
2. Scaling experiments to larger models (70B, 405B)
3. Production-realistic inference scenarios (continuous batching, multi-tenant)
4. Statistical analysis improvements (power analysis, optimal sampling)
5. Adversarial robustness testing

## Citation

If you use this work, please cite:
```
[Citation information pending publication]
```

Related work:
- RAND Report: "Securing Frontier AI" (WRA4077-1)  
- Internal document on compute governance (available on request)

## License

[License TBD]

## Contact

[Contact information]

---

**Status:** Active research project. Results are preliminary and subject to revision as we explore additional evasion vectors and scaling regimes.

**Last Updated:** November 2025
