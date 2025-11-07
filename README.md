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
- **Extraction:** Key vectors from attention layers at multiple positions, hidden states
- **Measurement:** L2 distance between key vectors or hidden states


## Experiments and Results

### 1. Repetition Reproducibility ✓
**Test:** Multiple runs of identical setup  
**Result:** Bit-exact reproduction on single CUDA stream  
**Implication:** Production inference (single stream, controlled batch size) is deterministic.

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
**Implication:** Some software updates can be detected (may need allowlisting)

### 8. Pipeline Parallelism Ranks ✓
**Test:** Different PP ranks spreading the model across 1, 2 and 4 A100s
**Result:** Bitwise identical
**Implication:** Not detectable via FP forensics alone

### 9. Parallel CUDA Streams
**Test:** Concurrent work on separate CUDA stream  
**Result:** 
- **Low GPU utilization:** Bit-exact reproduction (streams on independent SMs)
- **High GPU utilization:** Statistical noise detected (scheduling competition)

Not reproducible across repetitions.

**Implication:** Detection depends on utilization level. At realistic production utilization (~60-80%), separate streams conflict → detectable, statistical noise.

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


## Implications for AI Governance

TBD


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



**Status:** Active research project. Results are preliminary and subject to revision as we explore additional evasion vectors and scaling regimes.

**Last Updated:** November 2025
