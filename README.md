# Inference Verification via Floating-Point Forensics

## Context
Verification of an AI datacenter's ML computations (accurate and comprehensive reporting)

This repo documents experiments for an idea for technical AI governance that deals with what Baker et al. refer to as "Verification Subgoals" 1A and 2A. 
https://www.alphaxiv.org/abs/2507.15916

- 1A: Model & output histories: Verify declared ML compute uses are accurate
- 2A: Compute accounting: Verify no large, undeclared uses of declared clusters exist

## Scenario and assumptions

In future AI governance scenarios, for example international treaties, datacenters may be required to report all ML computations performed on their hardware.
To monitor and verify an (inference) server's outputs, a third party could use a security-hardened "verification cluster". The untrusted servers are required to cryptographically commit and log their ML outputs (could be AI-generated data and hashes of some activations/attn key vectors, for example). We assume that these cryptographic commitments verify the timing of an ML workload, for example via a challenge-response protocol. To verify cryptographic commitments against the claimed ML outputs they supposedly correspond to, the verification cluster reproduces a random selection of the untrusted record entries. This reproduction serves verificatio subgoal 1A. 

But how to ensure comprehensive reporting (subgoal 2A)?

## The problem
Even if timing verification can ensure that there are no "idle bubbles" in the untrusted server, a malicious host could process hidden work in parallel and "off the record".

## The solution (?)
ML workloads are "numerically sensitive" to the exact computational graph through a processor. Non-associative operations and accumulation leave a (practically) unique "compunded rounding error" for each graph. This is oftern refered to as "non-determinism" in ML. However, in this work, I make a critical distinction:

**GPU computations exhibit two types of "rounding error":**

- **Statistical noise:** Random, non-reproducible variation from scheduling, atomic operations, etc. Repeated attempts at the same calculation yield different results, even when controlling all (practically possible) variables. I call this "non-deterministic", because of its practial uncontrollability.
- **Systematic deviation:** Reproducible differences from the theoretical, infinite precision baseline, caused by controllable factors in the technical setup (GPU type, batch size, precision, kernels/software, parallelism) I call such differences "deterministic", unlike many others in this area of research, so beware the context.

**If statistical noise is low (or can be characterized), systematic deviations become detectable and attributable to specific (inference) setups. ML workloads computed in parallel to the recorded ML work (we verify timing and allow no idling) would interfere with the numerics of the recorded/comitted activations/key vectors**
At least this is the result I was hoping for with these experiments. Currently, this repo contains results from my first few experimental runs, and the results are promising so far.

## Experiments and Results

### 1. Repetition Reproducibility
**Test:** Multiple runs of identical setup  
**Result:** Bit-exact reproduction, except for INT quantized models
**Implication:** TBD

### 2. Hardware Variation
**Test:** Same setup on different GPUs
- Same GPU type, different device
- Same GPU type, different cloud provider  
- Different GPU architectures (A100 vs H100)

**Result:** Systematic deviations detected (~0.2-0.4 range)  
**Implication:** Hardware differences are detectable but small enough such that a verification server need not match exact hardware type (important for making this verification more accessible, not every processor can be cheaply security-hardened like H100 with CC).

### 3. Batch Size Changes
**Test:** Cross-hardware (A100 abd H100) with different batch sizes (bs1, bs2, bs4)  
**Result:** 81× larger than baseline L2 from hardware mismatch

**Implication:** Batch size changes create substantial activation differences, far exceeding the usual L2 across A100 and H100. The code in this repo can be easily repurposed to test across other hardware and batch sizes as well.

**The experiments below were so far only tested with matched hardware. They were all zero statistical noise (infinite signal/noise!), except for experiments with multiple CUDA streams AND high GPU utilization.**

### 4. Batch Composition (Fixed Size)
**Test:** Different sequences in batch positions 1-3, same reference at position 0  
**Result:** Zero deviation in reference sequence activations  
**Implication:** Batch composition doesn't affect individual sequence processing. This means that ML outputs need to be checked sequence-wise, not batch-wise.

### 5. Compilation (torch.compile)
**Test:** Compiled vs eager mode. This is a "model organism" for what "secret backend optimization" might be like for an inference setup where a malicious host tries to secretly free up unmonitored throughput capacity
**Result:** Systematic deviations detected  
**Implication:** Compilation optimization detectable

### 6. Attention Implementation (Eager vs FlashAttention2)
**Test:** Different attention kernels  
**Result:** Systematic deviations detected  
**Implication:** Kernel choice is detectable

### 7. CUDA Version Changes
**Test:** Different CUDA toolkit versions  
**Result:** Systematic deviations detected  
**Implication:** Some software updates can be detected. I expect that this includes any updates that change the logic on the processor (kernels, memaccess patterns, ...).

### 8. Pipeline Parallelism Ranks
**Test:** Different PP ranks spreading the model across 1, 2 and 4 A100s
**Result:** Bitwise identical
**Implication:** Not detectable via FP forensics alone

### 9. Parallel CUDA Streams
**Test:** Concurrent work on separate CUDA stream  
**Result:** 
Fully deterministic, only detectable via slowdown of default stream

**Implication:** Timing verification is key



## Implications for AI Governance

TBD, more experiments needed. If trends continue as they did so far: Promising method for datacenter monitoring under adversarial conditions!


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



**Status:** Active research project. Results are preliminary and subject to revision as I explore additional evasion vectors and scaling regimes.

**Last Updated:** November 2025
