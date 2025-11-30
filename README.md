# Inference Verification via Floating-Point Forensics

## Context
Verification of an AI datacenter's ML computations (accurate and comprehensive reporting)

This repo documents experiments for an idea for technical AI governance that deals with what Baker et al. refer to as "Verification Subgoals" 1A and 2A. 
https://www.alphaxiv.org/abs/2507.15916

- 1A: Model & output histories: Verify declared ML compute uses are accurate
- 2A: Compute accounting: Verify no large, undeclared uses of declared clusters exist

## Scenario and assumptions

In future AI governance scenarios, for example international treaties, datacenters may be required to report all ML computations performed on their hardware.
To monitor and verify an (inference) server's outputs, a third party could use a security-hardened "verification cluster". The untrusted servers are required to cryptographically commit and log their ML outputs (could be AI-generated data and hashes of some activations/attn key vectors, for example). We assume that these cryptographic commitments verify the timing of an ML workload, for example via a challenge-response protocol. To verify cryptographic commitments against the claimed ML outputs they supposedly correspond to, the verification cluster reproduces a random selection of the untrusted record entries. This reproduction serves verification subgoal 1A. 

But how to ensure comprehensive reporting (subgoal 2A)?

## The problem
Even if timing verification can ensure that there are no "idle bubbles" in the untrusted server, a malicious host could process hidden work in parallel and "off the record". This might be hidden from timing verification if the host can use more efficient execution than declared, matching the expected timing while freeing parallel capacity that goes unmonitored.

## The solution (?)
ML workloads are "numerically sensitive" to the exact computational graph through a processor. Non-associative operations and accumulation leave a (practically) unique "compounded rounding error" for each graph. This is often referred to as "non-determinism" in ML. However, in this work, I make a critical distinction:

**GPU computations exhibit two types of "rounding error":**

- **Statistical noise:** Random, non-reproducible variation from scheduling, atomic operations, etc. Repeated attempts at the same calculation yield different results, even when controlling all (practically possible) variables. I call this "non-deterministic", because of its practial uncontrollability.
- **Systematic deviation:** Reproducible differences from the theoretical, infinite precision baseline, caused by controllable factors in the technical setup (GPU type, batch size, precision, kernels/software, parallelism) I call such differences "deterministic", unlike many others in this area of research, so beware the context.

**If statistical noise is low (or can be characterized), systematic deviations become detectable and attributable to specific (inference) setups. ML workloads computed in parallel to the recorded ML work (we verify timing and allow no idling) would interfere with the numerics of the recorded/comitted activations/key vectors**
At least this is the result I was hoping for with these experiments. Currently, this repo contains results from my first few experimental runs, and the results are promising so far.

## Experiments and Results

For more information on results and their implications, check out this post:

https://docs.google.com/document/d/1-HhQYaEQiz4-cMBblJc3kbXR5Ncp519COoEAVtyT4-Y/edit?usp=sharing

**Last Updated:** November 2025
