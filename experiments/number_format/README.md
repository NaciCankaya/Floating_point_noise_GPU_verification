# Experimenting with different model quants and hardware
vLLM inference with different model quants.

##Key findings: 
- INT quantization, which is common for LLM inference, breaks determinism! No bit-exact repeatability even within one python process!
- Tested for both INT4 and INT8.
- Floating point quantization maintains reproducibility, as long as there is native hardware support for the precision format (FP8, FP4...).
- In an experiment with Qwen3 at INT8 and FP8 weights, FP16 activations, all else being equal, INT dequantization broke determinism seen in FP8.
- Also non-deterministic for Deepseek v2 coder and Kimi K2 thinking at INT4. See Floating_point_noise_GPU_verification/experiments/large_scale_experiment.

- Important Update: Not all INT quants are created equal. The INT models in this folder all had non-reproducible activations/logprobs for vLLM inference (reproducible in transformers!). But there are different quant formats even within one INT bit count, and some (typically the more optimized/modern) formats and kernels are deterministic. Check out the INT4 Qwen8B folder for experiment code and results.
- So INT inference in vLLM can be reproducible, without compromise in performance. Different quant methods all produce unique activations/logprobs -> Verifiability is still possible!

Across four different INT4 formats for Qwen3 8B:
================================================================================
LOGPROB COMPARISON RESULTS
================================================================================

✓ IDENTICAL LOGPROBS: None

✗ DIFFERENT LOGPROBS (10 pair(s)):
--------------------------------------------------------------------------------
  awq                  ≠ awq_marlin          
    L2 distance: 1.516037e-02
    Max diff:    8.132815e-03

  gptq                 ≠ gptq_marlin         
    L2 distance: 4.238056e-02
    Max diff:    3.059846e-02

  awq_marlin           ≠ pytorch_int4        
    L2 distance: 3.030119e+00
    Max diff:    1.180331e+00

  awq                  ≠ pytorch_int4        
    L2 distance: 3.031111e+00
    Max diff:    1.180280e+00

  awq_marlin           ≠ gptq                
    L2 distance: 3.272562e+00
    Max diff:    1.629966e+00

  awq                  ≠ gptq                
    L2 distance: 3.275672e+00
    Max diff:    1.633261e+00

  awq_marlin           ≠ gptq_marlin         
    L2 distance: 3.281563e+00
    Max diff:    1.629966e+00

  awq                  ≠ gptq_marlin         
    L2 distance: 3.284650e+00
    Max diff:    1.633261e+00

  gptq                 ≠ pytorch_int4        
    L2 distance: 3.429642e+00
    Max diff:    1.513399e+00

  gptq_marlin          ≠ pytorch_int4        
    L2 distance: 3.439844e+00
    Max diff:    1.513399e+00

================================================================================
SUMMARY
================================================================================
Identical pairs: 0
Different pairs: 10
