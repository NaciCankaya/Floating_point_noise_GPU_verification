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
