# Experimenting with different model quants and hardware
vLLM inference with different model quants.

##Key findings: 
- INT quantization, which is common for LLM inference, breaks determinism! No bit-exact repeatability even within one python process!
- Tested for both INT4 and INT8.
- Floating point quantization maintains reproducibility, as long as there is native hardware support for the precision format (FP8, FP4...).
- In an experiment with Qwen3 at INT8 and FP8 weights, FP16 activations, all else being equal, INT dequantization broke determinism seen in FP8.
- Also non-deterministic for Deepseek v2 coder and Kimi K2 thinking at INT4. See Floating_point_noise_GPU_verification/experiments/large_scale_experiment.
