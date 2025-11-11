#Testing whether determinism holds at frontier LLM scales, or whether CUDA etc. suddenly decide to use nondeterministic kernels for such heavy, tensor parallel workloads.

Tested for GLM4.6 and Kimi K2 at ~120k token context length, decoding in vLLM. Comparing logprobs here, since key vector/hidden state extraction is not natively supported in vLLM. 

##**Why not transformers?**
To simulate realistic production inference, optimized for performance, not reproducibility/science.

##**Results?**
- Deterministic for all TP ranks, for GLM4.6, Qwen3 30B A3, Deepseek v2 coder lite, Qwen2.5 7B, long and short context.
- NOT deterministic for Kimi K2 thinking. Tested if MLA was the reason by using DS v2 coder lite ->No. Tested v2 in INT4 -> YES. 
- This result motivated my number format experiments, to be found in Floating_point_noise_GPU_verification/experiments/number_format.
