# Prefill vs Decode: Batch Size Detectability
This experiment tests whether a verifier using prefill (parallel forward pass) can detect the batch size used during decode (autoregressive generation) by comparing key vectors and logprobs. 


## Critical correction
I found that for prefill inference and realistic sequence lengths (>300tokens) batch size did not affect numerics in the first batch element at all. Of course, this means that this kind of verification was doomed from the start. It might still be worth trying other differences in production methods. For example: Can we detect quantization format (AWQ vs. QPTQ, for example) or attention mode (eager vs. FA2) or tensor parallelism with prefill re-execution?
