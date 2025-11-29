# What is detectable despite non-identical hardware used for re-execution?

Arguably the most important experiments in this repo. The generation experiments test the variable in question (parallelis, batch size, attention mode, ...) within the same hardware. The verification experiments re-execute on different hardware (Here: A100 vs. H100) and compare results.

Key findings: Attn mode is still detectable (SNR ~4) while batch size and tensor parallelism are not. I expect quantization method to be detectable as well, even within the same number of bits (AQW vs. GPTQ etc.). But this remains to be tested.
