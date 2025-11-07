Test of activation difference across A100s of different cloud services. 
Expected deviation for matched setups: Zero.
Measured: Zero

Note how not all software was identical. Minor version difference in CUDA, some version difference in firmware. 
The update in question (CUDA 12.8 vs 12.9) does not affect numerics on A100, so the results are expected. This experiment 
was a "sanity check" after I did basically the same one on RunPod only.
