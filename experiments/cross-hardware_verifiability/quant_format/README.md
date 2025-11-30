# Are quantization differences detectable across different hardware types?

I do not mean FP4 vs. FP8 or so. That would be too easy to detect in numerics. Something more subtle: Same INT precision, different methods of storing/compressing bits into groups. I compare INT4 AWQ vs. INT4 GPTQ here. AWQ with or without vLLM's marlin kernel.

## Results (within identical hardware)

Within hardware (H100):
================================================================================
WITHIN-HARDWARE QUANTIZATION EFFECTS (PREFILL)
================================================================================
                       awq   awq_marlin  gptq_marlin
         awq     0.00e+00     3.88e-02     4.23e+00
  awq_marlin     3.88e-02     0.00e+00     4.24e+00
 gptq_marlin     4.16e+00     4.17e+00     0.00e+00

Off-diagonal stats:
  Mean: 2.81e+00
  Range: [3.88e-02, 4.24e+00]

Kernel equivalence classes:
  Class 1: ['awq']
  Class 2: ['awq_marlin']
  Class 3: ['gptq_marlin']


================================================================================
WITHIN-HARDWARE QUANTIZATION EFFECTS (DECODE)
================================================================================
Logprobs (L2 distance):
                       awq   awq_marlin  gptq_marlin
         awq     0.00e+00     2.07e-02     6.18e+00
  awq_marlin     2.07e-02     0.00e+00     6.18e+00
 gptq_marlin     6.18e+00     6.18e+00     0.00e+00

 for sequence generation that did not diverge. Diverging decode runs obviosly give inf in off-diagonals


 ### Implications

 Difference between GPTQ and AQW is much larger than within AWQ with or without the Marlin kernel. One is qualitatively different maths, the other is different kernel selection for basically the same maths.


 ## Results (A100 generation, H100 re-execution)

 ----------------------------------------
PAIRWISE SNR SUMMARY TABLE
----------------------------------------
Pair                               Signal      Noise      SNR
------------------------------------------------------------
awq_marlin_vs_gptq_marlin        2.45e+00   3.02e-02     81.1×
awq_vs_gptq_marlin               2.45e+00   3.22e-02     76.1×
awq_vs_awq_marlin                3.02e-02   3.17e-02      1.0×

In short: QPTQ vs. AWQ is stronger signal than cross-hardware-noise baseline, but Marlin kernel on/off is too subtle and only evident when re-executing with identical hardware.
