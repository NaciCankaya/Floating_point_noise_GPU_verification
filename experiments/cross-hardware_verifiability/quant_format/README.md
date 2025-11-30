# Are quantization differences detectable across different hardware types?

I do not mean FP4 vs. FP8 or so. That would be too easy to detect in numerics. Something more subtle: Same INT precision, different methods of storing/compressing bits into groups. I compare INT4 AWQ vs. INT4 GPTQ here. AWQ with or without vLLM's marlin kernel.

## Results (within identical hardware)

Within hardware (Top5 Logprob L2, H100):
```
----------------------------------------
NOISE FLOOR SUMMARY
----------------------------------------
  awq: decode=0.00e+00, prefill=0.00e+00
  awq_marlin: decode=0.00e+00, prefill=0.00e+00
  gptq_marlin: decode=0.00e+00, prefill=0.00e+00
  gptq: decode=4.67e-02, prefill=0.00e+00

Cross-format signal must exceed this noise floor plus the hardware difference to be detectable.

PREFILL
                       awq   awq_marlin  gptq_marlin         gptq
         awq     0.00e+00     3.99e-02     5.95e+00     5.96e+00
  awq_marlin     3.99e-02     0.00e+00     5.95e+00     5.96e+00
 gptq_marlin     5.13e+00     5.14e+00     0.00e+00     2.56e-01
        gptq     5.42e+00     5.42e+00     2.57e-01     0.00e+00

DECODE
                       awq   awq_marlin  gptq_marlin         gptq
         awq     0.00e+00     2.82e-02          inf          inf
  awq_marlin     2.82e-02     0.00e+00          inf          inf
 gptq_marlin          inf          inf     0.00e+00     8.58e-02
        gptq          inf          inf     8.58e-02     0.00e+00

inf distance where generated token sequences split apart
```

For sequence generation runs that did not split apart. Diverging decode runs obviosly give inf in off-diagonals


 ## Results (A100 generation, H100 re-execution)
```
Pair                               Signal      Noise      SNR
------------------------------------------------------------
awq_marlin_vs_gptq_marlin        2.18e+00   2.65e-02    82.16×
awq_vs_gptq_marlin               2.19e+00   2.85e-02    76.80×
awq_marlin_vs_gptq               2.18e+00   1.02e-01    21.30×
awq_vs_gptq                      2.19e+00   1.04e-01    20.99×
gptq_marlin_vs_gptq              1.25e-01   9.91e-02     1.26×
awq_vs_awq_marlin                3.02e-02   3.17e-02     0.95×
```
In short: QPTQ vs. AWQ is stronger signal than cross-hardware-noise baseline, but Marlin kernel on/off is too subtle and only evident when re-executing with identical hardware.

 ## Implications

 Difference between GPTQ and AQW is much larger than within AWQ with or without the Marlin kernel. One is qualitatively different maths, the other is different kernel selection for basically the same maths.
