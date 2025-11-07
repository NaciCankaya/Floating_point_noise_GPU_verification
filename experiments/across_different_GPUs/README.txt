Experiments comparing prefill forward pass activations for identical model and inputs across different GPUs.

All experiments were repeated ten times each, to check against "statistical" deviations. 

Key findings: 
1. Identical activations of all variables are fixed.
2. Activations differ between hardware types, but are perfectly reproducible when using the same hardware type, even across different physical devices
3. There was another perturbation tested: Different batching. Only activations of the first, unchanged sequence of the batch were fetched and stored. 
   BUT: Differet
   Batching DID affect activations, due to perturbations in the overall computational path (as expected). 
4. Interesting exceptions: H100, H200 and B200 did not show batch-dependence in these experiments. As it turned out, the reason was the naive 
   implementation: Batching one and the same sequence next to itself. Apparently, newer GPUs deduplicate this and don't re-compute. Simple fix: Just batch non-identical sequences together. This recovered batch-dependent behaviour seen everywhere else. In the "batching_experiments" folder you can see how this was done and the results

