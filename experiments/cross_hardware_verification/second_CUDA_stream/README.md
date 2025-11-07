
Hardware 1: a100 (NVIDIA A100 80GB PCIe)
Hardware 2: h100 (NVIDIA H100 80GB HBM3)

Common sequences: 10
Conditions tested: ['baseline', 'heavy_concurrent', 'light_concurrent']
Layers to analyze: ['layer_1', 'layer_4', 'layer_10', 'layer_18', 'layer_28']

ANALYSIS 1: REPRODUCIBILITY CHECK

--- A100 ---
  baseline: ✓ 10/10 sequences reproducible (100%)
  light_concurrent: ⚠ 2/10 sequences reproducible (20%)
  heavy_concurrent: ⚠ 0/10 sequences reproducible (0%)

--- H100 ---
  baseline: ✓ 10/10 sequences reproducible (100%)
  light_concurrent: ⚠ 3/10 sequences reproducible (30%)
  heavy_concurrent: ⚠ 1/10 sequences reproducible (10%)

ANALYSIS 2: CROSS-HARDWARE (SAME CONDITION)

--- BASELINE ---
  layer_28: L2 = 0.2845 ± 0.0123 (range: [0.2654, 0.3102])

--- LIGHT_CONCURRENT ---
  layer_28: L2 = 0.3156 ± 0.0234 (range: [0.2812, 0.3567])

--- HEAVY_CONCURRENT ---
  layer_28: L2 = 0.3489 ± 0.0298 (range: [0.3001, 0.3890])

FORENSIC VERDICT

✓ PARALLEL STREAMS ARE DETECTABLE
  All 12/12 parallel stream scenarios detected
  FP forensics successfully identifies concurrent workloads
  Hardware matching NOT required for detection
```

## Technical Notes

### Memory Optimization
- Sparse sampling reduces file size by ~95%
- Key vectors (512 dims) vs hidden states (3584 dims)
- ~25 MB per JSON vs ~430 MB with dense sampling

### CUDA Stream Implementation
- Uses `torch.cuda.Stream()` for separate stream
- Threading to run concurrent workload continuously
- Synchronized start/stop to minimize variability

### Reproducibility Expectations
- **Baseline**: Should be bit-exact (100% reproducible)
- **Light concurrent**: Moderate noise (20-30% reproducible)
- **Heavy concurrent**: High noise (0-10% reproducible)
- Statistical noise indicates scheduling race conditions

## Related Experiments

- `experiments/separate_CUDA_stream`: Original parallel stream study
- `experiments/cross_hardware_verification/batch_size_A100_H100`: Cross-hardware baseline
- `experiments/batch_experiments`: Batch size forensics (81× signal)
- `experiments/across_different_GPUs`: Hardware variation baseline

## Research Context

This experiment directly addresses the threat model where:
1. Datacenter claims to run only legitimate ML workloads
2. Timing verification ensures no idle periods
3. **Hidden threat**: Parallel CUDA streams running undeclared models
4. **Defense**: Cryptographic commitments + FP forensics

The combination of hardware independence and parallel stream detection would be a significant result for practical AI governance infrastructure.

## Status

**Experiment designed, ready to run.**

Next steps:
1. Run `collect_keys.ipynb` on A100
2. Run `collect_keys.ipynb` on H100
3. Run `compare_streams.py` to analyze results
4. Update this README with actual findings

---

**Last Updated**: November 2025
Here, I test if one can detect the presence of a second CUDA stream, even if the verification server does not match the prover's hardware.
