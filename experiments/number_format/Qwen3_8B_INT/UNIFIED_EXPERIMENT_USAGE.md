# Unified Quantization Experiment Usage Guide

## Overview

The `unified_quantization_experiment.py` script provides a single, unified codebase for testing non-determinism across multiple INT4 quantization formats. This replaces the previous approach of maintaining separate notebook files for each variant.

## Key Improvements

### 1. **Single Source of Truth**
- One script instead of 4-6 separate notebooks
- Consistent methodology across all variants
- Easier to maintain and update

### 2. **Comprehensive JSON Output**
Previously, analysis results were only printed to console. Now the JSON output includes:
- All statistical metrics (L2 distances, per-token std, etc.)
- **NEW:** Textual verdict and interpretation in `verdict` field:
  - `is_deterministic`: boolean flag
  - `summary`: high-level finding
  - `root_cause`: technical explanation
  - `forensic_implications`: guidance for verification use cases
  - `details`: human-readable analysis points

### 3. **Configuration-Driven**
All 6 quantization variants are configured in one place:
- AWQ (with/without Marlin)
- GPTQ (with/without Marlin)
- OpenVINO INT4
- PyTorch INT4

## Supported Variants

| Variant ID | Model | Quantization | Kernel |
|-----------|-------|--------------|--------|
| `awq_marlin` | Qwen/Qwen3-8B-AWQ | AWQ INT4 | Marlin |
| `awq` | Qwen/Qwen3-8B-AWQ | AWQ INT4 | Standard |
| `gptq_marlin` | JunHowie/Qwen3-8B-GPTQ-Int4 | GPTQ INT4 | Marlin |
| `gptq` | JunHowie/Qwen3-8B-GPTQ-Int4 | GPTQ INT4 | Standard |
| `openvino` | OpenVINO/Qwen3-8B-int4-ov | OpenVINO INT4 | N/A |
| `pytorch_int4` | pytorch/Qwen3-8B-INT4 | PyTorch INT4 | N/A |

## Usage

### List Available Variants
```bash
python unified_quantization_experiment.py --list-variants
```

### Run All Variants (Default)
```bash
python unified_quantization_experiment.py
```

Or explicitly:
```bash
python unified_quantization_experiment.py --variants all
```

### Run Specific Variants
```bash
# Single variant
python unified_quantization_experiment.py --variants awq_marlin

# Multiple variants
python unified_quantization_experiment.py --variants awq_marlin awq gptq_marlin

# Just the AWQ variants
python unified_quantization_experiment.py --variants awq_marlin awq
```

### Specify Output Directory
```bash
python unified_quantization_experiment.py --output-dir ./results/2025_01_15
```

## Output Format

Each variant produces a JSON file with the following structure:

```json
{
  "variant": "awq_marlin",
  "experiment": "quantization_nondeterminism_investigation",
  "timestamp": "2025-01-15T10:30:00.123456",
  "model": "Qwen/Qwen3-8B-AWQ",
  "quantization": "awq_marlin",
  "description": "AWQ INT4 with Marlin kernel",
  "config": {
    "tensor_parallel": 1,
    "max_model_len": 8192,
    "max_tokens": 50,
    "repetitions": 5,
    "temperature": 0.0,
    "seed": 42
  },
  "prompt_length": 2048,
  "baseline": {
    "stats": {
      "tokens_identical": true,
      "logprobs_exact": false,
      "l2_mean": 1.23e-05,
      "l2_std": 3.45e-06,
      "l2_max": 1.89e-05,
      "l2_min": 8.76e-06,
      "std_per_token_mean": 2.34e-06,
      "std_per_token_max": 5.67e-06,
      "std_per_token_min": 1.23e-07
    },
    "tokens": [...],
    "logprobs": [...]
  },
  "deterministic": {
    "succeeded": true,
    "stats": {...},
    "tokens": [...],
    "logprobs": [...]
  },
  "kernels": {
    "profiling_succeeded": true,
    "all_kernels": [...],
    "marlin_kernels": [...],
    "gemm_kernels": [...],
    "dequant_kernels": [...]
  },
  "verdict": {
    "is_deterministic": false,
    "deterministic_mode_tested": true,
    "summary": "BASELINE IS NON-DETERMINISTIC",
    "root_cause": "Non-deterministic parallel reduction, likely in kernel accumulation logic",
    "forensic_implications": "Noise level: ~1.23e-05 L2. For detection, need systematic deviation >> noise. Recommend: 3-5 samples for statistical significance",
    "details": [
      "Mean L2 deviation: 1.23e-05",
      "Max L2 deviation: 1.89e-05",
      "✓ DETERMINISTIC MODE FIXES IT",
      "Can be fixed but at performance cost"
    ]
  },
  "success": true
}
```

## Key JSON Fields

### `verdict` Object (NEW)
This object contains the interpreted analysis that was previously only in console output:

- **`is_deterministic`**: `true` if baseline is fully reproducible, `false` if noise detected
- **`deterministic_mode_tested`**: Whether deterministic PyTorch mode was successfully tested
- **`summary`**: One-line verdict (e.g., "BASELINE IS NON-DETERMINISTIC")
- **`root_cause`**: Technical explanation of why non-determinism occurs (or doesn't)
- **`forensic_implications`**: Practical guidance for using results in verification scenarios
- **`details`**: Array of human-readable findings (replaces console-only output)

### `baseline.stats` Object
Contains quantitative reproducibility metrics:
- `tokens_identical`: Whether token sequences match across runs
- `logprobs_exact`: Whether logprobs are bit-exact
- `l2_mean`, `l2_std`, `l2_max`, `l2_min`: L2 distance statistics
- `std_per_token_mean`, `std_per_token_max`, `std_per_token_min`: Per-position noise

### `kernels` Object
CUDA kernel profiling data:
- `all_kernels`: All detected CUDA kernels with timing
- `marlin_kernels`: Marlin-specific kernels (if any)
- `gemm_kernels`: Matrix multiplication kernels
- `dequant_kernels`: Dequantization/quantization kernels

## Experiment Workflow

For each variant, the script:

1. **Baseline Test**: Runs 5 repetitions in standard mode
   - Measures logprob noise between runs
   - Profiles CUDA kernels

2. **Deterministic Test**: Reloads model with PyTorch deterministic mode
   - Tests if deterministic algorithms eliminate noise
   - Helps identify root cause

3. **Analysis**: Generates verdict and interpretation
   - Classifies determinism level
   - Identifies likely root cause
   - Provides forensic implications

4. **Output**: Saves comprehensive JSON with all data + interpretation

## Comparison with Previous Approach

### Before (Separate Notebooks)
```
Qwen3_8B_AWQ_Marlin.ipynb       (2,000 lines)
Qwen3_8B_AWQnoMarlin.ipynb      (2,000 lines)
Qwen3_8B_GPTQ4Marlin.ipynb      (2,000 lines)
Qwen3_8B_GPTQ4_noMarlin.ipynb   (2,000 lines)
```
- Total: ~8,000 lines of duplicated code
- Analysis verdict only in console output
- Hard to maintain consistency

### After (Unified Script)
```
unified_quantization_experiment.py  (1,200 lines)
```
- Single source of truth
- Verdict saved to JSON
- Easy to add new variants
- Consistent methodology

## Adding New Variants

To add a new quantization format, simply add to `MODEL_CONFIGS`:

```python
MODEL_CONFIGS = {
    # ... existing configs ...
    "new_format": {
        "model_name": "organization/model-name",
        "quantization": "format_name",  # or None if special loading
        "description": "Description of this format",
        "note": "Any special considerations (optional)"
    }
}
```

No other code changes needed!

## Notes

- **OpenVINO and PyTorch INT4**: These variants may require different vLLM loading parameters or may not be directly supported by vLLM. The script will gracefully handle loading failures and report them in the JSON output.

- **GPU Memory**: Each variant reloads the model, so ensure sufficient GPU memory. The script clears GPU memory between variants.

- **Runtime**: Each variant takes ~5-10 minutes depending on hardware. Running all 6 variants takes ~30-60 minutes.

- **Deterministic Mode**: Some quantization methods may not support PyTorch's deterministic mode. This is expected and handled gracefully.

## Troubleshooting

### Model Loading Fails
```json
{
  "variant": "openvino",
  "error": "Model format not supported by vLLM",
  "success": false
}
```
This is expected for formats not yet supported by vLLM. The script continues with other variants.

### CUDA OOM (Out of Memory)
Reduce `GPU_MEMORY_UTILIZATION` in the script (default: 0.9) or run variants one at a time.

### Kernel Profiling Fails
Non-fatal - script continues without profiling data. The `kernels.profiling_succeeded` field will be `false`.

## Example Workflow

```bash
# 1. List available variants
python unified_quantization_experiment.py --list-variants

# 2. Test just AWQ variants first
python unified_quantization_experiment.py --variants awq_marlin awq --output-dir ./results/awq_test

# 3. Run full comparison across all formats
python unified_quantization_experiment.py --output-dir ./results/full_comparison

# 4. Analyze results
python analyze_results.py ./results/full_comparison/*.json
```

## Migration from Old Notebooks

If you have existing notebook-based experiments:

1. Results are compatible - same statistical metrics
2. New JSON includes `verdict` field with analysis
3. Can continue using old notebooks or switch to unified script
4. Unified script recommended for new experiments
