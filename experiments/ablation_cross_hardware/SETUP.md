# Setup Guide for Experiment 0

## Quick Start

### 1. Clone Repository (on vast.ai instance)

```bash
git clone https://github.com/NaciCankaya/Floating_point_noise_GPU_verification.git
cd Floating_point_noise_GPU_verification
git checkout claude/fix-ablation-readme-01LusMhWHzkH1omDt88MEsVv
cd experiments/ablation_cross_hardware
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA support (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# If flash-attn fails to install, try:
pip install flash-attn --no-build-isolation
```

### 3. Test Installation

```bash
python test_imports.py
```

If all imports succeed, you're ready to run the experiment!

### 4. Run Experiment 0

```bash
python exp0_reference.py
```

Expected runtime: ~10-20 minutes (depends on model download time)

## Expected Output

The script will:
1. Download Qwen/Qwen3-30B-A3B-GPTQ-Int4 model (~15GB)
2. Load text from DeepSeek_V3_2.pdf and truncate to ~6k tokens
3. Run 3 repetitions of inference (30 tokens each)
4. Verify bit-exact reproducibility
5. Save results to `reference_baseline.json`

## Output File

`reference_baseline.json` contains:
- Experiment metadata
- Configuration details (hardware, software versions)
- 3 runs with extracted signals:
  - Hidden states (3584-dim) at layers [1, 2, 4, 12, 39]
  - Key vectors (512-dim) at layers [1, 2, 4, 12, 39]
  - Top-10 logprobs
  - Extracted at token positions [-3, -2, -1] for each of 30 decode steps

## Troubleshooting

### Model download fails
- Check internet connection
- Ensure sufficient disk space (~20GB)
- Set HuggingFace cache: `export HF_HOME=/workspace/huggingface_cache`

### Flash Attention not installed
- The script will fall back to default attention
- To use flash_attention_2, install: `pip install flash-attn`

### CUDA out of memory
- You need at least 40GB GPU memory for Qwen3-30B-A3B-GPTQ-Int4
- Use A100-80GB or H100 instances

### Non-determinism detected
- This is a critical issue - stop and investigate
- Check CUDA version, PyTorch version
- Verify no concurrent GPU usage
- Try disabling any background processes

## Hardware Requirements

- **GPU**: A100-80GB or H100 (minimum 40GB VRAM)
- **RAM**: 32GB+ recommended
- **Disk**: 25GB+ free space
- **CUDA**: 11.8+ (12.1 or 12.8 recommended)

## Next Steps

After Experiment 0 completes successfully:
1. Verify `reference_baseline.json` was created
2. Check reproducibility passed (all 3 runs identical)
3. The baseline data will be reused for Experiments 1-6
4. Continue with batch size experiment (exp1)

## File Structure

```
experiments/
├── common/                      # Shared utilities
│   ├── __init__.py
│   ├── model_loader.py         # Model loading
│   ├── extraction.py           # Signal extraction
│   ├── runner.py               # Inference orchestration
│   ├── prompts.py              # PDF prompt loading
│   ├── json_writer.py          # JSON output
│   └── json_reader.py          # JSON reading
├── ablation_cross_hardware/
│   ├── exp0_reference.py       # ← Main experiment script
│   ├── requirements.txt        # Dependencies
│   ├── test_imports.py         # Import validation
│   ├── README.md               # Main documentation
│   ├── SETUP.md                # This file
│   ├── DeepSeek_V3_2.pdf       # Default prompt source
│   └── *.pdf                   # Other prompt sources
```
