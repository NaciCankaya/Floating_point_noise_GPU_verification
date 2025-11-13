#!/bin/bash
#
# Automated setup and execution script for Experiment 0
# Run this directly on the vast.ai instance
#
# Usage:
#   ssh -p 11887 root@ssh3.vast.ai 'bash -s' < run_exp0.sh
#   or
#   ssh -p 46127 root@207.180.148.74 'bash -s' < run_exp0.sh
#

set -e  # Exit on error

echo "================================================================================"
echo "EXPERIMENT 0: AUTOMATED SETUP AND EXECUTION"
echo "================================================================================"
echo "Start time: $(date)"
echo ""

# Check GPU
echo "================================================================================"
echo "CHECKING GPU AND CUDA"
echo "================================================================================"
nvidia-smi
echo ""

# Check disk space
echo "================================================================================"
echo "CHECKING DISK SPACE"
echo "================================================================================"
df -h | head -10
echo ""

# Install git if needed
if ! command -v git &> /dev/null; then
    echo "Installing git..."
    apt-get update && apt-get install -y git
fi

# Clone or update repository
echo "================================================================================"
echo "SETTING UP REPOSITORY"
echo "================================================================================"

if [ -d "Floating_point_noise_GPU_verification" ]; then
    echo "Repository exists, updating..."
    cd Floating_point_noise_GPU_verification
    git fetch origin claude/fix-ablation-readme-01LusMhWHzkH1omDt88MEsVv
    git checkout claude/fix-ablation-readme-01LusMhWHzkH1omDt88MEsVv
    git pull origin claude/fix-ablation-readme-01LusMhWHzkH1omDt88MEsVv
else
    echo "Cloning repository..."
    git clone https://github.com/NaciCankaya/Floating_point_noise_GPU_verification.git
    cd Floating_point_noise_GPU_verification
    git checkout claude/fix-ablation-readme-01LusMhWHzkH1omDt88MEsVv
fi

cd experiments/ablation_cross_hardware
echo "Current directory: $(pwd)"
echo ""

# Check Python
echo "================================================================================"
echo "CHECKING PYTHON"
echo "================================================================================"
python3 --version
which python3
echo ""

# Check if PyTorch is installed
echo "================================================================================"
echo "CHECKING/INSTALLING PYTORCH"
echo "================================================================================"

if python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    echo "✓ PyTorch already installed"
else
    echo "Installing PyTorch with CUDA 12.1 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi
echo ""

# Install requirements
echo "================================================================================"
echo "INSTALLING DEPENDENCIES"
echo "================================================================================"
echo "This may take 5-10 minutes..."

# Install base requirements
pip install -r requirements.txt

# Try to install optional dependencies
echo ""
echo "Installing optional dependencies..."

# Flash attention (optional, improves performance)
echo "Attempting to install flash-attention (may take 5-10 minutes)..."
pip install flash-attn --no-build-isolation || echo "⚠ flash-attn installation failed (will use default attention)"

# Note: Skipping GPTQ backend - using unquantized model instead
echo "  → Using unquantized model (Qwen2.5-7B-Instruct), skipping GPTQ backend"

echo ""

# Test imports
echo "================================================================================"
echo "TESTING IMPORTS"
echo "================================================================================"
python3 test_imports.py
IMPORT_EXIT_CODE=$?
echo ""

if [ $IMPORT_EXIT_CODE -ne 0 ]; then
    echo "⚠ Some imports failed, but continuing anyway..."
    echo "The experiment script will report specific errors if needed."
fi

# Run experiment
echo "================================================================================"
echo "RUNNING EXPERIMENT 0"
echo "================================================================================"
echo "This will take approximately 10-20 minutes:"
echo "  - Model download: ~5-10 min (~15GB)"
echo "  - Inference (3 reps × 30 tokens): ~5-10 min"
echo "================================================================================"
echo ""

python3 exp0_reference.py
EXP_EXIT_CODE=$?

echo ""
echo "================================================================================"
echo "EXPERIMENT COMPLETED"
echo "================================================================================"
echo "End time: $(date)"
echo "Exit code: $EXP_EXIT_CODE"
echo ""

if [ $EXP_EXIT_CODE -eq 0 ]; then
    echo "✓✓✓ SUCCESS ✓✓✓"
    echo ""
    echo "Output file:"
    ls -lh reference_baseline.json
    echo ""
    echo "File contains:"
    wc -l reference_baseline.json
    du -h reference_baseline.json
    echo ""
    echo "To retrieve the file:"
    echo "  scp -P 11887 root@ssh3.vast.ai:$(pwd)/reference_baseline.json ."
    echo "  or"
    echo "  scp -P 46127 root@207.180.148.74:$(pwd)/reference_baseline.json ."
else
    echo "✗✗✗ EXPERIMENT FAILED ✗✗✗"
    echo "Exit code: $EXP_EXIT_CODE"
    echo ""
    echo "Troubleshooting:"
    echo "1. Check error messages above"
    echo "2. Verify GPU has enough memory (need 40GB+)"
    echo "3. Check disk space (need 25GB+)"
    echo "4. Ensure CUDA 12.x is installed"
fi

echo ""
echo "================================================================================"
