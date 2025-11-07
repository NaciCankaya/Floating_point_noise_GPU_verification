#!/bin/bash
# RunPod Setup Script for Floating Point Verification Experiments
# Run this script when starting a new RunPod instance

set -e  # Exit on error

echo "===================================="
echo "RunPod Setup Script"
echo "===================================="
echo ""

# 1. Display GPU Information
echo "📊 GPU Information:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo ""

# 2. Pull latest changes
echo "📥 Pulling latest changes from repository..."
git pull origin main || echo "⚠️  Git pull failed (this is OK if you're already up to date)"
echo ""

# 3. Install dependencies
echo "📦 Installing dependencies..."
echo "   Installing basic packages (fast)..."
pip install -q transformers hf_transfer accelerate ninja packaging wheel

echo "   Installing flash-attn (this may take 5-10 minutes)..."
pip install flash-attn --no-build-isolation

echo ""

# 4. Verify installation
echo "✅ Verifying setup..."
python3 -c "
import torch
import transformers
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Transformers version: {transformers.__version__}')
print(f'Number of GPUs: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "===================================="
echo "✅ Setup Complete!"
echo "===================================="
echo ""
echo "💡 Tips:"
echo "  - Run your experiments"
echo "  - Commit results: git add . && git commit -m 'Results from <GPU_TYPE>'"
echo "  - Push before terminating: git push origin main"
echo ""
