#!/bin/bash
#
# Install GPTQ backend with proper CUDA version detection
#

echo "Detecting CUDA version..."
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9\.]*\).*/\1/')
echo "  Detected CUDA: $CUDA_VERSION"

# Map CUDA version to package suffix
if [[ "$CUDA_VERSION" == 12.8* ]]; then
    CUDA_SUFFIX="cu128"
elif [[ "$CUDA_VERSION" == 12.4* ]]; then
    CUDA_SUFFIX="cu124"
elif [[ "$CUDA_VERSION" == 12.1* ]]; then
    CUDA_SUFFIX="cu121"
elif [[ "$CUDA_VERSION" == 11.8* ]]; then
    CUDA_SUFFIX="cu118"
else
    CUDA_SUFFIX="cu121"  # Default fallback
    echo "  ⚠ Unknown CUDA version, trying cu121 as default"
fi

echo "  Using CUDA suffix: $CUDA_SUFFIX"
echo ""

# Try to install auto-gptq with matching CUDA version
echo "Attempting to install auto-gptq with pre-built wheels..."
pip install auto-gptq --no-build-isolation --extra-index-url https://huggingface.github.io/autogptq-index/whl/${CUDA_SUFFIX}/

if [ $? -eq 0 ]; then
    echo "✓ auto-gptq installed successfully"
    exit 0
fi

# If that fails, try common CUDA versions
echo "  First attempt failed, trying alternate CUDA versions..."

for cuda_v in cu121 cu124 cu118; do
    if [ "$cuda_v" != "$CUDA_SUFFIX" ]; then
        echo "  Trying $cuda_v..."
        pip install auto-gptq --no-build-isolation --extra-index-url https://huggingface.github.io/autogptq-index/whl/${cuda_v}/
        if [ $? -eq 0 ]; then
            echo "✓ auto-gptq installed successfully with $cuda_v"
            exit 0
        fi
    fi
done

# Last resort: try gptqmodel
echo "  All auto-gptq attempts failed, trying gptqmodel..."
pip install gptqmodel --no-build-isolation

if [ $? -eq 0 ]; then
    echo "✓ gptqmodel installed successfully"
    exit 0
fi

echo "✗ All GPTQ backend installation attempts failed"
echo "  The experiment may not work without a GPTQ backend"
exit 1
