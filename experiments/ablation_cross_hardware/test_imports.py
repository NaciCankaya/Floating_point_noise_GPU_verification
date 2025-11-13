#!/usr/bin/env python3
"""
Test script to verify all imports work correctly before running experiments.
"""

import sys
from pathlib import Path

print("Testing imports...")
print("="*80)

# Test standard library imports
print("\n1. Testing standard library imports...")
try:
    import os
    import json
    import time
    import socket
    from datetime import datetime
    print("  ✓ Standard library imports OK")
except ImportError as e:
    print(f"  ✗ Standard library import failed: {e}")
    sys.exit(1)

# Test PyTorch
print("\n2. Testing PyTorch...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.version.cuda}")
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠ CUDA not available (OK for local testing)")
except ImportError as e:
    print(f"  ✗ PyTorch import failed: {e}")
    sys.exit(1)

# Test Transformers
print("\n3. Testing Transformers...")
try:
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"  ✓ Transformers {transformers.__version__}")
except ImportError as e:
    print(f"  ✗ Transformers import failed: {e}")
    sys.exit(1)

# Test PyPDF2
print("\n4. Testing PyPDF2...")
try:
    import PyPDF2
    print(f"  ✓ PyPDF2 OK")
except ImportError as e:
    print(f"  ✗ PyPDF2 import failed: {e}")
    print("  → Install with: pip install PyPDF2")
    sys.exit(1)

# Test numpy
print("\n5. Testing numpy...")
try:
    import numpy as np
    print(f"  ✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"  ✗ NumPy import failed: {e}")
    sys.exit(1)

# Test common utilities
print("\n6. Testing common utilities...")
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from common import (
        load_model,
        get_model_info,
        load_prompt_from_pdf,
        run_multiple_repetitions,
        extract_signals,
        ExperimentWriter,
        ExperimentReader,
        DEFAULT_PDF,
    )
    print("  ✓ Common utilities imported successfully")
except ImportError as e:
    print(f"  ✗ Common utilities import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test PDF file exists
print("\n7. Testing PDF file...")
if Path(DEFAULT_PDF).exists():
    print(f"  ✓ Default PDF found: {DEFAULT_PDF}")
else:
    print(f"  ⚠ Default PDF not found: {DEFAULT_PDF}")
    print("    (This is OK if running locally without the PDF)")

print("\n" + "="*80)
print("✓ All imports successful!")
print("\nYou're ready to run exp0_reference.py")
print("\nOptional checks:")
print("  - Flash Attention: ", end="")
try:
    import flash_attn
    print(f"✓ v{flash_attn.__version__}")
except ImportError:
    print("✗ Not installed (will use default attention)")

print("  - Auto-GPTQ: ", end="")
try:
    import auto_gptq
    print("✓ Installed")
except ImportError:
    print("✗ Not installed (install for GPTQ support)")

print("  - BitsAndBytes: ", end="")
try:
    import bitsandbytes
    print("✓ Installed")
except ImportError:
    print("✗ Not installed (install for BNB support)")

print("\n" + "="*80)
