Experiments comparing prefill forward pass activations for identical model and inputs across different GPUs.

All experiments were repeated ten times each, to check against "statistical" deviations. 

Key findings: 
1. Identical activations if all variables are fixed.
2. Activations differ between hardware types, but are perfectly reproducible when using the same hardware type, even across different physical devices

Results from different_hardware.py

======================================================================
HARDWARE TYPE COMPARISON
======================================================================

Setup verification:
  Models: {'Qwen/Qwen2.5-7B-Instruct'}
  Dtypes: {'float16'}
  Prompt tokens: {1134}
  Hidden dims: {3584}

GPU types found: 7
  NVIDIA A100-SXM4-80GB: 1 file(s)
  NVIDIA A100 80GB PCIe: 3 file(s)
  NVIDIA A40: 1 file(s)
  NVIDIA H100 NVL: 1 file(s)
  NVIDIA H100 PCIe: 1 file(s)
  NVIDIA H200: 1 file(s)
  NVIDIA L40S: 1 file(s)

======================================================================
CROSS-HARDWARE L2 DISTANCES
======================================================================

Batch size 1:
  NVIDIA A100-SXM4-80GB vs NVIDIA A100 80GB PCIe
    L2 distance: 0.000000
  NVIDIA A100-SXM4-80GB vs NVIDIA A40
    L2 distance: 0.495769
  NVIDIA A100-SXM4-80GB vs NVIDIA H100 NVL
    L2 distance: 0.526376
  NVIDIA A100-SXM4-80GB vs NVIDIA H100 PCIe
    L2 distance: 0.526376
  NVIDIA A100-SXM4-80GB vs NVIDIA H200
    L2 distance: 0.526376
  NVIDIA A100-SXM4-80GB vs NVIDIA L40S
    L2 distance: 0.449527
  NVIDIA A100 80GB PCIe vs NVIDIA A40
    L2 distance: 0.495769
  NVIDIA A100 80GB PCIe vs NVIDIA H100 NVL
    L2 distance: 0.526376
  NVIDIA A100 80GB PCIe vs NVIDIA H100 PCIe
    L2 distance: 0.526376
  NVIDIA A100 80GB PCIe vs NVIDIA H200
    L2 distance: 0.526376
  NVIDIA A100 80GB PCIe vs NVIDIA L40S
    L2 distance: 0.449527
  NVIDIA A40 vs NVIDIA H100 NVL
    L2 distance: 0.548924
  NVIDIA A40 vs NVIDIA H100 PCIe
    L2 distance: 0.548924
  NVIDIA A40 vs NVIDIA H200
    L2 distance: 0.548924
  NVIDIA A40 vs NVIDIA L40S
    L2 distance: 0.576388
  NVIDIA H100 NVL vs NVIDIA H100 PCIe
    L2 distance: 0.000000
  NVIDIA H100 NVL vs NVIDIA H200
    L2 distance: 0.000000
  NVIDIA H100 NVL vs NVIDIA L40S
    L2 distance: 0.456233
  NVIDIA H100 PCIe vs NVIDIA H200
    L2 distance: 0.000000
  NVIDIA H100 PCIe vs NVIDIA L40S
    L2 distance: 0.456233
  NVIDIA H200 vs NVIDIA L40S
    L2 distance: 0.456233

Batch size 2:
  NVIDIA A100-SXM4-80GB vs NVIDIA A100 80GB PCIe
    L2 distance: 0.000000
  NVIDIA A100-SXM4-80GB vs NVIDIA A40
    L2 distance: 0.572550
  NVIDIA A100-SXM4-80GB vs NVIDIA H100 NVL
    L2 distance: 0.604769
  NVIDIA A100-SXM4-80GB vs NVIDIA H100 PCIe
    L2 distance: 0.604769
  NVIDIA A100-SXM4-80GB vs NVIDIA H200
    L2 distance: 0.604769
  NVIDIA A100-SXM4-80GB vs NVIDIA L40S
    L2 distance: 0.642044
  NVIDIA A100 80GB PCIe vs NVIDIA A40
    L2 distance: 0.572550
  NVIDIA A100 80GB PCIe vs NVIDIA H100 NVL
    L2 distance: 0.604769
  NVIDIA A100 80GB PCIe vs NVIDIA H100 PCIe
    L2 distance: 0.604769
  NVIDIA A100 80GB PCIe vs NVIDIA H200
    L2 distance: 0.604769
  NVIDIA A100 80GB PCIe vs NVIDIA L40S
    L2 distance: 0.642044
  NVIDIA A40 vs NVIDIA H100 NVL
    L2 distance: 0.464086
  NVIDIA A40 vs NVIDIA H100 PCIe
    L2 distance: 0.464086
  NVIDIA A40 vs NVIDIA H200
    L2 distance: 0.464086
  NVIDIA A40 vs NVIDIA L40S
    L2 distance: 0.571982
  NVIDIA H100 NVL vs NVIDIA H100 PCIe
    L2 distance: 0.000000
  NVIDIA H100 NVL vs NVIDIA H200
    L2 distance: 0.000000
  NVIDIA H100 NVL vs NVIDIA L40S
    L2 distance: 0.444335
  NVIDIA H100 PCIe vs NVIDIA H200
    L2 distance: 0.000000
  NVIDIA H100 PCIe vs NVIDIA L40S
    L2 distance: 0.444335
  NVIDIA H200 vs NVIDIA L40S
    L2 distance: 0.444335

Batch size 4:
  NVIDIA A100-SXM4-80GB vs NVIDIA A100 80GB PCIe
    L2 distance: 0.000000
  NVIDIA A100-SXM4-80GB vs NVIDIA A40
    L2 distance: 0.527562
  NVIDIA A100-SXM4-80GB vs NVIDIA H100 NVL
    L2 distance: 0.456900
  NVIDIA A100-SXM4-80GB vs NVIDIA H100 PCIe
    L2 distance: 0.456900
  NVIDIA A100-SXM4-80GB vs NVIDIA H200
    L2 distance: 0.456900
  NVIDIA A100-SXM4-80GB vs NVIDIA L40S
    L2 distance: 0.631859
  NVIDIA A100 80GB PCIe vs NVIDIA A40
    L2 distance: 0.527562
  NVIDIA A100 80GB PCIe vs NVIDIA H100 NVL
    L2 distance: 0.456900
  NVIDIA A100 80GB PCIe vs NVIDIA H100 PCIe
    L2 distance: 0.456900
  NVIDIA A100 80GB PCIe vs NVIDIA H200
    L2 distance: 0.456900
  NVIDIA A100 80GB PCIe vs NVIDIA L40S
    L2 distance: 0.631859
  NVIDIA A40 vs NVIDIA H100 NVL
    L2 distance: 0.464086
  NVIDIA A40 vs NVIDIA H100 PCIe
    L2 distance: 0.464086
  NVIDIA A40 vs NVIDIA H200
    L2 distance: 0.464086
  NVIDIA A40 vs NVIDIA L40S
    L2 distance: 0.560219
  NVIDIA H100 NVL vs NVIDIA H100 PCIe
    L2 distance: 0.000000
  NVIDIA H100 NVL vs NVIDIA H200
    L2 distance: 0.000000
  NVIDIA H100 NVL vs NVIDIA L40S
    L2 distance: 0.480879
  NVIDIA H100 PCIe vs NVIDIA H200
    L2 distance: 0.000000
  NVIDIA H100 PCIe vs NVIDIA L40S
    L2 distance: 0.480879
  NVIDIA H200 vs NVIDIA L40S
    L2 distance: 0.480879

======================================================================
STATISTICAL NOISE (within-hardware, across runs)
======================================================================

NVIDIA A100-SXM4-80GB:
  Reported statistical noise: {'batch_size_1': {'mean': 0.0, 'std': 0.0}, 'batch_size_2': {'mean': 0.0, 'std': 0.0}, 'batch_size_4': {'mean': 0.0, 'std': 0.0}}

NVIDIA A100 80GB PCIe:
  Reported statistical noise: {'batch_size_1': {'mean': 0.0, 'std': 0.0}, 'batch_size_2': {'mean': 0.0, 'std': 0.0}, 'batch_size_4': {'mean': 0.0, 'std': 0.0}}

NVIDIA A40:
  Reported statistical noise: {'batch_size_1': {'mean': 0.0, 'std': 0.0}, 'batch_size_2': {'mean': 0.0, 'std': 0.0}, 'batch_size_4': {'mean': 0.0, 'std': 0.0}}

NVIDIA H100 NVL:
  Reported statistical noise: {'batch_size_1': {'mean': 0.0, 'std': 0.0}, 'batch_size_2': {'mean': 0.0, 'std': 0.0}, 'batch_size_4': {'mean': 0.0, 'std': 0.0}}

NVIDIA H100 PCIe:
  Reported statistical noise: {'batch_size_1': {'mean': 0.0, 'std': 0.0}, 'batch_size_2': {'mean': 0.0, 'std': 0.0}, 'batch_size_4': {'mean': 0.0, 'std': 0.0}}

NVIDIA H200:
  Reported statistical noise: {'batch_size_1': {'mean': 0.0, 'std': 0.0}, 'batch_size_2': {'mean': 0.0, 'std': 0.0}, 'batch_size_4': {'mean': 0.0, 'std': 0.0}}

NVIDIA L40S:
  Reported statistical noise: {'batch_size_1': {'mean': 0.0, 'std': 0.0}, 'batch_size_2': {'mean': 0.0, 'std': 0.0}, 'batch_size_4': {'mean': 0.0, 'std': 0.0}}