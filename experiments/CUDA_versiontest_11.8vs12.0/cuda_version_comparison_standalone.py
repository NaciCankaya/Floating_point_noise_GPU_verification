#!/usr/bin/env python3
"""
Standalone CUDA version comparison experiment.

Installs two CUDA builds of torch in sequence, collects activation tensors
on each, then compares layer-by-layer deviations.

Usage:
  python cuda_version_comparison_standalone.py                      # cu128 vs cu129
  python cuda_version_comparison_standalone.py --v1 cu118 --v2 cu121
  python cuda_version_comparison_standalone.py --collect            # collect only (current torch)
  python cuda_version_comparison_standalone.py --compare f1.json f2.json

The orchestrator re-invokes this script as a subprocess for each collection
phase so that the freshly-installed torch is imported in a clean process.
"""

import argparse
import gc
import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ============================================================
# CONFIGURATION  (edit these or override via CLI)
# ============================================================

MODEL_NAME    = "Qwen/Qwen2.5-7B-Instruct"
CACHE_DIR     = "/workspace/huggingface_cache"
OUTPUT_DIR    = "/workspace/experiments"
TORCH_VERSION = "2.8.0"
DEFAULT_V1    = "cu128"
DEFAULT_V2    = "cu129"
PROMPT_FILE   = "dummytext.txt"
REPETITIONS   = 5
SAMPLE_LAYERS = [1, 2, 3, 4, 7, 10, 14, 18, 22]  # last layer appended automatically

FALLBACK_PROMPT = (
    "The development of large language models has fundamentally transformed "
    "natural language processing and artificial intelligence more broadly. "
    "These models, trained on vast corpora of text data, have demonstrated "
    "remarkable capabilities across a wide range of tasks, from translation "
    "and summarization to question answering and creative writing. However, "
    "their deployment raises significant challenges related to computational "
    "efficiency, interpretability, and safety."
)

# ============================================================
# INSTALL
# ============================================================

def install_torch(cuda_tag: str):
    url = f"https://download.pytorch.org/whl/{cuda_tag}"
    # torchvision must be reinstalled alongside torch to stay ABI-compatible;
    # transformers imports it via image_utils even for text-only models.
    pkgs = [f"torch=={TORCH_VERSION}", "torchvision"]
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + pkgs + ["--index-url", url]
    print(f"\n  pip install torch=={TORCH_VERSION} torchvision --index-url .../{cuda_tag}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"pip install failed for {cuda_tag}")

# ============================================================
# COLLECTION  (torch imported lazily — must run in a fresh subprocess
#              after install_torch() has completed)
# ============================================================

def run_collection(output_dir: str) -> str:
    os.environ.setdefault("HF_HOME", CACHE_DIR)
    os.environ.setdefault("TRANSFORMERS_CACHE", CACHE_DIR)

    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pv = torch.__version__
    cuda_build = ("cu" + pv.split("+cu")[1].split(".")[0]) if "+cu" in pv else "unknown"

    print(f"  GPU:     {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch: {pv}")
    print(f"  CUDA:    {torch.version.cuda}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="eager",
    )

    try:
        prompt = Path(PROMPT_FILE).read_text(encoding="utf-8").strip()
        print(f"  Prompt: {PROMPT_FILE}")
    except FileNotFoundError:
        prompt = FALLBACK_PROMPT
        print("  Prompt: fallback text")

    prompt_tokens = len(tokenizer.encode(prompt))
    num_layers = model.config.num_hidden_layers
    layers = sorted(set(SAMPLE_LAYERS + [num_layers]))

    print(f"  Layers: {layers}  |  prompt tokens: {prompt_tokens}")

    all_reps = {f"layer_{l}": [] for l in layers}

    for rep in range(REPETITIONS):
        inputs = tokenizer([prompt], return_tensors="pt", padding=True)
        last_pos = int(inputs["attention_mask"][0].sum()) - 1
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, use_cache=False)

        for l in layers:
            vec = out.hidden_states[l][0, last_pos, :].cpu().clone()
            all_reps[f"layer_{l}"].append(vec.float().numpy().tolist())

        del out
        gc.collect()
        torch.cuda.empty_cache()

    last_key = f"layer_{layers[-1]}"
    a = __import__("numpy").array(all_reps[last_key])
    identical = bool((a == a[0]).all())
    print(f"  Within-version reproducibility: {'✓' if identical else '✗'}")

    result = {
        "experiment": "cuda_version_forensics",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "hostname": socket.gethostname(),
        },
        "software": {
            "pytorch_version": pv,
            "cuda_runtime": torch.version.cuda,
            "cuda_build": cuda_build,
            "attention_implementation": "eager",
        },
        "config": {
            "dtype": "bfloat16",
            "prompt_tokens": prompt_tokens,
            "repetitions": REPETITIONS,
            "layers_sampled": layers,
            "hidden_dim": len(all_reps[f"layer_{layers[0]}"][0]),
        },
        "reproducibility": {"all_repetitions_identical": identical},
        "raw_activations": all_reps,
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    gpu_tag = torch.cuda.get_device_name(0).replace(" ", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"{gpu_tag}_{cuda_build}_7b_bf16_eager_{ts}.json"
    path.write_text(json.dumps(result, indent=2))

    print(f"JSON_OUTPUT: {path}")   # parsed by orchestrator
    return str(path)

# ============================================================
# COMPARISON  (from CompareCUDA.py, output_dir parameter added)
# ============================================================

def compare_cuda_versions(file1: str, file2: str, output_dir: str) -> dict:
    import numpy as np

    print("=" * 60)
    print("CUDA VERSION COMPARISON ANALYSIS")
    print("=" * 60)

    with open(file1) as f: results1 = json.load(f)
    with open(file2) as f: results2 = json.load(f)

    cuda1 = results1["software"]["cuda_build"]
    cuda2 = results2["software"]["cuda_build"]
    print(f"\nComparing: {cuda1} vs {cuda2}")

    out = {
        "timestamp": datetime.now().isoformat(),
        "files": {"file1": file1, "file2": file2},
        "cuda_versions": {"cuda1": cuda1, "cuda2": cuda2},
    }

    # Setup verification
    print("\n" + "-" * 60)
    print("SETUP VERIFICATION")
    print("-" * 60)

    checks = {
        "GPU":          (results1["hardware"]["gpu"], results2["hardware"]["gpu"]),
        "PyTorch":      (results1["software"]["pytorch_version"].split("+")[0],
                         results2["software"]["pytorch_version"].split("+")[0]),
        "Model":        (results1["model"], results2["model"]),
        "Attention":    (results1["software"]["attention_implementation"],
                         results2["software"]["attention_implementation"]),
        "Precision":    (results1["config"]["dtype"], results2["config"]["dtype"]),
        "Prompt tokens":(results1["config"]["prompt_tokens"],
                         results2["config"]["prompt_tokens"]),
    }

    setup_checks = {}
    all_match = True
    for key, (v1, v2) in checks.items():
        match = v1 == v2
        print(f"  {'✓' if match else '✗'} {key}: {v1} {'==' if match else '!='} {v2}")
        setup_checks[key] = {"value1": v1, "value2": v2, "match": match}
        if not match and key != "PyTorch":
            all_match = False

    out["setup_verification"] = {"checks": setup_checks, "all_match": all_match}

    if not all_match:
        print("\n⚠ WARNING: Experimental setups differ — aborting comparison.")
        _save_comparison(out, cuda1, cuda2, output_dir)
        return out

    # Within-version reproducibility
    layers = sorted(results1["config"]["layers_sampled"])
    repro = {
        cuda1: results1["reproducibility"]["all_repetitions_identical"],
        cuda2: results2["reproducibility"]["all_repetitions_identical"],
    }
    print(f"\nWithin-version reproducibility:  {cuda1}: {repro[cuda1]}  |  {cuda2}: {repro[cuda2]}")
    out["reproducibility"] = repro

    # Layer-by-layer deviations
    print("\n" + "=" * 60)
    print("LAYER-BY-LAYER DEVIATION ANALYSIS")
    print("=" * 60)
    print(f"\nLayers: {layers}\n")

    deviations = {}
    for l in layers:
        key = f"layer_{l}"
        m1 = np.array(results1["raw_activations"][key]).mean(axis=0)
        m2 = np.array(results2["raw_activations"][key]).mean(axis=0)
        diff = np.abs(m1 - m2)
        l2   = float(np.linalg.norm(m1 - m2))
        norm1 = float(np.linalg.norm(m1))
        rel  = l2 / norm1 if norm1 > 0 else 0.0

        deviations[l] = {
            "l2_distance":   l2,
            "relative_diff": rel,
            "max_diff":      float(diff.max()),
            "dims_affected": int((diff > 0.01).sum()),
            "dims_total":    len(diff),
            "norm1":         norm1,
            "norm2":         float(np.linalg.norm(m2)),
        }
        print(f"Layer {l:2d}: L2={l2:.4f}  rel={rel*100:.3f}%  "
              f"dims affected={deviations[l]['dims_affected']}/{len(diff)}")

    out["layer_deviations"] = deviations

    # Error propagation
    first_l2 = deviations[layers[0]]["l2_distance"]
    last_l2  = deviations[layers[-1]]["l2_distance"]
    growth   = last_l2 / first_l2 if first_l2 > 0 else float("inf")

    print(f"\nError propagation: layer {layers[0]} L2={first_l2:.4f} "
          f"→ layer {layers[-1]} L2={last_l2:.4f}  ({growth:.1f}× growth)")

    out["error_propagation"] = {
        "first_layer": layers[0], "last_layer": layers[-1],
        "first_l2": first_l2, "last_l2": last_l2,
        "growth_factor": growth, "grows": last_l2 > first_l2,
    }

    # Diagnosis
    if   last_l2 > 10:  strength = "EXCELLENT"
    elif last_l2 > 1:   strength = "STRONG"
    elif last_l2 > 0.1: strength = "WEAK"
    else:               strength = "NOT_DETECTABLE"

    perfect = repro[cuda1] and repro[cuda2]
    conclusion = ("DISTINGUISHABLE" if last_l2 > 1
                  else "MEASURABLE" if last_l2 > 0.1
                  else "IDENTICAL")

    print(f"\nSignal: {strength}  |  Conclusion: {conclusion}  |  "
          f"Perfect within-version reproducibility: {perfect}")

    out["diagnosis"] = {
        "signal_strength": strength,
        "last_l2_distance": last_l2,
        "error_propagation_confirmed": last_l2 > first_l2,
        "perfect_within_version_reproducibility": perfect,
        "conclusion": conclusion,
    }

    _save_comparison(out, cuda1, cuda2, output_dir)
    return out


def _save_comparison(data: dict, cuda1: str, cuda2: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"comparison_{cuda1}_vs_{cuda2}_{ts}.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"\nComparison saved: {path}")

# ============================================================
# ORCHESTRATOR
# ============================================================

def orchestrate(v1: str, v2: str, output_dir: str):
    json_files = []

    for tag in (v1, v2):
        print(f"\n{'='*60}")
        print(f"INSTALLING torch=={TORCH_VERSION} ({tag})")
        print("=" * 60)
        install_torch(tag)

        print(f"\nCOLLECTING activations ({tag})")
        cmd = [sys.executable, __file__, "--collect", "--out", output_dir]
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Collection subprocess failed for {tag}")

        # Find the JSON just written (most recent matching this tag)
        candidates = sorted(
            Path(output_dir).glob(f"*_{tag}_*.json"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise RuntimeError(f"No JSON found for {tag} in {output_dir}")
        json_files.append(str(candidates[-1]))
        print(f"  -> {candidates[-1].name}")

    print(f"\n{'='*60}")
    print("COMPARING")
    print("=" * 60)
    compare_cuda_versions(json_files[0], json_files[1], output_dir)

# ============================================================
# ENTRY POINT
# ============================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collect",  action="store_true",
                   help="Collect activations with currently installed torch and save JSON")
    p.add_argument("--compare",  nargs=2, metavar=("FILE1", "FILE2"),
                   help="Compare two previously collected JSON files")
    p.add_argument("--v1",  default=DEFAULT_V1, help=f"First CUDA tag  (default: {DEFAULT_V1})")
    p.add_argument("--v2",  default=DEFAULT_V2, help=f"Second CUDA tag (default: {DEFAULT_V2})")
    p.add_argument("--out", default=OUTPUT_DIR, help=f"Output directory (default: {OUTPUT_DIR})")
    args = p.parse_args()

    if args.collect:
        run_collection(args.out)
    elif args.compare:
        compare_cuda_versions(args.compare[0], args.compare[1], args.out)
    else:
        orchestrate(args.v1, args.v2, args.out)


if __name__ == "__main__":
    main()
