#!/usr/bin/env python3
"""
Cross-Hardware Prefill vs Decode Detectability Experiment (Transformers)

Tests whether batch size claims can be verified across different GPU architectures
using floating-point forensics (key vectors and logprobs).

Workflow:
1. Run on Machine A (e.g., A100) with TEACHER_FORCING = False
   → Generates tokens, extracts prefill + decode signals, saves to JSON
2. Copy JSON to Machine B (e.g., H100)
3. Run on Machine B with TEACHER_FORCING = True
   → Teacher-forces A's tokens, extracts signals, compares
"""

import os
os.environ['HF_HOME'] = '/workspace/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/huggingface_cache'

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from datetime import datetime
import json
import socket
import platform
import sys
import glob
import PyPDF2

# ============================================================================
# CONFIGURATION
# ============================================================================

TEACHER_FORCING = False
REFERENCE_FILE = "/workspace/experiments/decode_reference.json"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
CACHE_DIR = '/workspace/huggingface_cache'
ATTN_IMPLEMENTATION = "sdpa"  # Options: "eager", "sdpa", "flash_attention_2"

BATCH_SIZES = [1, 2, 3, 4, 5, 8, 9, 16, 17]
LAYER_INDICES = [28]  # Last layer only
MAX_NEW_TOKENS = 20
TOKENS_PER_SLICE = 1000
NUM_REFERENCES = 3

# Threshold for considering two batch sizes "equivalent" (same kernel)
EQUIVALENCE_THRESHOLD = 1e-9

# Will be initialized from PDF in main()
REFERENCE_SEQUENCES = None
DUMMY_SETS = None

# ============================================================================
# LOGGING SETUP
# ============================================================================

LOG_FILE = None

def setup_logging(output_dir='/workspace/experiments'):
    """Setup logging to file."""
    global LOG_FILE
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "verify" if TEACHER_FORCING else "generate"
    log_path = os.path.join(output_dir, f"experiment_{mode}_{timestamp}.txt")
    LOG_FILE = open(log_path, 'w')
    return log_path

def log_print(*args, **kwargs):
    """Print to both console and log file."""
    print(*args, **kwargs)
    if LOG_FILE:
        log_kwargs = {k: v for k, v in kwargs.items() if k != 'file'}
        print(*args, **log_kwargs, file=LOG_FILE)
        LOG_FILE.flush()

def close_logging():
    """Close log file."""
    global LOG_FILE
    if LOG_FILE:
        LOG_FILE.close()
        LOG_FILE = None

# ============================================================================
# PDF LOADING
# ============================================================================

def load_pdf_text(pdf_path):
    """Load text content from a PDF file."""
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.strip()

def create_sequences_from_pdf(tokenizer, num_references=NUM_REFERENCES):
    """
    Load all PDFs and split into equal-length slices.
    Returns REFERENCE_SEQUENCES and DUMMY_SETS dictionaries.
    """
    pdf_files = glob.glob("/workspace/*.pdf")
    if not pdf_files:
        pdf_files = glob.glob("*.pdf")
    if not pdf_files:
        raise FileNotFoundError("No PDF files found")

    log_print(f"Found {len(pdf_files)} PDF(s)")

    all_tokens = []
    for pdf_path in pdf_files:
        log_print(f"  Loading: {pdf_path}")
        text = load_pdf_text(pdf_path)
        tokens = tokenizer.encode(text, add_special_tokens=True)
        all_tokens.extend(tokens)
        log_print(f"    → {len(tokens)} tokens")

    log_print(f"Total tokens: {len(all_tokens)}")

    max_batch_size = max(BATCH_SIZES)
    slices_needed = num_references * max_batch_size
    tokens_needed = slices_needed * TOKENS_PER_SLICE

    if len(all_tokens) < tokens_needed:
        raise ValueError(f"Need {tokens_needed} tokens but only have {len(all_tokens)}")

    log_print(f"Creating {slices_needed} slices of {TOKENS_PER_SLICE} tokens each")

    slices = []
    for i in range(slices_needed):
        start = i * TOKENS_PER_SLICE
        end = start + TOKENS_PER_SLICE
        slice_tokens = all_tokens[start:end]
        slice_text = tokenizer.decode(slice_tokens)
        slices.append(slice_text)

    reference_sequences = {}
    dummy_sets = {}

    for ref_idx in range(num_references):
        ref_name = f"ref_{ref_idx}"
        base_idx = ref_idx * max_batch_size
        reference_sequences[ref_name] = slices[base_idx]
        dummy_sets[ref_name] = slices[base_idx + 1 : base_idx + max_batch_size]

    return reference_sequences, dummy_sets

# ============================================================================
# SYSTEM INFO
# ============================================================================

def collect_system_info():
    """Collect comprehensive environment information."""
    import transformers
    
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "cudnn_version": str(torch.backends.cudnn.version()) if torch.cuda.is_available() else "N/A",
        "transformers_version": transformers.__version__,
        "numpy_version": np.__version__,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    try:
        import flash_attn
        info["flash_attn_version"] = flash_attn.__version__
    except ImportError:
        info["flash_attn_version"] = "N/A"
        
    return info

def validate_environment_match(reference_env, verifier_env):
    """
    Validate that software environments match between reference and verifier.
    """
    log_print("\n" + "="*80)
    log_print("ENVIRONMENT VALIDATION")
    log_print("="*80)
    
    container_fields = ['python_version', 'cuda_version', 'cudnn_version']
    pip_fields = ['torch_version', 'transformers_version', 'numpy_version', 'flash_attn_version']
    config_fields = ['attn_implementation']
    expected_different = ['gpu_name', 'hostname']
    
    container_mismatches = []
    pip_mismatches = []
    config_mismatches = []
    
    log_print("\nScript configuration:")
    for field in config_fields:
        ref_val = reference_env.get(field, 'N/A')
        ver_val = verifier_env.get(field, 'N/A')
        if ref_val == ver_val:
            log_print(f"  ✓ {field}: {ref_val}")
        else:
            log_print(f"  ✗ {field}: reference={ref_val}, verifier={ver_val}")
            config_mismatches.append((field, ref_val, ver_val))

    log_print("\nContainer-level dependencies:")
    for field in container_fields:
        ref_val = reference_env.get(field, 'N/A')
        ver_val = verifier_env.get(field, 'N/A')
        if ref_val == ver_val:
            log_print(f"  ✓ {field}: {ref_val}")
        else:
            log_print(f"  ✗ {field}: reference={ref_val}, verifier={ver_val}")
            container_mismatches.append((field, ref_val, ver_val))

    log_print("\nPip-installable packages:")
    for field in pip_fields:
        ref_val = reference_env.get(field, 'N/A')
        ver_val = verifier_env.get(field, 'N/A')
        
        if field == 'flash_attn_version':
            ref_attn = reference_env.get('attn_implementation', '')
            ver_attn = verifier_env.get('attn_implementation', '')
            if ref_attn != 'flash_attention_2' and ver_attn != 'flash_attention_2':
                log_print(f"  - {field}: not using flash_attention_2 (skip)")
                continue
            if ref_val == 'N/A' or ver_val == 'N/A':
                log_print(f"  ✗ {field}: reference={ref_val}, verifier={ver_val}")
                pip_mismatches.append((field, ref_val, ver_val))
                continue
        
        if ref_val == 'N/A' and ver_val == 'N/A':
            log_print(f"  - {field}: not installed (OK)")
            continue
            
        if ref_val == ver_val:
            log_print(f"  ✓ {field}: {ref_val}")
        else:
            log_print(f"  ✗ {field}: reference={ref_val}, verifier={ver_val}")
            pip_mismatches.append((field, ref_val, ver_val))

    log_print("\nExpected differences (hardware):")
    for field in expected_different:
        ref_val = reference_env.get(field, 'N/A')
        ver_val = verifier_env.get(field, 'N/A')
        if ref_val != ver_val:
            log_print(f"  ✓ {field}: reference={ref_val}, verifier={ver_val}")
        else:
            log_print(f"  ⚠ {field}: SAME ({ref_val}) - are you on different hardware?")

    if not container_mismatches and not pip_mismatches and not config_mismatches:
        log_print("\n" + "-"*60)
        log_print("✓ ENVIRONMENT VALIDATION PASSED")
        return {'valid': True, 'mismatches': []}
    
    log_print("\n" + "="*80)
    log_print("✗ ENVIRONMENT MISMATCH - FIX REQUIRED")
    sys.exit(1)

# ============================================================================
# SIGNAL EXTRACTION
# ============================================================================

def extract_signals_from_output(outputs, layer_indices, position=-1):
    """
    Extract key vectors and logprobs from element 0 at specified position.
    """
    signals = {
        'key_vectors': {},
        'logprobs': {}
    }

    for layer_idx in layer_indices:
        layer_keys = outputs.past_key_values[layer_idx - 1][0]
        token_keys = layer_keys[0, :, position, :]
        key_dim = token_keys.shape[0] * token_keys.shape[1]
        key_vector = token_keys.reshape(key_dim).cpu().clone()
        signals['key_vectors'][f'layer_{layer_idx}'] = key_vector.float().numpy().tolist()

    logits = outputs.logits[0, position, :]
    log_probs = F.log_softmax(logits, dim=-1)
    top_k = torch.topk(log_probs, k=20)

    signals['logprobs'] = {
        'token_ids': top_k.indices.cpu().tolist(),
        'log_probs': top_k.values.cpu().tolist()
    }

    return signals

def extract_prefill_signals(outputs, layer_indices, positions=[-3, -2, -1]):
    """Extract signals from multiple positions during prefill."""
    prefill_signals = {}
    for pos in positions:
        pos_label = f"pos_{pos}"
        prefill_signals[pos_label] = extract_signals_from_output(outputs, layer_indices, position=pos)
    return prefill_signals

def extract_signals_for_token_ids(outputs, layer_indices, token_ids, position=-1):
    """Extract signals for SPECIFIC token IDs (used in verification)."""
    signals = {
        'key_vectors': {},
        'logprobs': {}
    }

    for layer_idx in layer_indices:
        layer_keys = outputs.past_key_values[layer_idx - 1][0]
        token_keys = layer_keys[0, :, position, :]
        key_dim = token_keys.shape[0] * token_keys.shape[1]
        key_vector = token_keys.reshape(key_dim).cpu().clone()
        signals['key_vectors'][f'layer_{layer_idx}'] = key_vector.float().numpy().tolist()

    logits = outputs.logits[0, position, :]
    log_probs = F.log_softmax(logits, dim=-1)
    token_ids_tensor = torch.tensor(token_ids, device=logits.device)
    selected_logprobs = log_probs[token_ids_tensor]

    signals['logprobs'] = {
        'token_ids': token_ids,
        'log_probs': selected_logprobs.cpu().tolist()
    }

    return signals

def extract_prefill_signals_for_token_ids(outputs, layer_indices, ref_prefill_signals, positions=[-3, -2, -1]):
    """Extract prefill signals using reference token IDs."""
    prefill_signals = {}
    for pos in positions:
        pos_label = f"pos_{pos}"
        if pos_label in ref_prefill_signals:
            ref_token_ids = ref_prefill_signals[pos_label]['logprobs']['token_ids']
            prefill_signals[pos_label] = extract_signals_for_token_ids(
                outputs, layer_indices, ref_token_ids, position=pos
            )
    return prefill_signals

# ============================================================================
# DECODE GENERATION (TEACHER_FORCING = False)
# ============================================================================

def compute_min_length_across_batches(ref_text, ref_name, tokenizer, batch_sizes):
    """Pre-compute minimum sequence length across all batch configurations."""
    ref_dummies = DUMMY_SETS[ref_name]
    min_length = float('inf')
    
    for batch_size in batch_sizes:
        if batch_size == 1:
            batch_texts = [ref_text]
        else:
            batch_texts = [ref_text] + ref_dummies[:batch_size-1]
        
        token_lengths = [len(tokenizer.encode(t, add_special_tokens=True)) for t in batch_texts]
        min_length = min(min_length, min(token_lengths))
        
    return min_length

def run_decode_with_extraction(model, tokenizer, ref_text, ref_name, batch_size, 
                               layer_indices, forced_length=None):
    """
    Run decode generation and extract signals from last 3 generation steps.
    """
    torch.cuda.empty_cache()
    
    ref_dummies = DUMMY_SETS[ref_name]
    if batch_size == 1:
        batch_texts = [ref_text]
    else:
        batch_texts = [ref_text] + ref_dummies[:batch_size-1]
    
    all_token_ids = [tokenizer.encode(t, add_special_tokens=True) for t in batch_texts]
    
    if forced_length is not None:
        min_length = forced_length
    else:
        min_length = min(len(ids) for ids in all_token_ids)
        
    truncated_token_ids = [ids[:min_length] for ids in all_token_ids]
    
    input_ids = torch.tensor(truncated_token_ids, dtype=torch.long, device='cuda')
    attention_mask = torch.ones_like(input_ids)
    
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    
    prompt_length = input_ids.shape[1]
    log_print(f"      Prompt: {prompt_length} tokens", end="")
    
    all_batch_generated_ids = [[] for _ in range(batch_size)]
    generation_signals = []
    
    # FIRST STEP: Prefill
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        
    past_kv = outputs.past_key_values
    prefill_signals = extract_prefill_signals(outputs, layer_indices, positions=[-3, -2, -1])
    
    next_tokens = outputs.logits[:, -1, :].argmax(dim=-1)
    for batch_idx in range(batch_size):
        all_batch_generated_ids[batch_idx].append(next_tokens[batch_idx].item())
        
    signals = extract_signals_from_output(outputs, layer_indices, position=-1)
    absolute_position_index = inputs['input_ids'].shape[1] - 1
    
    generation_signals.append({
        'step': 0,
        'absolute_position': absolute_position_index,
        'signals': signals
    })
    
    attention_mask = torch.cat([
        inputs['attention_mask'], 
        torch.ones((inputs['attention_mask'].shape[0], 1), device='cuda')
    ], dim=1)
    
    # SUBSEQUENT STEPS
    for step in range(1, MAX_NEW_TOKENS):
        new_inputs = {
            'input_ids': next_tokens.unsqueeze(1),
            'attention_mask': attention_mask,
            'past_key_values': past_kv,
            'use_cache': True
        }
        
        with torch.no_grad():
            outputs = model(**new_inputs)
        
        past_kv = outputs.past_key_values
        
        next_tokens = outputs.logits[:, -1, :].argmax(dim=-1)
        for batch_idx in range(batch_size):
            all_batch_generated_ids[batch_idx].append(next_tokens[batch_idx].item())
            
        signals = extract_signals_from_output(outputs, layer_indices, position=-1)
        current_cache_length = past_kv[0][0].shape[2]
        absolute_position_index = current_cache_length - 1
        
        generation_signals.append({
            'step': step,
            'absolute_position': absolute_position_index,
            'signals': signals
        })
        
        attention_mask = torch.cat([
            attention_mask, 
            torch.ones((attention_mask.shape[0], 1), device='cuda')
        ], dim=1)
        
        if all_batch_generated_ids[0][-1] == tokenizer.eos_token_id:
            break
            
    # Extract last 3 decode signals
    num_generated = len(generation_signals)
    if num_generated >= 3:
        last_3_signals = {
            'pos_-3': generation_signals[-3],
            'pos_-2': generation_signals[-2],
            'pos_-1': generation_signals[-1]
        }
    elif num_generated == 2:
        last_3_signals = {
            'pos_-2': generation_signals[-2],
            'pos_-1': generation_signals[-1]
        }
    elif num_generated == 1:
        last_3_signals = {
            'pos_-1': generation_signals[-1]
        }
    else:
        last_3_signals = {}
        
    del outputs, inputs, past_kv
    torch.cuda.empty_cache()
    
    final_length = prompt_length + num_generated
    log_print(f" → Final: {final_length} tokens ({num_generated} generated)")
    
    return {
        'generated_ids': all_batch_generated_ids[0],
        'all_batch_generated_ids': all_batch_generated_ids,
        'prompt_token_ids': truncated_token_ids,
        'prompt_length': prompt_length,
        'prefill_signals': prefill_signals,
        'decode_signals': last_3_signals,
        'num_generated': num_generated
    }

# ============================================================================
# TEACHER-FORCED DECODE (TEACHER_FORCING = True)
# ============================================================================

def run_teacher_forced_decode(model, tokenizer, ref_name, reference_data, 
                              verify_batch_size, layer_indices, is_diagonal):
    """
    Teacher-forced decode: feed reference tokens, extract signals.
    """
    torch.cuda.empty_cache()
    
    ref_prompt_ids = reference_data['prompt_token_ids'][0]
    ref_generated_ids = reference_data['generated_ids']
    ref_batch_size = len(reference_data['prompt_token_ids'])
    
    log_print(f"      Prompt: {len(ref_prompt_ids)}, Gen: {len(ref_generated_ids)}", end="")
    
    if is_diagonal:
        log_print(f", exact neighbors (bs={ref_batch_size})", end="")
        batch_prompt_ids = reference_data['prompt_token_ids']
        batch_generated_ids = reference_data['all_batch_generated_ids']
        actual_batch_size = ref_batch_size
    else:
        log_print(f", arb neighbors (bs={verify_batch_size})", end="")
        batch_prompt_ids = [ref_prompt_ids]
        batch_generated_ids = [ref_generated_ids]
        
        ref_dummies = DUMMY_SETS[ref_name]
        for i in range(verify_batch_size - 1):
            dummy_ids = tokenizer.encode(ref_dummies[i], add_special_tokens=True)
            dummy_ids = dummy_ids[:len(ref_prompt_ids)]
            if len(dummy_ids) < len(ref_prompt_ids):
                dummy_ids = dummy_ids + [tokenizer.pad_token_id or 0] * (len(ref_prompt_ids) - len(dummy_ids))
            batch_prompt_ids.append(dummy_ids)
            batch_generated_ids.append([])
        
        actual_batch_size = verify_batch_size
        
    input_ids = torch.tensor(batch_prompt_ids, dtype=torch.long, device='cuda')
    attention_mask = torch.ones_like(input_ids)
    
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    generation_signals = []
    num_steps = len(ref_generated_ids)
    
    # FIRST STEP: Prefill
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        
    past_kv = outputs.past_key_values
    ref_prefill_signals = reference_data['prefill_signals']
    prefill_signals = extract_prefill_signals_for_token_ids(
        outputs, layer_indices, ref_prefill_signals, positions=[-3, -2, -1]
    )
    
    ref_decode_signals = reference_data['decode_signals']
    ref_step_data = list(ref_decode_signals.values())[0] if ref_decode_signals else None
    if ref_step_data:
        ref_token_ids = ref_step_data['signals']['logprobs']['token_ids']
        signals = extract_signals_for_token_ids(outputs, layer_indices, ref_token_ids, position=-1)
    else:
        signals = extract_signals_from_output(outputs, layer_indices, position=-1)
        
    absolute_position_index = inputs['input_ids'].shape[1] - 1
    generation_signals.append({
        'step': 0,
        'absolute_position': absolute_position_index,
        'signals': signals
    })
    
    # Prepare next tokens
    if is_diagonal:
        next_tokens = torch.tensor(
            [batch_generated_ids[i][0] for i in range(actual_batch_size)],
            dtype=torch.long, device='cuda'
        )
    else:
        next_tokens_list = [ref_generated_ids[0]]
        argmax_tokens = outputs.logits[1:, -1, :].argmax(dim=-1)
        for i in range(actual_batch_size - 1):
            next_tokens_list.append(argmax_tokens[i].item())
            batch_generated_ids[i + 1].append(argmax_tokens[i].item())
        next_tokens = torch.tensor(next_tokens_list, dtype=torch.long, device='cuda')
        
    attention_mask = torch.cat([
        inputs['attention_mask'], 
        torch.ones((actual_batch_size, 1), device='cuda')
    ], dim=1)
    
    # SUBSEQUENT STEPS
    for step in range(1, num_steps):
        new_inputs = {
            'input_ids': next_tokens.unsqueeze(1),
            'attention_mask': attention_mask,
            'past_key_values': past_kv,
            'use_cache': True
        }
        
        with torch.no_grad():
            outputs = model(**new_inputs)
        
        past_kv = outputs.past_key_values
        
        ref_signals_list = list(ref_decode_signals.values())
        if step < len(ref_signals_list):
            ref_token_ids = ref_signals_list[step]['signals']['logprobs']['token_ids']
            signals = extract_signals_for_token_ids(outputs, layer_indices, ref_token_ids, position=-1)
        else:
            signals = extract_signals_from_output(outputs, layer_indices, position=-1)
            
        current_cache_length = past_kv[0][0].shape[2]
        absolute_position_index = current_cache_length - 1
        
        generation_signals.append({
            'step': step,
            'absolute_position': absolute_position_index,
            'signals': signals
        })
        
        if step < num_steps - 1:
            if is_diagonal:
                next_tokens = torch.tensor(
                    [batch_generated_ids[i][step] for i in range(actual_batch_size)],
                    dtype=torch.long, device='cuda'
                )
            else:
                next_tokens_list = [ref_generated_ids[step]]
                argmax_tokens = outputs.logits[1:, -1, :].argmax(dim=-1)
                for i in range(actual_batch_size - 1):
                    next_tokens_list.append(argmax_tokens[i].item())
                    batch_generated_ids[i + 1].append(argmax_tokens[i].item())
                next_tokens = torch.tensor(next_tokens_list, dtype=torch.long, device='cuda')
        
        attention_mask = torch.cat([
            attention_mask, 
            torch.ones((actual_batch_size, 1), device='cuda')
        ], dim=1)
        
    # Extract last 3
    num_generated = len(generation_signals)
    if num_generated >= 3:
        last_3_signals = {
            'pos_-3': generation_signals[-3],
            'pos_-2': generation_signals[-2],
            'pos_-1': generation_signals[-1]
        }
    elif num_generated == 2:
        last_3_signals = {
            'pos_-2': generation_signals[-2],
            'pos_-1': generation_signals[-1]
        }
    elif num_generated == 1:
        last_3_signals = {
            'pos_-1': generation_signals[-1]
        }
    else:
        last_3_signals = {}
        
    del outputs, inputs, past_kv
    torch.cuda.empty_cache()
    
    log_print(f" → {num_generated} steps")
    
    return {
        'prefill_signals': prefill_signals,
        'decode_signals': last_3_signals,
        'num_generated': num_generated
    }

# ============================================================================
# ANALYSIS
# ============================================================================

def check_token_consistency(decode_measurements, tokenizer):
    """Verify element 0 generates identical tokens across all batch sizes."""
    log_print("\n" + "="*80)
    log_print("TOKEN GENERATION CONSISTENCY CHECK")
    log_print("="*80)
    
    tokens_by_bs = {}
    for bs, data in decode_measurements.items():
        tokens_by_bs[bs] = data['generated_ids']
        
    bs_list = sorted(tokens_by_bs.keys())
    reference_tokens = tokens_by_bs[bs_list[0]]
    
    all_same = True
    log_print("\nGenerated tokens by batch size:")
    for bs in bs_list:
        tokens = tokens_by_bs[bs]
        match_str = "✓" if tokens == reference_tokens else "✗ DIFFERENT"
        decoded_text = tokenizer.decode(tokens)
        log_print(f"  bs={bs}:")
        log_print(f"    IDs:  {tokens}")
        log_print(f"    Text: {repr(decoded_text)}")
        log_print(f"    {match_str}")
        if tokens != reference_tokens:
            all_same = False
            
    if all_same:
        log_print("\n✓ Element 0 generates IDENTICAL tokens across all batch sizes")
    else:
        log_print("\n⚠ Element 0 generates DIFFERENT tokens across batch sizes")
        
    return all_same

def compute_l2_distance(vec1, vec2):
    """Compute L2 distance between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.linalg.norm(v1 - v2))

def compute_logprob_distance(logprobs1, logprobs2):
    """
    Compute L2 distance between logprob distributions.
    Uses top 5 token IDs from first signal as canonical (stored top 20 as buffer).
    """
    # Top 5 from first signal as canonical
    canonical_ids = logprobs1['token_ids'][:5]
    vec1 = logprobs1['log_probs'][:5]
    
    # Look up these token IDs in second signal
    map2 = dict(zip(logprobs2['token_ids'], logprobs2['log_probs']))
    
    vec2 = []
    for tid in canonical_ids:
        if tid in map2:
            vec2.append(map2[tid])
        else:
            return float('inf')  # token not in top-20 of other run
    
    return float(np.linalg.norm(np.array(vec1) - np.array(vec2)))

def compare_signals(signals1, signals2, layer_indices):
    """Compare two signal sets, return distances."""
    common_positions = set(signals1.keys()) & set(signals2.keys())
    
    all_key_dists = []
    all_logprob_dists = []
    
    for pos_label in common_positions:
        sig1 = signals1[pos_label]['signals'] if 'signals' in signals1[pos_label] else signals1[pos_label]
        sig2 = signals2[pos_label]['signals'] if 'signals' in signals2[pos_label] else signals2[pos_label]
        
        # Key vectors
        for layer_name in sig1['key_vectors'].keys():
            dist = compute_l2_distance(
                sig1['key_vectors'][layer_name],
                sig2['key_vectors'][layer_name]
            )
            all_key_dists.append(dist)
            
        # Logprobs
        dist = compute_logprob_distance(sig1['logprobs'], sig2['logprobs'])
        all_logprob_dists.append(dist)
        
    return {
        'key_vectors_max': max(all_key_dists) if all_key_dists else 0.0,
        'key_vectors_mean': np.mean(all_key_dists) if all_key_dists else 0.0,
        'logprobs_max': max(all_logprob_dists) if all_logprob_dists else 0.0,
        'logprobs_mean': np.mean(all_logprob_dists) if all_logprob_dists else 0.0
    }

def find_equivalent_pairs(matrix, batch_sizes, threshold=EQUIVALENCE_THRESHOLD):
    """
    Find pairs of batch sizes that produce equivalent results (same kernel).
    Returns list of (bs1, bs2) tuples where bs1 < bs2.
    """
    equivalent_pairs = []
    n_bs = len(batch_sizes)
    
    for i in range(n_bs):
        for j in range(i + 1, n_bs):
            if matrix[i, j] < threshold:
                equivalent_pairs.append((batch_sizes[i], batch_sizes[j]))
    
    return equivalent_pairs

def format_kernel_classes(equivalent_pairs, batch_sizes):
    """
    Group batch sizes into kernel equivalence classes.
    """
    parent = {bs: bs for bs in batch_sizes}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    for bs1, bs2 in equivalent_pairs:
        union(bs1, bs2)
    
    groups = {}
    for bs in batch_sizes:
        root = find(bs)
        if root not in groups:
            groups[root] = set()
        groups[root].add(bs)
    
    return list(groups.values())

def analyze_within_hardware(measurements, batch_sizes, layer_indices, signal_source='decode'):
    """Analyze within-hardware batch size effects."""
    log_print("\n" + "="*80)
    log_print(f"WITHIN-HARDWARE BATCH SIZE EFFECTS ({signal_source.upper()})")
    log_print("="*80)
    
    by_ref = {}
    for m in measurements:
        ref = m['ref_name']
        if ref not in by_ref:
            by_ref[ref] = {}
        signals_key = 'prefill_signals' if signal_source == 'prefill' else 'decode_signals'
        by_ref[ref][m['batch_size']] = m[signals_key]

    all_key_matrices = []
    all_logprob_matrices = []
    n_bs = len(batch_sizes)

    for ref_name in sorted(by_ref.keys()):
        log_print(f"\n{'='*80}")
        log_print(f"{ref_name.upper()}")
        log_print("="*80)
        
        ref_data = by_ref[ref_name]
        matrix_key = np.zeros((n_bs, n_bs))
        matrix_logprob = np.zeros((n_bs, n_bs))
        
        for i, bs1 in enumerate(batch_sizes):
            for j, bs2 in enumerate(batch_sizes):
                if bs1 not in ref_data or bs2 not in ref_data:
                    continue
                if i == j:
                    matrix_key[i, j] = 0.0
                    matrix_logprob[i, j] = 0.0
                else:
                    distances = compare_signals(ref_data[bs1], ref_data[bs2], layer_indices)
                    matrix_key[i, j] = distances['key_vectors_mean']
                    matrix_logprob[i, j] = distances['logprobs_mean']
        
        header = "       " + "".join([f"bs={bs:>3} " for bs in batch_sizes])
        log_print("\nKey Vectors (mean L2 distance):")
        log_print(header)
        for i, bs in enumerate(batch_sizes):
            row_str = f"bs={bs:<3}"
            for j in range(n_bs):
                row_str += f"  {matrix_key[i,j]:6.2e}"
            log_print(row_str)
        
        log_print("\nLogprobs (mean L2 distance):")
        log_print(header)
        for i, bs in enumerate(batch_sizes):
            row_str = f"bs={bs:<3}"
            for j in range(n_bs):
                row_str += f"  {matrix_logprob[i,j]:6.2e}"
            log_print(row_str)
            
        all_key_matrices.append(matrix_key)
        all_logprob_matrices.append(matrix_logprob)

    # Aggregate
    avg_key_matrix = np.mean(all_key_matrices, axis=0)
    avg_logprob_matrix = np.mean(all_logprob_matrices, axis=0)
    
    log_print("\n" + "="*80)
    log_print("AGGREGATE (average across references):")
    log_print("="*80)
    
    header = "       " + "".join([f"bs={bs:>3} " for bs in batch_sizes])
    log_print("\nKey Vectors (mean L2 distance):")
    log_print(header)
    for i, bs in enumerate(batch_sizes):
        row_str = f"bs={bs:<3}"
        for j in range(n_bs):
            row_str += f"  {avg_key_matrix[i,j]:6.2e}"
        log_print(row_str)
    
    log_print("\nLogprobs (mean L2 distance):")
    log_print(header)
    for i, bs in enumerate(batch_sizes):
        row_str = f"bs={bs:<3}"
        for j in range(n_bs):
            row_str += f"  {avg_logprob_matrix[i,j]:6.2e}"
        log_print(row_str)
    
    # Use key vectors for equivalence detection (more sensitive)
    off_diag_key = avg_key_matrix[np.triu_indices(n_bs, k=1)]
    off_diag_logprob = avg_logprob_matrix[np.triu_indices(n_bs, k=1)]
    
    key_mean = np.mean(off_diag_key)
    logprob_mean = np.mean(off_diag_logprob)
    
    log_print(f"\nOff-diagonal stats:")
    log_print(f"  Key vectors - Mean: {key_mean:.2e}, Range: [{np.min(off_diag_key):.2e}, {np.max(off_diag_key):.2e}]")
    log_print(f"  Logprobs - Mean: {logprob_mean:.2e}, Range: [{np.min(off_diag_logprob):.2e}, {np.max(off_diag_logprob):.2e}]")
    
    # Find equivalent pairs using key vectors
    equivalent_pairs = find_equivalent_pairs(avg_key_matrix, batch_sizes)
    kernel_classes = format_kernel_classes(equivalent_pairs, batch_sizes)
    
    # Check if all zeros
    zero_count = np.sum(off_diag_key < EQUIVALENCE_THRESHOLD)
    total_count = len(off_diag_key)
    
    if zero_count == total_count:
        log_print(f"\n⚠ WARNING: {zero_count}/{total_count} comparisons are EXACTLY ZERO")
        log_print("  All batch sizes produce identical results (single kernel class)")
    elif zero_count > 0:
        log_print(f"\n⚠ NOTE: {zero_count}/{total_count} comparisons are equivalent (< {EQUIVALENCE_THRESHOLD})")
    
    # Display kernel classes
    log_print(f"\nKernel equivalence classes:")
    for i, cls in enumerate(kernel_classes):
        log_print(f"  Class {i+1}: {sorted(cls)}")
    
    if equivalent_pairs:
        log_print(f"\nEquivalent pairs (will be excluded from cross-hardware signal):")
        for bs1, bs2 in equivalent_pairs:
            log_print(f"  ({bs1}, {bs2})")
    
    threshold = 1e-10
    if key_mean > threshold and logprob_mean > threshold:
        log_print("\n✓ SANITY CHECK PASSED")
    else:
        log_print("\n✗ SANITY CHECK FAILED (No variation)")
        
    return {
        'key_matrix': avg_key_matrix.tolist(),
        'logprob_matrix': avg_logprob_matrix.tolist(),
        'per_reference_key_matrices': [m.tolist() for m in all_key_matrices],
        'per_reference_logprob_matrices': [m.tolist() for m in all_logprob_matrices],
        'key_vectors_mean': float(key_mean),
        'logprobs_mean': float(logprob_mean),
        'equivalent_pairs': equivalent_pairs,
        'kernel_classes': [sorted(list(cls)) for cls in kernel_classes]
    }

def analyze_cross_hardware_matrix(comparison_results, batch_sizes, layer_indices, 
                                   signal_source='decode', equivalent_pairs=None):
    """
    Analyze the comparison matrix and determine detectability.
    Excludes equivalent pairs from SNR signal calculation.
    """
    log_print("\n" + "="*80)
    log_print(f"CROSS-HARDWARE BATCH SIZE DETECTABILITY ({signal_source.upper()})")
    log_print("="*80)
    
    if equivalent_pairs is None:
        equivalent_pairs = []
    
    # Convert to set of both orderings for easy lookup
    equiv_set = set()
    for bs1, bs2 in equivalent_pairs:
        equiv_set.add((bs1, bs2))
        equiv_set.add((bs2, bs1))
    
    dist_key = 'prefill_distances' if signal_source == 'prefill' else 'decode_distances'
    
    by_ref = {}
    for result in comparison_results:
        ref = result['ref_name']
        if ref not in by_ref:
            by_ref[ref] = {}
        key = (result['claimed_batch_size'], result['verify_batch_size'])
        by_ref[ref][key] = result

    all_key_matrices = []
    all_logprob_matrices = []
    n_bs = len(batch_sizes)

    # Per-reference matrices
    for ref_name in sorted(by_ref.keys()):
        log_print(f"\n{'='*80}")
        log_print(f"{ref_name.upper()}")
        log_print("="*80)
        
        ref_data = by_ref[ref_name]
        matrix_key = np.zeros((n_bs, n_bs))
        matrix_logprob = np.zeros((n_bs, n_bs))
        
        for i, claimed_bs in enumerate(batch_sizes):
            for j, verify_bs in enumerate(batch_sizes):
                key = (claimed_bs, verify_bs)
                if key in ref_data:
                    matrix_key[i, j] = ref_data[key][dist_key]['key_vectors_mean']
                    matrix_logprob[i, j] = ref_data[key][dist_key]['logprobs_mean']
        
        # Display matrices
        header = "              " + "".join([f"v_bs={bs:>3} " for bs in batch_sizes])
        
        log_print(f"\nKey Vectors (mean L2):")
        log_print(header)
        for i, claimed_bs in enumerate(batch_sizes):
            row_str = f"c_bs={claimed_bs:<3} "
            for j in range(n_bs):
                row_str += f"  {matrix_key[i,j]:8.2e}"
            log_print(row_str)
        
        log_print(f"\nLogprobs (mean L2):")
        log_print(header)
        for i, claimed_bs in enumerate(batch_sizes):
            row_str = f"c_bs={claimed_bs:<3} "
            for j in range(n_bs):
                row_str += f"  {matrix_logprob[i,j]:8.2e}"
            log_print(row_str)
            
        all_key_matrices.append(matrix_key)
        all_logprob_matrices.append(matrix_logprob)

    # Aggregate matrices
    avg_key_matrix = np.mean(all_key_matrices, axis=0)
    avg_logprob_matrix = np.mean(all_logprob_matrices, axis=0)
    
    log_print("\n" + "="*80)
    log_print("AGGREGATE (average across references):")
    log_print("="*80)
    
    header = "              " + "".join([f"v_bs={bs:>3} " for bs in batch_sizes])
    
    log_print(f"\nKey Vectors (mean L2):")
    log_print(header)
    for i, claimed_bs in enumerate(batch_sizes):
        row_str = f"c_bs={claimed_bs:<3} "
        for j in range(n_bs):
            row_str += f"  {avg_key_matrix[i,j]:8.2e}"
        log_print(row_str)
    
    log_print(f"\nLogprobs (mean L2):")
    log_print(header)
    for i, claimed_bs in enumerate(batch_sizes):
        row_str = f"c_bs={claimed_bs:<3} "
        for j in range(n_bs):
            row_str += f"  {avg_logprob_matrix[i,j]:8.2e}"
        log_print(row_str)
    
    # Compute statistics for both signal types
    results = {}
    
    for signal_type, avg_matrix in [('key_vectors', avg_key_matrix), ('logprobs', avg_logprob_matrix)]:
        diagonal = [avg_matrix[i, i] for i in range(n_bs)]
        
        off_diagonal_all = []
        off_diagonal_meaningful = []
        excluded_pairs = []
        
        for i, bs1 in enumerate(batch_sizes):
            for j, bs2 in enumerate(batch_sizes):
                if i != j:
                    off_diagonal_all.append(avg_matrix[i, j])
                    if (bs1, bs2) in equiv_set:
                        excluded_pairs.append((bs1, bs2))
                    else:
                        off_diagonal_meaningful.append(avg_matrix[i, j])
        
        baseline_mean = np.mean(diagonal)
        signal_all_mean = np.mean(off_diagonal_all) if off_diagonal_all else 0.0
        signal_meaningful_mean = np.mean(off_diagonal_meaningful) if off_diagonal_meaningful else 0.0
        
        snr_all = signal_all_mean / baseline_mean if baseline_mean > 0 else float('inf')
        snr_meaningful = signal_meaningful_mean / baseline_mean if baseline_mean > 0 else float('inf')
        
        results[signal_type] = {
            'matrix': avg_matrix.tolist(),
            'baseline_mean': float(baseline_mean),
            'signal_all_mean': float(signal_all_mean),
            'signal_meaningful_mean': float(signal_meaningful_mean),
            'snr_all': float(snr_all),
            'snr_meaningful': float(snr_meaningful),
            'n_meaningful_pairs': len(off_diagonal_meaningful)
        }
    
    # Print SNR summary
    log_print("\n" + "="*80)
    log_print("SNR ANALYSIS")
    log_print("="*80)
    
    log_print(f"\nDiagonal (baseline = cross-hardware, same batch size):")
    log_print(f"  Key vectors: {results['key_vectors']['baseline_mean']:.2e}")
    log_print(f"  Logprobs: {results['logprobs']['baseline_mean']:.2e}")
    
    log_print(f"\nOff-diagonal (all pairs):")
    log_print(f"  Key vectors - Mean: {results['key_vectors']['signal_all_mean']:.2e}, SNR: {results['key_vectors']['snr_all']:.2f}×")
    log_print(f"  Logprobs - Mean: {results['logprobs']['signal_all_mean']:.2e}, SNR: {results['logprobs']['snr_all']:.2f}×")
    
    if equivalent_pairs:
        log_print(f"\nExcluded equivalent pairs (same kernel within-hardware):")
        for bs1, bs2 in equivalent_pairs:
            log_print(f"  ({bs1}, {bs2}) and ({bs2}, {bs1})")
        log_print(f"  Total excluded: {len(equivalent_pairs) * 2} cells")
    
    log_print(f"\nOff-diagonal (meaningful pairs only):")
    log_print(f"  Count: {results['key_vectors']['n_meaningful_pairs']}")
    if results['key_vectors']['n_meaningful_pairs'] > 0:
        log_print(f"  Key vectors - Mean: {results['key_vectors']['signal_meaningful_mean']:.2e}, SNR: {results['key_vectors']['snr_meaningful']:.2f}×")
        log_print(f"  Logprobs - Mean: {results['logprobs']['signal_meaningful_mean']:.2e}, SNR: {results['logprobs']['snr_meaningful']:.2f}×")
    else:
        log_print("  No meaningful pairs (all batch sizes are equivalent)")
    
    return {
        'per_reference_key_matrices': [m.tolist() for m in all_key_matrices],
        'per_reference_logprob_matrices': [m.tolist() for m in all_logprob_matrices],
        'key_vectors': results['key_vectors'],
        'logprobs': results['logprobs'],
        'excluded_pairs': equivalent_pairs
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    global REFERENCE_SEQUENCES, DUMMY_SETS
    
    log_path = setup_logging()
    system_info = collect_system_info()
    
    mode = "VERIFICATION (teacher-forcing)" if TEACHER_FORCING else "GENERATION (reference)"
    log_print("="*80)
    log_print(f"CROSS-HARDWARE BATCH SIZE DETECTABILITY - {mode}")
    log_print("="*80)
    
    log_print(f"\nSystem: {system_info['hostname']}")
    log_print(f"GPU: {system_info['gpu_name']}")
    log_print(f"Attention: {ATTN_IMPLEMENTATION}")
    log_print(f"Layers: {LAYER_INDICES}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    REFERENCE_SEQUENCES, DUMMY_SETS = create_sequences_from_pdf(tokenizer)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation=ATTN_IMPLEMENTATION
    )
    
    output_dir = '/workspace/experiments'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if TEACHER_FORCING:
        # VERIFICATION MODE
        with open(REFERENCE_FILE, 'r') as f:
            reference = json.load(f)
            
        ref_env = reference['metadata']['environment']
        validate_environment_match(ref_env, system_info)
        
        # Load equivalent pairs from reference
        prefill_sanity = reference.get('prefill_sanity_check', {})
        decode_sanity = reference.get('decode_sanity_check', {})
        
        prefill_equiv_pairs = [tuple(p) for p in prefill_sanity.get('equivalent_pairs', [])]
        decode_equiv_pairs = [tuple(p) for p in decode_sanity.get('equivalent_pairs', [])]
        
        log_print(f"\nLoaded equivalent pairs from reference:")
        log_print(f"  Prefill: {prefill_equiv_pairs}")
        log_print(f"  Decode: {decode_equiv_pairs}")
        
        comparison_results = []
        ref_by_key = {}
        for m in reference['measurements']:
            key = (m['ref_name'], m['batch_size'])
            ref_by_key[key] = m
            
        for ref_name in sorted(REFERENCE_SEQUENCES.keys()):
            log_print(f"\n{'='*80}")
            log_print(f"REFERENCE: {ref_name}")
            log_print("="*80)
            
            for claimed_bs in BATCH_SIZES:
                ref_key = (ref_name, claimed_bs)
                if ref_key not in ref_by_key:
                    log_print(f"  ⚠ No reference data for {ref_name} bs={claimed_bs}")
                    continue
                ref_data = ref_by_key[ref_key]
                
                log_print(f"\n  Claimed batch size: {claimed_bs}")
                
                for verify_bs in BATCH_SIZES:
                    is_diagonal = (claimed_bs == verify_bs)
                    
                    log_print(f"    Verify bs={verify_bs} ({'diag' if is_diagonal else 'off'}):", end="")
                    
                    verify_result = run_teacher_forced_decode(
                        model, tokenizer, ref_name, ref_data,
                        verify_bs, LAYER_INDICES, is_diagonal
                    )
                    
                    prefill_distances = compare_signals(
                        ref_data['prefill_signals'], verify_result['prefill_signals'], LAYER_INDICES
                    )
                    decode_distances = compare_signals(
                        ref_data['decode_signals'], verify_result['decode_signals'], LAYER_INDICES
                    )
                    
                    log_print(f"      Key: {decode_distances['key_vectors_mean']:.2e}, LP: {decode_distances['logprobs_mean']:.2e}")
                    
                    comparison_results.append({
                        'ref_name': ref_name,
                        'claimed_batch_size': claimed_bs,
                        'verify_batch_size': verify_bs,
                        'prefill_distances': prefill_distances,
                        'decode_distances': decode_distances,
                    })

        prefill_analysis = analyze_cross_hardware_matrix(
            comparison_results, BATCH_SIZES, LAYER_INDICES, 
            signal_source='prefill', equivalent_pairs=prefill_equiv_pairs
        )
        decode_analysis = analyze_cross_hardware_matrix(
            comparison_results, BATCH_SIZES, LAYER_INDICES,
            signal_source='decode', equivalent_pairs=decode_equiv_pairs
        )
        
        log_print("\n" + "="*80)
        log_print("PREFILL vs DECODE COMPARISON (meaningful SNR)")
        log_print("="*80)
        log_print(f"Prefill - Key: {prefill_analysis['key_vectors']['snr_meaningful']:.2f}×, LP: {prefill_analysis['logprobs']['snr_meaningful']:.2f}×")
        log_print(f"Decode  - Key: {decode_analysis['key_vectors']['snr_meaningful']:.2f}×, LP: {decode_analysis['logprobs']['snr_meaningful']:.2f}×")
        
        results = {
            'metadata': {
                'reference_environment': ref_env,
                'verifier_environment': system_info,
                'batch_sizes': BATCH_SIZES,
                'layer_indices': LAYER_INDICES,
                'prefill_equivalent_pairs': prefill_equiv_pairs,
                'decode_equivalent_pairs': decode_equiv_pairs
            },
            'comparisons': comparison_results,
            'prefill_analysis': prefill_analysis,
            'decode_analysis': decode_analysis
        }
        filepath = os.path.join(output_dir, f"verify_{timestamp}.json")
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
            
        log_print(f"\n✓ Results saved to: {filepath}")
            
    else:
        # GENERATION MODE
        results = {
            'metadata': {
                'environment': system_info,
                'batch_sizes': BATCH_SIZES,
                'layer_indices': LAYER_INDICES
            },
            'measurements': []
        }
        
        for ref_name, ref_text in REFERENCE_SEQUENCES.items():
            log_print(f"\n{'='*80}")
            log_print(f"REFERENCE: {ref_name}")
            log_print("="*80)
            
            min_prompt_length = compute_min_length_across_batches(
                ref_text, ref_name, tokenizer, BATCH_SIZES
            )
            log_print(f"\nGlobal minimum prompt length: {min_prompt_length} tokens\n")
            
            for batch_size in BATCH_SIZES:
                log_print(f"  bs={batch_size}:", end="")
                decode_data = run_decode_with_extraction(
                    model, tokenizer, ref_text, ref_name, batch_size, LAYER_INDICES,
                    forced_length=min_prompt_length
                )
                results['measurements'].append({
                    'ref_name': ref_name,
                    'batch_size': batch_size,
                    'generated_ids': decode_data['generated_ids'],
                    'prompt_token_ids': decode_data['prompt_token_ids'],
                    'prefill_signals': decode_data['prefill_signals'],
                    'decode_signals': decode_data['decode_signals'],
                    'all_batch_generated_ids': decode_data['all_batch_generated_ids']
                })
        
        # Token consistency check
        for ref_name in REFERENCE_SEQUENCES.keys():
            log_print(f"\n--- Token consistency for {ref_name} ---")
            ref_measurements = {m['batch_size']: m for m in results['measurements'] if m['ref_name'] == ref_name}
            check_token_consistency(ref_measurements, tokenizer)
                
        # Within-hardware analysis
        prefill_sanity = analyze_within_hardware(
            results['measurements'], BATCH_SIZES, LAYER_INDICES, 'prefill'
        )
        decode_sanity = analyze_within_hardware(
            results['measurements'], BATCH_SIZES, LAYER_INDICES, 'decode'
        )
        
        results['prefill_sanity_check'] = prefill_sanity
        results['decode_sanity_check'] = decode_sanity
        
        filepath = os.path.join(output_dir, f"decode_{timestamp}.json")
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
            
        log_print(f"\n✓ Saved to: {filepath}")
        log_print(f"\nNext step: Copy to verifier machine, set TEACHER_FORCING=True")

    close_logging()

if __name__ == "__main__":
    main()
