
import os
os.environ['HF_HOME'] = '/workspace/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/huggingface_cache'

from vllm import LLM, SamplingParams
import numpy as np
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

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
CACHE_DIR = '/workspace/huggingface_cache'

BATCH_SIZES = [1, 2, 3, 4, 5, 8, 9]
MAX_NEW_TOKENS = 20
TOKENS_PER_SLICE = 10000
NUM_REFERENCES = 3
TOP_K_LOGPROBS = 20  # Store top-20 to ensure overlap for comparison

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
    log_path = os.path.join(output_dir, f"vllm_experiment_generation_{timestamp}.txt")
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
        tokens = tokenizer.encode(text)
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
    import torch
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
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    try:
        import vllm
        info["vllm_version"] = vllm.__version__
    except (ImportError, AttributeError):
        info["vllm_version"] = "unknown"

    return info

def validate_environment_match(reference_env, verifier_env):
    """
    Validate that software environments match between reference and verifier.
    """
    log_print("\n" + "="*80)
    log_print("ENVIRONMENT VALIDATION")
    log_print("="*80)

    critical_fields = ['vllm_version', 'torch_version', 'cuda_version']
    expected_different = ['gpu_name', 'hostname']

    mismatches = []

    log_print("\nCritical dependencies:")
    for field in critical_fields:
        ref_val = reference_env.get(field, 'N/A')
        ver_val = verifier_env.get(field, 'N/A')

        if ref_val == ver_val:
            log_print(f"  ✓ {field}: {ref_val}")
        else:
            log_print(f"  ✗ {field}: reference={ref_val}, verifier={ver_val}")
            mismatches.append((field, ref_val, ver_val))

    log_print("\nExpected differences (hardware):")
    for field in expected_different:
        ref_val = reference_env.get(field, 'N/A')
        ver_val = verifier_env.get(field, 'N/A')

        if ref_val != ver_val:
            log_print(f"  ✓ {field}: reference={ref_val}, verifier={ver_val}")
        else:
            log_print(f"  ⚠ {field}: SAME ({ref_val}) - are you on different hardware?")

    if not mismatches:
        log_print("\n✓ ENVIRONMENT VALIDATION PASSED")
        return {'valid': True, 'mismatches': []}
    else:
        log_print("\n⚠ ENVIRONMENT MISMATCHES DETECTED")
        log_print("  Results may be affected by software differences, not just hardware.")
        return {'valid': False, 'mismatches': mismatches}

# ============================================================================
# LOGPROB EXTRACTION
# ============================================================================

def extract_logprobs_from_output(output, positions=[-3, -2, -1]):
    """
    Extract logprobs from vLLM output at specified positions.
    """
    signals = {}
    
    logprobs_list = output.outputs[0].logprobs
    
    if logprobs_list is None:
        return signals
    
    num_generated = len(logprobs_list)
    
    for pos in positions:
        actual_idx = pos if pos >= 0 else num_generated + pos
        
        if actual_idx < 0 or actual_idx >= num_generated:
            continue
        
        pos_label = f"pos_{pos}"
        token_logprobs = logprobs_list[actual_idx]
        
        token_ids = []
        log_probs = []
        
        for token_id, logprob_obj in token_logprobs.items():
            token_ids.append(token_id)
            log_probs.append(logprob_obj.logprob)
        
        signals[pos_label] = {
            'logprobs': {
                'token_ids': token_ids,
                'log_probs': log_probs
            }
        }
    
    return signals

def extract_prompt_logprobs(output, prompt_length, positions=[-3, -2, -1]):
    """
    Extract logprobs from prompt positions (for prefill analysis).
    """
    signals = {}
    
    prompt_logprobs_list = output.prompt_logprobs
    
    if prompt_logprobs_list is None:
        return signals
    
    for pos in positions:
        actual_idx = pos if pos >= 0 else prompt_length + pos
        
        if actual_idx < 0 or actual_idx >= len(prompt_logprobs_list):
            continue
        
        pos_label = f"pos_{pos}"
        token_logprobs = prompt_logprobs_list[actual_idx]
        
        if token_logprobs is None:
            continue
        
        token_ids = []
        log_probs = []
        
        for token_id, logprob_obj in token_logprobs.items():
            token_ids.append(token_id)
            log_probs.append(logprob_obj.logprob)
        
        signals[pos_label] = {
            'logprobs': {
                'token_ids': token_ids,
                'log_probs': log_probs
            }
        }
    
    return signals

# ============================================================================
# GENERATION MODE
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

        token_lengths = [len(tokenizer.encode(t)) for t in batch_texts]
        min_length = min(min_length, min(token_lengths))

    return min_length

def run_generation(llm, tokenizer, ref_text, ref_name, batch_size, forced_length=None):
    """
    Run generation with specified batch size and extract signals.
    """
    ref_dummies = DUMMY_SETS[ref_name]
    
    if batch_size == 1:
        batch_texts = [ref_text]
    else:
        batch_texts = [ref_text] + ref_dummies[:batch_size-1]
    
    all_token_ids = [tokenizer.encode(t) for t in batch_texts]
    
    if forced_length is not None:
        min_length = forced_length
    else:
        min_length = min(len(ids) for ids in all_token_ids)
    
    truncated_token_ids = [ids[:min_length] for ids in all_token_ids]
    truncated_texts = [tokenizer.decode(ids) for ids in truncated_token_ids]
    
    prompt_length = len(truncated_token_ids[0])
    log_print(f"      Prompt: {prompt_length} tokens", end="")
    
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        logprobs=TOP_K_LOGPROBS,
        prompt_logprobs=TOP_K_LOGPROBS,
    )
    
    outputs = llm.generate(truncated_texts, sampling_params)
    
    output_0 = outputs[0]
    generated_ids = list(output_0.outputs[0].token_ids)
    num_generated = len(generated_ids)
    
    prefill_signals = extract_prompt_logprobs(output_0, prompt_length, positions=[-3, -2, -1])
    decode_signals = extract_logprobs_from_output(output_0, positions=[-3, -2, -1])
    
    all_batch_generated_ids = [list(out.outputs[0].token_ids) for out in outputs]
    
    log_print(f" → Final: {prompt_length + num_generated} tokens ({num_generated} generated)")
    
    return {
        'generated_ids': generated_ids,
        'all_batch_generated_ids': all_batch_generated_ids,
        'prompt_token_ids': truncated_token_ids,
        'prompt_length': prompt_length,
        'prefill_signals': prefill_signals,
        'decode_signals': decode_signals,
        'num_generated': num_generated
    }


# ============================================================================
# ANALYSIS
# ============================================================================

def compute_logprob_distance_canonical(logprobs1, logprobs2, canonical_ids):
    """
    Compute L2 distance between logprobs for a canonical set of token IDs.
    """
    lp1 = dict(zip(logprobs1['token_ids'], logprobs1['log_probs']))
    lp2 = dict(zip(logprobs2['token_ids'], logprobs2['log_probs']))

    vec1 = []
    vec2 = []

    for tid in canonical_ids:
        if tid in lp1 and tid in lp2:
            vec1.append(lp1[tid])
            vec2.append(lp2[tid])

    if len(vec1) == 0:
        return float('inf')

    return float(np.linalg.norm(np.array(vec1) - np.array(vec2)))

def compare_signals(signals1, signals2):
    """Compare two signal sets using top 5 token IDs from first signal as canonical."""
    common_positions = set(signals1.keys()) & set(signals2.keys())

    all_dists = []

    for pos_label in common_positions:
        sig1 = signals1[pos_label]
        sig2 = signals2[pos_label]

        # Use top 5 for comparison (stored top 20 as buffer)
        canonical_ids = sig1['logprobs']['token_ids'][:5]
        dist = compute_logprob_distance_canonical(
            sig1['logprobs'], sig2['logprobs'], canonical_ids
        )
        all_dists.append(dist)

    return {
        'logprobs_max': max(all_dists) if all_dists else 0.0,
        'logprobs_mean': np.mean(all_dists) if all_dists else 0.0
    }

def compare_signals_with_canonical_ids(signals1, signals2, canonical_token_ids):
    """Compare using pre-specified canonical token IDs (top 5)."""
    common_positions = set(signals1.keys()) & set(signals2.keys())

    all_dists = []

    for pos_label in common_positions:
        sig1 = signals1[pos_label]
        sig2 = signals2[pos_label]

        # Use top 5 for comparison
        canonical_ids = canonical_token_ids.get(pos_label, sig1['logprobs']['token_ids'][:5])[:5]
        dist = compute_logprob_distance_canonical(
            sig1['logprobs'], sig2['logprobs'], canonical_ids
        )
        all_dists.append(dist)

    return {
        'logprobs_max': max(all_dists) if all_dists else 0.0,
        'logprobs_mean': np.mean(all_dists) if all_dists else 0.0
    }

def check_token_consistency(measurements, tokenizer):
    """Verify element 0 generates identical tokens across all batch sizes."""
    log_print("\n" + "="*80)
    log_print("TOKEN GENERATION CONSISTENCY CHECK")
    log_print("="*80)

    tokens_by_bs = {}
    for bs, data in measurements.items():
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
    Returns list of sets, each set containing batch sizes using the same kernel.
    """
    # Union-find style grouping
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
    
    # Group by root
    groups = {}
    for bs in batch_sizes:
        root = find(bs)
        if root not in groups:
            groups[root] = set()
        groups[root].add(bs)
    
    return list(groups.values())

def analyze_within_hardware(measurements, batch_sizes, signal_source='decode'):
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

    all_matrices = []
    n_bs = len(batch_sizes)

    for ref_name in sorted(by_ref.keys()):
        log_print(f"\n{ref_name}:")

        ref_data = by_ref[ref_name]
        matrix = np.zeros((n_bs, n_bs))

        # Use first batch size as canonical
        canonical_bs = batch_sizes[0]
        canonical_signals = ref_data.get(canonical_bs)

        canonical_token_ids = {}
        if canonical_signals:
            for pos_label, pos_data in canonical_signals.items():
                canonical_token_ids[pos_label] = pos_data['logprobs']['token_ids']

        for i, bs1 in enumerate(batch_sizes):
            for j, bs2 in enumerate(batch_sizes):
                if bs1 not in ref_data or bs2 not in ref_data:
                    continue

                if i == j:
                    matrix[i, j] = 0.0
                else:
                    distances = compare_signals_with_canonical_ids(
                        ref_data[bs1], ref_data[bs2], canonical_token_ids
                    )
                    matrix[i, j] = distances['logprobs_mean']

        # Display matrix
        header = "       " + "".join([f"bs={bs:>3} " for bs in batch_sizes])
        log_print(header)
        for i, bs in enumerate(batch_sizes):
            row_str = f"bs={bs:<3}"
            for j in range(n_bs):
                row_str += f"  {matrix[i,j]:6.2e}"
            log_print(row_str)

        all_matrices.append(matrix)

    # Aggregate
    avg_matrix = np.mean(all_matrices, axis=0)

    log_print("\n" + "="*80)
    log_print("AGGREGATE (average across references):")
    log_print("="*80)

    header = "       " + "".join([f"bs={bs:>3} " for bs in batch_sizes])
    log_print(header)
    for i, bs in enumerate(batch_sizes):
        row_str = f"bs={bs:<3}"
        for j in range(n_bs):
            row_str += f"  {avg_matrix[i,j]:6.2e}"
        log_print(row_str)

    # Statistics
    off_diag = avg_matrix[np.triu_indices(n_bs, k=1)]
    
    log_print(f"\nOff-diagonal stats:")
    log_print(f"  Mean: {np.mean(off_diag):.2e}")
    log_print(f"  Range: [{np.min(off_diag):.2e}, {np.max(off_diag):.2e}]")
    
    # Find equivalent pairs
    equivalent_pairs = find_equivalent_pairs(avg_matrix, batch_sizes)
    kernel_classes = format_kernel_classes(equivalent_pairs, batch_sizes)
    
    # Check if all zeros
    zero_count = np.sum(off_diag < EQUIVALENCE_THRESHOLD)
    total_count = len(off_diag)
    
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
    
    return {
        'matrix': avg_matrix.tolist(),
        'per_reference_matrices': [m.tolist() for m in all_matrices],
        'off_diagonal_mean': float(np.mean(off_diag)),
        'off_diagonal_range': [float(np.min(off_diag)), float(np.max(off_diag))],
        'equivalent_pairs': equivalent_pairs,
        'kernel_classes': [sorted(list(cls)) for cls in kernel_classes]
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    global REFERENCE_SEQUENCES, DUMMY_SETS

    log_path = setup_logging()
    system_info = collect_system_info()

    mode = "GENERATION (decode)"
    log_print("="*80)
    log_print(f"vLLM CROSS-HARDWARE BATCH SIZE DETECTABILITY - {mode}")
    log_print("="*80)

    log_print(f"\nSystem: {system_info['hostname']}")
    log_print(f"GPU: {system_info['gpu_name']}")
    log_print(f"vLLM: {system_info['vllm_version']}")
    log_print(f"PyTorch: {system_info['torch_version']}")
    log_print(f"CUDA: {system_info['cuda_version']}")

    log_print(f"\nConfiguration:")
    log_print(f"  Model: {MODEL_NAME}")
    log_print(f"  Batch sizes: {BATCH_SIZES}")
    log_print(f"  Max tokens: {MAX_NEW_TOKENS}")
    log_print(f"  Top-k logprobs: {TOP_K_LOGPROBS}")
    log_print()

    # Initialize vLLM
    log_print("Loading vLLM model...")
    llm = LLM(
        model=MODEL_NAME,
        download_dir=CACHE_DIR,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.7,
    )
    tokenizer = llm.get_tokenizer()
    log_print("✓ Model loaded\n")

    # Initialize sequences from PDF
    REFERENCE_SEQUENCES, DUMMY_SETS = create_sequences_from_pdf(tokenizer)
    log_print(f"Created {len(REFERENCE_SEQUENCES)} reference sequences\n")

    output_dir = '/workspace/experiments'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    # ================================================================
    # GENERATION MODE
    # ================================================================
    results = {
        'metadata': {
            'environment': system_info,
            'model': MODEL_NAME,
            'batch_sizes': BATCH_SIZES,
            'max_new_tokens': MAX_NEW_TOKENS,
            'top_k_logprobs': TOP_K_LOGPROBS,
            'timestamp': timestamp
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
            log_print(f"  bs={batch_size}:", end=" ")

            gen_data = run_generation(
                llm, tokenizer, ref_text, ref_name, batch_size,
                forced_length=min_prompt_length
            )

            results['measurements'].append({
                'ref_name': ref_name,
                'batch_size': batch_size,
                'generated_ids': gen_data['generated_ids'],
                'all_batch_generated_ids': gen_data['all_batch_generated_ids'],
                'prompt_token_ids': gen_data['prompt_token_ids'],
                'prompt_length': gen_data['prompt_length'],
                'prefill_signals': gen_data['prefill_signals'],
                'decode_signals': gen_data['decode_signals'],
                'num_generated': gen_data['num_generated']
            })

    filepath = os.path.join(output_dir, f"vllm_decode_{timestamp}.json")
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    log_print(f"\n✓ Generation results saved to: {filepath}")

    # Token consistency check
    for ref_name in REFERENCE_SEQUENCES.keys():
        log_print(f"\n--- Token consistency for {ref_name} ---")
        ref_measurements = {m['batch_size']: m for m in results['measurements'] if m['ref_name'] == ref_name}
        check_token_consistency(ref_measurements, tokenizer)

    # Within-hardware analysis
    prefill_sanity = analyze_within_hardware(
        results['measurements'], BATCH_SIZES, signal_source='prefill'
    )
    decode_sanity = analyze_within_hardware(
        results['measurements'], BATCH_SIZES, signal_source='decode'
    )

    results['prefill_sanity_check'] = prefill_sanity
    results['decode_sanity_check'] = decode_sanity

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    log_print(f"\nNext step: Copy {filepath} to verifier machine")
    log_print(f"Then set TEACHER_FORCING = True and REFERENCE_FILE = '<path>'")

    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    log_print(f"File size: {file_size_mb:.1f} MB")

    log_print(f"\n{'='*80}")
    log_print("EXPERIMENT COMPLETE")
    log_print("="*80 + "\n")

    close_logging()

if __name__ == "__main__":
    main()
