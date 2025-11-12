Same model, all INT4, just different quant methods.

Example code:



```python

#!/usr/bin/env python3
"""
AWQ Non-Determinism Root Cause Investigation
Tests vLLM INT4-AWQ quantization reproducibility with kernel profiling

Model: Qwen3-8B (AWQ4)
Focus: Identify whether Marlin kernel has non-deterministic behavior
"""
'''
# ============================================================================
# SUPPRESS VERBOSE LOGGING
# ============================================================================
import os
os.environ['HF_HOME'] = '/tmp/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf_cache'
os.environ['VLLM_LOGGING_LEVEL'] = 'WARNING'
os.environ['VLLM_CONFIGURE_LOGGING'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('vllm').setLevel(logging.ERROR)
logging.getLogger('vllm.engine').setLevel(logging.ERROR)
logging.getLogger('vllm.worker').setLevel(logging.ERROR)
logging.getLogger('vllm.executor').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.INFO)
logging.getLogger('huggingface_hub.file_download').setLevel(logging.INFO)
'''
# ============================================================================
# IMPORTS
# ============================================================================

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import numpy as np
from datetime import datetime
import json
import torch
import gc
import sys

# Try to import new profiler API, fall back to old if needed
try:
    from torch.profiler import profile, ProfilerActivity
    PROFILER_NEW_API = True
except ImportError:
    import torch.autograd.profiler as profiler_module
    PROFILER_NEW_API = False

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "Qwen/Qwen3-8B-AWQ"
QUANTIZATION = "awq"
TENSOR_PARALLEL_SIZE = 1
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.9

# Generation configuration
MAX_TOKENS = 50
NUM_REPETITIONS = 5
TEMPERATURE = 0.0
SEED = 42
TOP_LOGPROBS = 10

# Test prompt
TEST_PROMPT = """The Evolution of Large Language Models: Technical Foundations and Societal Implications

Introduction

The field of artificial intelligence has witnessed a remarkable transformation over the past decade, driven primarily by advances in deep learning and the emergence of increasingly sophisticated language models. These models, trained on vast corpora of text data, have demonstrated remarkable capabilities across a wide range of tasks, from translation and summarization to question answering and creative writing. However, their deployment raises significant challenges related to computational efficiency, interpretability, and safety.

This document explores the technical foundations of modern large language models, their architectural innovations, the computational infrastructure required for their deployment, and the broader implications for AI governance and safety. We examine how these systems process information, the sources of variation in their outputs, and the methods being developed to ensure their reliable and safe operation at scale.

Part I: Architectural Foundations

1.1 The Transformer Architecture

At the heart of modern language models lies the transformer architecture, introduced in 2017 by Vaswani et al. The transformer's key innovation was the self-attention mechanism, which allows models to dynamically weigh the importance of different parts of their input when processing each element. Unlike recurrent neural networks, transformers can process sequences in parallel, enabling efficient training on modern hardware.

The self-attention mechanism computes attention scores between all pairs of positions in a sequence. For a sequence of length n, this creates an n×n attention matrix, where each entry represents how much focus position i should place on position j when computing its representation. This mechanism enables the model to capture long-range dependencies without the gradient flow problems that plagued earlier sequential architectures.

The transformer architecture consists of multiple layers, each containing two main components: a multi-head attention mechanism and a position-wise feed-forward network. The multi-head attention allows the model to attend to different aspects of the input simultaneously, with each "head" learning to focus on different patterns. The feed-forward networks apply learned transformations independently to each position, introducing non-linearity and increasing the model's representational capacity.

1.2 Scaling Laws and Emergent Capabilities

Research has revealed consistent scaling laws that govern language model performance. These laws demonstrate that model capability improves predictably with three key factors: the number of parameters, the size of the training dataset, and the amount of computational resources invested in training. This predictability has enabled researchers to forecast the capabilities of future models and make informed decisions about resource allocation.

As models scale beyond certain thresholds, they begin to exhibit emergent capabilities—abilities that were not explicitly trained but arise from the combination of scale and diverse training data. These capabilities include few-shot learning, where models can adapt to new tasks with minimal examples, and chain-of-thought reasoning, where models break down complex problems into intermediate steps.

The relationship between model size and capability is not always smooth. Some abilities appear suddenly at particular scales, suggesting that certain computational thresholds must be crossed before specific capabilities emerge. This phenomenon has important implications for AI safety, as it means that scaling up models may lead to unexpected new capabilities that require careful evaluation and monitoring.

1.3 Attention Mechanisms and Their Variants

While the standard multi-head attention mechanism has proven highly effective, researchers have developed numerous variants to address specific challenges. Grouped query attention (GQA) reduces the computational cost by sharing key and value projections across multiple query heads, maintaining most of the expressiveness while significantly reducing memory requirements.

Multi-latent attention (MLA) represents another innovation, particularly valuable for models deployed in memory-constrained environments. MLA compresses the key-value cache through learned projections, achieving dramatic reductions in memory usage—often 90-95% compression—while maintaining model quality. This compression is especially important for inference scenarios with long contexts, where the KV cache would otherwise dominate memory consumption.

FlashAttention and its successors have revolutionized attention computation by reorganizing the order of operations to maximize GPU utilization. By computing attention in blocks and carefully managing data movement between GPU memory hierarchies, FlashAttention achieves significant speedups without changing the mathematical operations being performed. This algorithmic innovation demonstrates that substantial performance improvements can come from careful consideration of hardware characteristics rather than changes to the underlying model.

Part II: Computational Infrastructure

2.1 GPU Architecture and Floating-Point Computation

Modern GPUs are highly specialized processors designed to perform massive numbers of parallel floating-point operations. NVIDIA's Hopper and Blackwell architectures, for instance, contain thousands of CUDA cores and specialized tensor cores optimized for matrix multiplication—the fundamental operation in neural network inference and training.

Floating-point arithmetic, however, is not exact. IEEE 754 floating-point numbers can only represent a finite subset of real numbers, leading to rounding errors in computation. Moreover, floating-point arithmetic is non-associative: (a + b) + c may yield a different result than a + (b + c) due to rounding at each step. This property has profound implications for reproducibility in distributed computing environments.

Different GPU architectures implement floating-point operations with varying degrees of precision and through different execution paths. Even when using the same numerical precision (e.g., bfloat16 or float32), different GPU models may produce slightly different results due to differences in their microarchitecture, instruction scheduling, or specialized hardware accelerators. These hardware-level variations become important when considering verification and reproducibility in production deployments.

2.2 Tensor Parallelism and Distributed Inference

Large language models often exceed the memory capacity of a single GPU, necessitating distribution across multiple devices. Tensor parallelism splits individual weight matrices across GPUs, requiring careful coordination of matrix multiplications and communication between devices. When a layer is distributed across n GPUs, each GPU computes a portion of the output, which must then be combined through collective communication operations.

The specific parallelization strategy affects not just performance but also numerical behavior. Different decompositions of the same computation lead to different orders of floating-point operations, potentially resulting in different final values even when starting from identical weights and inputs. This sensitivity to parallelization strategy has important implications for model verification and monitoring.

Pipeline parallelism represents an alternative approach, where different layers of the model reside on different GPUs. Forward passes proceed in a pipelined fashion, with activations flowing from one GPU to the next. While pipeline parallelism typically introduces less numerical variation than tensor parallelism (since each layer's computation remains intact), it requires careful management of micro-batching to maintain efficiency.

2.3 Memory Hierarchies and Caching

Modern GPUs have complex memory hierarchies, including registers, L1 cache, L2 cache, shared memory, and global memory (VRAM). Efficient inference requires careful orchestration of data movement through these levels, as memory bandwidth often becomes the primary bottleneck rather than computational throughput.

The KV (key-value) cache exemplifies memory management challenges in language model inference. During autoregressive generation, previously computed key and value vectors must be retained and accessed at each step. For long contexts, this cache can grow to dominate memory usage. Innovations like paged attention manage the KV cache more efficiently by storing it in non-contiguous memory blocks, similar to how operating systems manage virtual memory.

Part III: Quantization and Efficiency

3.1 Quantization Techniques

Quantization reduces the precision of model weights and activations, trading some accuracy for significant improvements in memory usage and computational efficiency. Early post-training quantization methods simply converted trained models to lower precision, but more sophisticated approaches now integrate quantization awareness into the training process.

INT8 quantization represents weights and activations as 8-bit integers rather than 32-bit floating-point numbers, achieving 4× memory reduction and enabling the use of specialized integer arithmetic units on modern processors. More aggressive quantization schemes, including INT4 and even binary networks, push these boundaries further, though with increasing risk to model quality.

Weight-only quantization, where activations remain in higher precision while weights are quantized, often provides an attractive trade-off. This approach maintains most of the model's accuracy while still achieving significant memory savings and bandwidth improvements. The asymmetry reflects the fact that weight values are known at deployment time and can be carefully calibrated, while activations vary with each input.

3.2 Quantization-Aware Training

Quantization-aware training (QAT) incorporates quantization operations into the training process itself, allowing the model to learn weight distributions that are more amenable to low-precision representation. During QAT, the forward pass simulates quantization effects using fake quantization operations, while the backward pass still uses full-precision gradients.

The effectiveness of QAT depends critically on the quantization scheme used. Symmetric quantization maps both positive and negative values uniformly, while asymmetric quantization can adapt to weight distributions that don't center on zero. Per-tensor quantization uses a single scale factor for an entire tensor, while per-channel or per-group quantization allows finer-grained adaptation to local statistics.

Recent advances in microscaling (MXFP) quantization formats, such as MXFP4 and MXFP6, provide a middle ground between integer and floating-point quantization. These formats maintain a small floating-point exponent while using very few mantissa bits, preserving the dynamic range of floating-point numbers while approaching the memory efficiency of integer quantization.

Part IV: Inference Optimization and Kernels

4.1 CUDA Kernels and Operator Fusion

Inference efficiency depends critically on the implementation of individual operations (kernels) that execute on the GPU. Standard deep learning frameworks provide default kernel implementations, but specialized hand-written kernels can often achieve substantial speedups by exploiting specific hardware features or operation patterns.

Operator fusion combines multiple sequential operations into a single kernel, reducing memory traffic by keeping intermediate results in fast memory rather than writing them back to global memory. For instance, fusing a matrix multiplication with its subsequent activation function eliminates the need to store the pre-activation values in VRAM. Modern frameworks use sophisticated graph optimization passes to identify and execute such fusion opportunities automatically.

The specific kernel implementation chosen for an operation can affect not just performance but also numerical results. Different kernel implementations may use different algorithms or accumulation orders, leading to different rounding errors. This variation becomes particularly pronounced when comparing implementations across different framework versions or hardware generations.

4.2 Compilation and Just-In-Time Optimization

Modern deep learning frameworks increasingly employ just-in-time compilation to generate optimized code for specific model configurations and hardware targets. PyTorch's torch.compile, for instance, captures the computational graph and applies a variety of optimizations before generating machine code.

These compiler optimizations include operation reordering, dead code elimination, constant folding, and memory layout transformations. While these transformations preserve mathematical correctness in exact arithmetic, they can alter the order and grouping of floating-point operations, potentially affecting numerical outputs. The difference is typically small—often on the order of 10^-6 in relative terms—but may be detectable when comparing against uncompiled baselines.

Graph compilation can also unlock new optimization opportunities that aren't available in eager execution mode. For instance, memory planning algorithms can allocate activation buffers more efficiently when they have visibility into the entire computational graph, reducing peak memory usage. Similarly, automatic kernel selection can make globally optimal choices rather than greedy local decisions.

Part V: Production Deployment and Batching

5.1 Continuous Batching and Request Scheduling

Production inference services must handle streams of incoming requests efficiently. Continuous batching, also called iteration-level batching, allows new requests to join a running batch at each generation step. This approach maximizes GPU utilization by ensuring the batch remains full even as individual requests complete at different times.

The specific composition of a batch—which particular inputs are grouped together—generally doesn't affect individual outputs in modern attention mechanisms, thanks to the attention mask that prevents information leakage between batch elements. However, the batch size itself does matter: larger batches often employ different computational strategies that may introduce small numerical differences.

Memory management for continuous batching requires sophisticated orchestration. The paged attention mechanism treats the KV cache as a virtual memory space, allocating physical memory blocks dynamically as requests arrive and deallocating them as requests complete. This flexibility enables much higher throughput than static batching approaches, though it introduces additional complexity in memory access patterns.

5.2 Scheduling and Resource Allocation

Inference servers must make real-time decisions about which requests to batch together and how to allocate limited GPU resources. These decisions involve trade-offs between latency (how long individual requests wait) and throughput (how many requests can be served per second). Different scheduling policies lead to different performance characteristics.

CUDA streams provide a mechanism for overlapping computation and communication operations. Multiple streams can execute concurrently on the same GPU, potentially improving utilization by allowing I/O operations and computation to proceed in parallel. However, when multiple streams compete for the same compute resources, their interactions can introduce timing variations and potential numerical artifacts from scheduling conflicts.

Part VI: Verification and Monitoring

6.1 The Challenge of Comprehensive Reporting

In scenarios where datacenter operators must report all ML computations performed on their hardware—such as might arise under international AI governance agreements—verification becomes critical. Even if operators provide detailed logs of their inference workloads, how can external auditors verify that these logs are comprehensive?

A malicious operator might attempt to hide unauthorized computations by secretly improving the efficiency of their declared workloads, freeing up capacity for off-the-books inference. Traditional monitoring approaches might detect obvious violations, such as periods of idle GPU time being used for hidden work, but wouldn't catch more sophisticated evasion strategies.

6.2 Floating-Point Forensics

One promising approach to verification leverages the numerical sensitivity we've discussed throughout this document. If an operator claims to have produced certain outputs using a specific computational setup, a verifier can attempt to reproduce those outputs on a trusted verification cluster. Exact numerical matching would provide strong evidence that the claimed setup was indeed used.

This forensic approach relies on two key properties we've established: first, that identical setups produce bit-exact or near-identical results (low statistical noise), and second, that different setups produce systematically different outputs (detectable systematic deviations). The combination of these properties creates a unique numerical "fingerprint" for each inference configuration.

Different aspects of the computational setup leave distinct signatures. Batch size changes typically produce L2 deviations on the order of 0.3-0.6, while changes in attention implementation (e.g., eager attention vs FlashAttention) create much larger deviations, often exceeding 40. CUDA version changes can lead to dramatic differences, with errors growing by several orders of magnitude between major versions. Even compilation optimizations like torch.compile introduce detectable deviations of around 2-5 in L2 norm.

6.3 Attack Vectors and Detection Limits

An adversary attempting to perform hidden inference in parallel with declared workloads would likely affect the computational environment in detectable ways. Running additional work on separate CUDA streams, for instance, creates resource contention and scheduling conflicts. At high GPU utilization—the regime most relevant for production systems—these conflicts introduce measurable timing variations and numerical artifacts.

The economics of different evasion strategies also matter. Small-scale hidden workloads that might evade detection provide minimal benefit to an attacker, as they can't significantly offset the cost of running a datacenter. Larger hidden workloads that would provide meaningful economic value are more likely to create detectable signatures, either through timing anomalies, numerical deviations, or both.

However, certain attack vectors remain challenging to detect. For instance, if an adversary can successfully hide parallel work that affects only low-arithmetic-intensity operations (like attention operations over long sequences), they might avoid creating floating-point deviations even while causing measurable slowdowns. The detectability of such scenarios requires careful analysis of both numerical and timing forensics.

Part VII: Future Directions

7.1 Hardware Evolution

Next-generation AI accelerators continue to push the boundaries of performance and efficiency. Blackwell's B200 GPUs, for instance, include native support for FP4 and FP6 microscaling formats, enabling even more aggressive quantization while maintaining model quality. As hardware capabilities evolve, the landscape of numerical behavior and reproducibility characteristics will continue to shift.

Specialized AI chips from various vendors introduce additional diversity in computational behavior. Each architecture makes different trade-offs in its implementation of floating-point operations, memory hierarchies, and specialized accelerators. This diversity enriches the space of possible forensic signatures while also complicating verification protocols that must account for legitimate hardware variations.

7.2 Software Stack Evolution

Deep learning frameworks continue to evolve rapidly, with new optimizations and features appearing in each release. This evolution creates challenges for reproducibility: results from one framework version may differ from another, even when using identical models and inputs. From a forensics perspective, framework version becomes another factor that must be either controlled or characterized.

The trend toward greater automation in optimization—such as automatic kernel selection, dynamic batching strategies, and just-in-time compilation—generally improves performance but can make numerical behavior less predictable. Balancing the benefits of these optimizations against the need for reproducibility and verifiability represents an ongoing challenge for production ML systems.

7.3 Governance and Policy Implications

The techniques discussed in this document have implications beyond technical verification. If floating-point forensics proves reliable for detecting various forms of computational evasion, it could inform the design of AI governance mechanisms and international agreements. The ability to verify claimed computations without requiring complete transparency into internal operations might enable monitoring frameworks that balance accountability with legitimate concerns about intellectual property and competitive advantage.

However, realizing this potential requires continued research into the limits and capabilities of forensic approaches. What types of computational changes are reliably detectable? What attack vectors remain? How do detection capabilities scale to different model architectures, sizes, and deployment scenarios? Answering these questions will require systematic experimentation across a wide range of conditions.

Conclusion

The numerical behavior of large language models reflects a complex interplay between model architecture, hardware characteristics, software implementation, and deployment configurations. While this complexity initially appears to pose challenges for reproducibility, it also creates opportunities: the unique numerical fingerprint of each setup can serve as a basis for verification and monitoring in governance contexts.

The path forward requires continued research at the intersection of systems optimization, numerical computing, and AI safety. As models grow larger and deployment scenarios more diverse, maintaining the ability to verify and monitor AI systems becomes increasingly important. The forensic approaches discussed here represent one promising direction, but their ultimate viability depends on rigorous empirical validation across realistic production conditions.

This document has surveyed the technical foundations necessary to understand these verification challenges and potential solutions. The field continues to evolve rapidly, and many questions remain open. Nevertheless, the convergence of growing AI capabilities, increasing deployment scale, and emerging governance frameworks makes this research area both timely and critical for the safe development of advanced AI systems."""

# ============================================================================
# EXPERIMENT SETUP
# ============================================================================

print("=" * 80)
print("AWQ NON-DETERMINISM ROOT CAUSE INVESTIGATION")
print("=" * 80)
print()
print(f"Model: {MODEL_NAME}")
print(f"Quantization: {QUANTIZATION} (INT4)")
print(f"Prompt length: {len(TEST_PROMPT)} chars")
print(f"Repetitions per mode: {NUM_REPETITIONS}")
print()
print("Investigation plan:")
print("  1. Baseline: Standard vLLM generation (non-deterministic expected)")
print("  2. With determinism: torch.use_deterministic_algorithms(True)")
print("  3. Kernel profiling: Identify which kernels are called")
print("  4. Analysis: Compare noise levels and kernel usage")
print()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def set_deterministic_mode(enable=True):
    """Enable/disable PyTorch deterministic algorithms"""
    if enable:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("✓ Deterministic mode ENABLED")
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        print("✓ Deterministic mode DISABLED (standard vLLM)")
    print()

def clear_gpu():
    """Aggressively clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def run_repetitions(llm, prompt_text, sampling_params, num_reps, mode_name):
    """Run multiple generations and collect results"""
    print(f"Running {num_reps} repetitions in {mode_name} mode...")
    
    results_tokens = []
    results_logprobs = []
    results_distributions = []
    
    for rep in range(num_reps):
        clear_gpu()
        
        outputs = llm.generate(prompt_text, sampling_params=sampling_params)
        output = outputs[0]
        
        # Extract data
        token_ids = output.outputs[0].token_ids
        results_tokens.append(token_ids)
        
        # Logprobs
        logprobs_data = output.outputs[0].logprobs
        selected_logprobs = [lp[token_ids[i]].logprob for i, lp in enumerate(logprobs_data)]
        results_logprobs.append(np.array(selected_logprobs))
        
        # Distributions
        rep_distributions = []
        for position_logprobs in logprobs_data:
            sorted_items = sorted(position_logprobs.items(), 
                                key=lambda x: x[1].logprob, 
                                reverse=True)[:TOP_LOGPROBS]
            rep_distributions.append([(tok, lp.logprob) for tok, lp in sorted_items])
        results_distributions.append(rep_distributions)
        
        if (rep + 1) % 5 == 0:
            print(f"  Completed {rep + 1}/{num_reps} repetitions")
    
    print(f"✓ {mode_name} mode complete: {num_reps} repetitions")
    print()
    
    return results_tokens, results_logprobs, results_distributions

def analyze_reproducibility(results_tokens, results_logprobs, mode_name):
    """Analyze reproducibility statistics"""
    print(f"Analysis: {mode_name}")
    print("-" * 60)
    
    # Check token sequences
    tokens_identical = all(
        results_tokens[0] == results_tokens[i] 
        for i in range(1, len(results_tokens))
    )
    print(f"Token sequences identical: {tokens_identical}")
    
    # Check logprobs
    first_logprobs = results_logprobs[0]
    logprobs_exact = all(
        np.allclose(first_logprobs, results_logprobs[i], rtol=0, atol=1e-10)
        for i in range(1, len(results_logprobs))
    )
    print(f"Logprobs bit-exact: {logprobs_exact}")
    
    if not logprobs_exact:
        # Compute L2 distances
        l2_distances = []
        for i in range(1, len(results_logprobs)):
            l2 = np.linalg.norm(first_logprobs - results_logprobs[i])
            l2_distances.append(l2)
        
        print(f"\nLogprob deviations:")
        print(f"  Mean L2: {np.mean(l2_distances):.6e}")
        print(f"  Std L2:  {np.std(l2_distances):.6e}")
        print(f"  Min L2:  {np.min(l2_distances):.6e}")
        print(f"  Max L2:  {np.max(l2_distances):.6e}")
        
        # Per-token statistics
        all_logprobs = np.array(results_logprobs)
        std_per_token = all_logprobs.std(axis=0)
        print(f"\nPer-token std:")
        print(f"  Mean: {std_per_token.mean():.6e}")
        print(f"  Max:  {std_per_token.max():.6e}")
        print(f"  Min:  {std_per_token.min():.6e}")
        
        return {
            'tokens_identical': tokens_identical,
            'logprobs_exact': logprobs_exact,
            'l2_mean': float(np.mean(l2_distances)),
            'l2_std': float(np.std(l2_distances)),
            'l2_max': float(np.max(l2_distances)),
            'std_per_token_mean': float(std_per_token.mean()),
            'std_per_token_max': float(std_per_token.max())
        }
    else:
        return {
            'tokens_identical': tokens_identical,
            'logprobs_exact': logprobs_exact,
            'l2_mean': 0.0,
            'l2_std': 0.0,
            'l2_max': 0.0,
            'std_per_token_mean': 0.0,
            'std_per_token_max': 0.0
        }

# ============================================================================
# PREPARE PROMPT
# ============================================================================

print("Preparing prompt...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir='/tmp/hf_cache',
    trust_remote_code=True
)

messages = [{"role": "user", "content": TEST_PROMPT}]
prompt_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

prompt_tokens = tokenizer.encode(prompt_text)
prompt_length = len(prompt_tokens)

print(f"✓ Prompt prepared: {prompt_length} tokens")
print()

sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    seed=SEED,
    logprobs=TOP_LOGPROBS,
    skip_special_tokens=False
)

# ============================================================================
# EXPERIMENT 1: BASELINE (NON-DETERMINISTIC MODE)
# ============================================================================

print("=" * 80)
print("EXPERIMENT 1: BASELINE (Standard vLLM)")
print("=" * 80)
print()

set_deterministic_mode(False)

print("Loading model...")
llm = LLM(
    model=MODEL_NAME,
    quantization=QUANTIZATION,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    trust_remote_code=True,
    seed=SEED,
    enforce_eager=True,
    enable_prefix_caching=False
)
print("✓ Model loaded")
print()

# Warmup
print("Warmup...")
for _ in range(3):
    _ = llm.generate(prompt_text, sampling_params=sampling_params)
clear_gpu()
print("✓ Warmup complete")
print()

# Run baseline
baseline_tokens, baseline_logprobs, baseline_dists = run_repetitions(
    llm, prompt_text, sampling_params, NUM_REPETITIONS, "BASELINE"
)

baseline_stats = analyze_reproducibility(baseline_tokens, baseline_logprobs, "BASELINE")
print()

# ============================================================================
# KERNEL PROFILING
# ============================================================================

print("=" * 80)
print("KERNEL PROFILING")
print("=" * 80)
print()
print("Profiling kernel calls during generation...")

clear_gpu()

try:
    if PROFILER_NEW_API:
        # New PyTorch profiler API (1.8+)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=False,
            profile_memory=False,
            record_shapes=False
        ) as prof:
            _ = llm.generate(prompt_text, sampling_params=sampling_params)
    else:
        # Old PyTorch profiler API
        with profiler_module.profile(
            use_cuda=True,
            with_stack=False,
            profile_memory=False,
            record_shapes=False
        ) as prof:
            _ = llm.generate(prompt_text, sampling_params=sampling_params)

    print()
    print("Top 20 kernels by CUDA time:")
    print("-" * 80)
    kernel_table = prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20
    )
    print(kernel_table)
    print()

    # Extract kernel names for analysis
    kernel_data = []
    for evt in prof.key_averages():
        # Handle both API versions
        is_cuda = False
        if PROFILER_NEW_API:
            is_cuda = hasattr(evt, 'device_type') and evt.device_type == ProfilerActivity.CUDA
        else:
            is_cuda = evt.cuda_time_total > 0
        
        if is_cuda and evt.cuda_time_total > 0:
            kernel_data.append({
                'name': evt.key,
                'cuda_time_us': evt.cuda_time_total,
                'count': evt.count,
                'avg_time_us': evt.cuda_time_total / evt.count if evt.count > 0 else 0
            })
    
    profiling_succeeded = True

except Exception as e:
    print(f"⚠ Kernel profiling failed: {e}")
    print("Continuing without profiling data...")
    print()
    kernel_data = []
    marlin_kernels = []
    gemm_kernels = []
    dequant_kernels = []
    profiling_succeeded = False

# Look for Marlin-specific kernels
if profiling_succeeded:
    marlin_kernels = [k for k in kernel_data if 'marlin' in k['name'].lower()]
    gemm_kernels = [k for k in kernel_data if 'gemm' in k['name'].lower()]
    dequant_kernels = [k for k in kernel_data if 'dequant' in k['name'].lower() or 'quant' in k['name'].lower()]

    print("Kernel categories detected:")
    print(f"  Marlin kernels: {len(marlin_kernels)}")
    print(f"  GEMM kernels: {len(gemm_kernels)}")
    print(f"  Dequant kernels: {len(dequant_kernels)}")
    print()

    if marlin_kernels:
        print("Marlin-specific kernels:")
        for k in marlin_kernels[:5]:
            print(f"  {k['name'][:60]} - {k['cuda_time_us']/1000:.2f}ms")
        print()
else:
    marlin_kernels = []
    gemm_kernels = []
    dequant_kernels = []

# ============================================================================
# EXPERIMENT 2: WITH DETERMINISTIC MODE
# ============================================================================

print("=" * 80)
print("EXPERIMENT 2: WITH DETERMINISTIC MODE")
print("=" * 80)
print()

# Unload model
del llm
clear_gpu()

# Enable deterministic mode
set_deterministic_mode(True)

# Reload model
print("Reloading model with deterministic settings...")
try:
    llm = LLM(
        model=MODEL_NAME,
        quantization=QUANTIZATION,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        trust_remote_code=True,
        seed=SEED,
        enforce_eager=True,
        enable_prefix_caching=False
    )
    print("✓ Model loaded with deterministic mode")
    print()
    
    # Warmup
    print("Warmup...")
    for _ in range(3):
        _ = llm.generate(prompt_text, sampling_params=sampling_params)
    clear_gpu()
    print("✓ Warmup complete")
    print()
    
    # Run with deterministic mode
    det_tokens, det_logprobs, det_dists = run_repetitions(
        llm, prompt_text, sampling_params, NUM_REPETITIONS, "DETERMINISTIC"
    )
    
    det_stats = analyze_reproducibility(det_tokens, det_logprobs, "DETERMINISTIC")
    print()
    
    deterministic_succeeded = True

except Exception as e:
    print(f"⚠ Deterministic mode failed: {e}")
    print("This suggests deterministic algorithms don't support all operations")
    print()
    deterministic_succeeded = False
    det_stats = None

# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================

print("=" * 80)
print("COMPARATIVE ANALYSIS")
print("=" * 80)
print()

print("Baseline vs Deterministic Mode:")
print("-" * 60)
print(f"{'Metric':<30} {'Baseline':<20} {'Deterministic':<20}")
print("-" * 60)

if deterministic_succeeded and det_stats:
    print(f"{'Tokens identical':<30} {str(baseline_stats['tokens_identical']):<20} {str(det_stats['tokens_identical']):<20}")
    print(f"{'Logprobs exact':<30} {str(baseline_stats['logprobs_exact']):<20} {str(det_stats['logprobs_exact']):<20}")
    print(f"{'Mean L2 distance':<30} {baseline_stats['l2_mean']:.6e}  {det_stats['l2_mean']:.6e}")
    print(f"{'Max L2 distance':<30} {baseline_stats['l2_max']:.6e}  {det_stats['l2_max']:.6e}")
    print(f"{'Per-token std (mean)':<30} {baseline_stats['std_per_token_mean']:.6e}  {det_stats['std_per_token_mean']:.6e}")
    print(f"{'Per-token std (max)':<30} {baseline_stats['std_per_token_max']:.6e}  {det_stats['std_per_token_max']:.6e}")
else:
    print(f"{'Tokens identical':<30} {str(baseline_stats['tokens_identical']):<20} {'N/A':<20}")
    print(f"{'Logprobs exact':<30} {str(baseline_stats['logprobs_exact']):<20} {'N/A':<20}")
    print(f"{'Mean L2 distance':<30} {baseline_stats['l2_mean']:.6e}  {'N/A':<20}")

print()

# ============================================================================
# VERDICT
# ============================================================================

print("=" * 80)
print("VERDICT")
print("=" * 80)
print()

if not baseline_stats['logprobs_exact']:
    print("❌ BASELINE IS NON-DETERMINISTIC")
    print(f"   Mean L2 deviation: {baseline_stats['l2_mean']:.6e}")
    print(f"   Max L2 deviation: {baseline_stats['l2_max']:.6e}")
    print()
    
    if deterministic_succeeded and det_stats:
        if det_stats['logprobs_exact']:
            print("✓ DETERMINISTIC MODE FIXES IT")
            print("  → Root cause: Non-deterministic parallel reduction")
            print("  → Likely in Marlin kernel accumulation logic")
            print("  → Can be fixed but at performance cost")
        else:
            improvement = baseline_stats['l2_mean'] / det_stats['l2_mean'] if det_stats['l2_mean'] > 0 else float('inf')
            if improvement > 2:
                print("⚠ DETERMINISTIC MODE REDUCES NOISE")
                print(f"  → {improvement:.1f}x reduction in mean L2")
                print("  → Partial fix, some non-determinism remains")
            else:
                print("❌ DETERMINISTIC MODE DOESN'T HELP")
                print("  → Root cause is NOT parallel reduction races")
                print("  → Likely in quantization/dequantization logic itself")
    else:
        print("⚠ Could not test deterministic mode (not supported)")
        print("  → Suggests operations incompatible with deterministic algorithms")
    
    print()
    print("Forensic implications:")
    print(f"  Noise level: ~{baseline_stats['l2_mean']:.2e} L2")
    print("  For detection, need systematic deviation >> noise")
    print("  Recommend: 3-5 samples for statistical significance")
else:
    print("✓ BASELINE IS DETERMINISTIC")
    print("  → Unexpected! AWQ should be non-deterministic in vLLM")
    print("  → May have been fixed in recent version")

print()

# ============================================================================
# SAVE RESULTS
# ============================================================================

output_data = {
    "experiment": "awq_nondeterminism_investigation",
    "timestamp": datetime.now().isoformat(),
    "model": MODEL_NAME,
    "quantization": QUANTIZATION,
    "config": {
        "tensor_parallel": TENSOR_PARALLEL_SIZE,
        "max_model_len": MAX_MODEL_LEN,
        "max_tokens": MAX_TOKENS,
        "repetitions": NUM_REPETITIONS,
        "temperature": TEMPERATURE,
        "seed": SEED
    },
    "prompt_length": prompt_length,
    "baseline": {
        "stats": baseline_stats,
        "tokens": baseline_tokens,
        "logprobs": [lp.tolist() for lp in baseline_logprobs]
    },
    "deterministic": {
        "succeeded": deterministic_succeeded,
        "stats": det_stats if det_stats else None,
        "tokens": det_tokens if deterministic_succeeded else None,
        "logprobs": [lp.tolist() for lp in det_logprobs] if deterministic_succeeded else None
    },
    "kernels": {
        "profiling_succeeded": profiling_succeeded,
        "all_kernels": kernel_data if profiling_succeeded else [],
        "marlin_kernels": marlin_kernels if profiling_succeeded else [],
        "gemm_kernels": gemm_kernels if profiling_succeeded else [],
        "dequant_kernels": dequant_kernels if profiling_succeeded else []
    }
}

output_file = f"awq_investigation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"Results saved to: {output_file}")
print()
print("=" * 80)
print("INVESTIGATION COMPLETE")
print("=" * 80)
```

