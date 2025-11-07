import os
os.environ['HF_HOME'] = '/workspace/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/workspace/huggingface_cache'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from datetime import datetime
import json
import socket

# Capture system info for verification
HOSTNAME = socket.gethostname()
CONTAINER_ID = os.environ.get('HOSTNAME', 'unknown')

print(f"System Info:")
print(f"  Hostname: {HOSTNAME}")
print(f"  Container: {CONTAINER_ID}")
print()

# Capture relevant environment variables
print("Environment Variables:")
env_vars = {}
for key in sorted(os.environ.keys()):
    if any(x in key.upper() for x in ['CUDA', 'TORCH', 'NCCL', 'CUDNN', 'PYTORCH']):
        env_vars[key] = os.environ[key]
        print(f"  {key}={os.environ[key]}")
if not env_vars:
    print("  (No CUDA/TORCH env vars set)")
print()

# Similar-length dummy sequences (~1100 tokens each, similar to base prompt)
DUMMY_SEQUENCES = [
    """The automated data-processing pipeline ingests raw telemetry from distributed sensors across multiple geographic locations. A proprietary algorithm then normalizes the dataset, filtering for anomalies based on predefined statistical parameters derived from historical patterns. The resulting output is a clean, structured matrix ready for machine learning model ingestion and downstream analytical workflows. System efficiency is monitored in real-time through a comprehensive dashboard, with automated alerts triggered if latency exceeds the established threshold or if data quality metrics fall below acceptable ranges. This protocol ensures data integrity and operational stability for all downstream analytical processes, maintaining compliance with industry standards and regulatory requirements. The architecture supports horizontal scaling to accommodate growing data volumes while maintaining sub-second response times for critical monitoring functions. Advanced compression techniques and intelligent caching strategies optimize storage utilization and reduce network bandwidth requirements across the distributed infrastructure. Regular audits and automated testing procedures validate system performance and data accuracy, ensuring continuous improvement and operational excellence in all aspects of the data pipeline. Security measures include end-to-end encryption, role-based access controls, and comprehensive audit logging to protect sensitive information and maintain system integrity throughout the entire data lifecycle. Integration with existing enterprise systems follows established API standards and authentication protocols, enabling seamless data exchange and interoperability across the organizational technology stack. Performance benchmarks demonstrate consistent throughput of millions of records per hour with minimal resource consumption, achieving cost-efficiency targets while exceeding service level agreements for data freshness and availability. The modular design allows for incremental updates and feature additions without disrupting ongoing operations, supporting agile development practices and rapid iteration cycles. Comprehensive documentation and training materials ensure effective knowledge transfer and support long-term maintainability of the system. Disaster recovery procedures and redundancy mechanisms provide business continuity guarantees, with automated failover capabilities and regular backup validations ensuring data protection and system resilience. Monitoring and alerting systems provide 24/7 visibility into system health and performance metrics, enabling proactive identification and resolution of potential issues before they impact service quality or user experience. The platform's extensible architecture accommodates custom plugins and integrations, allowing organizations to tailor functionality to specific business requirements and industry-specific use cases. Continuous performance optimization and capacity planning activities ensure the system maintains optimal performance as data volumes grow and usage patterns evolve over time. Regular security assessments and penetration testing validate the effectiveness of security controls and identify potential vulnerabilities for remediation. User feedback mechanisms and usage analytics inform ongoing product improvements and feature prioritization decisions. The system's design emphasizes reliability, scalability, and maintainability as core principles, ensuring long-term value and operational efficiency for all stakeholders involved in the data processing workflow.""",
    
    """An amber twilight bled across the distant horizon, painting the undersides of the scattered clouds in delicate hues of apricot and rose that seemed to pulse with their own inner luminescence. The wind, a cool whisper threading through the ancient pines standing sentinel on the hillside, carried with it the rich scent of damp earth and forgotten seasons, memories of spring rains and autumn harvests long past. Below the ridge, the valley settled gradually into a deep, contemplative blue, a world holding its collective breath in anticipation of the coming night. It was one of those rare moments suspended perfectly in time, where memory and sensation felt somehow more tangible and real than the solid ground beneath one's feet, a quiet echo reverberating through the chambers of the heart and mind. The trees stood motionless, their branches etched in sharp silhouette against the fading light, each leaf and needle rendered in exquisite detail by the golden hour's soft illumination. Somewhere in the distance, a bird called out its evening song, the notes clear and pure in the still air, a reminder of the persistent vitality of life even as day surrendered to darkness. The shadows lengthened and deepened, pooling in the hollows and valleys like dark water, while the highest peaks still caught and held the dying sunlight, glowing briefly with an otherworldly radiance before they too succumbed to twilight's advance. In this liminal space between day and night, the world seemed to reveal hidden truths and forgotten dreams, speaking in a language older than words, communicating directly with the soul through beauty and silence. The grass beneath moved gently in waves, responding to currents of air too subtle to feel, creating patterns that shifted and flowed like water or smoke, ephemeral and eternal all at once. Far above, the first stars began their nightly emergence, faint pinpricks of light gradually strengthening as the sky deepened from pale blue to indigo to the rich velvet black of true night. The temperature dropped perceptibly, the warmth of day giving way to the cool embrace of evening, a gentle reminder of the earth's eternal rhythms and cycles. In the growing darkness, other senses awakened and sharpened, the sounds of nocturnal creatures beginning their activities, the feel of dew beginning to form on leaf and blade, the smell of night-blooming flowers releasing their perfume into the air. This transformation, enacted countless times since the world began, never lost its power to move and inspire, to connect the present moment with all the moments that came before and all those yet to come. The boundary between observer and observed seemed to dissolve, subject and object merging in a unified experience of being and becoming, presence and absence, known and unknown all simultaneously existing in perfect balance and harmony.""",
    
    """Our comprehensive cross-functional Q3 strategic initiative is specifically designed to leverage our organization's core competencies and significantly enhance stakeholder engagement across all business units and geographic regions. By carefully operationalizing a new, highly scalable workflow management system, we aim to drive meaningful synergy across key vertical markets and horizontal integration points throughout the enterprise ecosystem. The project's ultimate success will be rigorously measured against a comprehensive suite of predefined key performance indicators, with detailed weekly progress reports delivered directly to the senior leadership team and board of directors for review and strategic guidance. This strategic pivot represents a fundamental transformation in how we approach market opportunities and will empower our diverse teams to consistently exceed ambitious quarterly targets while establishing a completely new paradigm for customer-centric value delivery and innovation. The initiative ensures a robust, sustainable market position through careful attention to emerging trends and competitive dynamics in our industry sector. Implementation phases will roll out sequentially across all departments, beginning with pilot programs in selected high-impact areas before scaling to full organizational deployment. Change management protocols have been established to facilitate smooth transitions and minimize disruption to ongoing operations and client deliverables. Training programs and resource allocation have been carefully planned to support team members throughout the transformation process, ensuring everyone has the tools and knowledge needed to succeed in the new operational framework. Regular checkpoints and milestone reviews will track progress against established timelines and allow for agile adjustments based on real-world feedback and evolving market conditions. Executive sponsorship and active involvement from senior leadership demonstrate the organization's commitment to the initiative's success and long-term sustainability. Risk mitigation strategies address potential challenges and contingencies, with backup plans ready for rapid deployment if circumstances require course corrections. Stakeholder communication plans ensure transparency and maintain alignment across all organizational levels and external partners. The initiative incorporates industry best practices and lessons learned from similar transformations, avoiding common pitfalls while capitalizing on proven success factors. Technology investments support the initiative's goals while maintaining fiscal responsibility and delivering measurable return on investment within projected timeframes. Cultural change elements address the human side of transformation, fostering adoption and building enthusiasm for new ways of working. Performance incentives and recognition programs reward desired behaviors and outcomes, reinforcing the importance of the initiative to organizational success.""",
    
    """Pursuant to the provisions set forth in Section 4.B of the aforementioned contractual agreement executed between the parties on the date first written above, the undersigned party, hereinafter referred to and designated as the "Recipient" throughout this document and all related materials, hereby formally acknowledges and unconditionally accepts all terms, conditions, obligations, and responsibilities set forth herein and incorporated by reference. Notwithstanding any and all prior written or oral communications, preliminary discussions, draft agreements, term sheets, letters of intent, or verbal understandings between the parties or their respective representatives, this document and its attached exhibits constitute the complete, entire, and exclusive understanding and binding legal agreement between the duly authorized parties with respect to the subject matter hereof. Any proposed modification, amendment, waiver, or alteration to any provision of this clause or any other section of this agreement must be executed in writing on official letterhead, signed by duly authorized representatives of both entities holding proper signatory authority, and delivered via certified mail or other verifiable means to the addresses specified in the notice provisions section of this agreement. Failure to strictly comply with any material provision of this agreement, whether through action or omission, shall be deemed and treated as a material breach of contract, entitling the non-breaching party to pursue all available remedies at law or in equity, including but not limited to specific performance, injunctive relief, monetary damages, and recovery of reasonable attorneys' fees and costs incurred in enforcement proceedings. The parties acknowledge and agree that time is of the essence with respect to all obligations and deadlines specified in this agreement, and that delays in performance may cause irreparable harm for which monetary damages may not constitute adequate compensation. This agreement shall be governed by and construed in accordance with the laws of the jurisdiction specified in the choice of law provision, without giving effect to any conflicts of law principles that might otherwise require application of the substantive law of a different jurisdiction. Any disputes arising out of or relating to this agreement or the breach, termination, or validity thereof shall be resolved exclusively through the dispute resolution procedures specified herein, and the parties hereby consent to the personal jurisdiction and venue of the courts or arbitration forums designated in the dispute resolution section. Each party represents and warrants that it has full legal capacity and authority to enter into this agreement and that the execution and performance of this agreement does not violate any other agreement, court order, or legal restriction to which such party is subject. The invalidity or unenforceability of any provision of this agreement shall not affect the validity or enforceability of any other provision, which shall remain in full force and effect."""
]

def collect_activations_parallel_batch(model, tokenizer, base_prompt, batch_size=1, device="cuda"):
    """Forward pass where element 0 is ALWAYS base_prompt (dummy_0), with other dummy sequences
    
    CRITICAL: All sequences are ~500 tokens to minimize padding artifacts
    FIXED: Extract from last VALID position using attention_mask, not position -1
           This ensures we extract from actual content, not padding tokens
    
    - bs=1: [dummy_0]
    - bs=2: [dummy_0, dummy_1]
    - bs=4: [dummy_0, dummy_1, dummy_2, dummy_3]
    """
    torch.cuda.empty_cache()
    
    if batch_size == 1:
        prompts = [base_prompt]  # dummy_0 only
    else:
        # Element 0 is dummy_0 (base_prompt)
        # Elements 1+ are dummy_1, dummy_2, dummy_3
        prompts = [base_prompt] + DUMMY_SEQUENCES[1:batch_size]
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    
    # DEBUG: Verify setup and check padding
    actual_batch_size = inputs['input_ids'].shape[0]
    seq_len = inputs['input_ids'].shape[1]
    if actual_batch_size != batch_size:
        print(f"WARNING: Expected batch_size={batch_size}, got {actual_batch_size}")
    
    if batch_size > 1:
        # Check sequence lengths to verify minimal padding
        for i in range(actual_batch_size):
            attn_sum = inputs['attention_mask'][i].sum().item()
            padding = seq_len - attn_sum
            if i == 0:
                print(f"  Seq lengths: elem[0]={int(attn_sum)} tokens, padding={int(padding)}", end="")
        
        # Verify element 0 and element 1 are different
        elem0_last5 = inputs['input_ids'][0, -5:].tolist()
        elem1_last5 = inputs['input_ids'][1, -5:].tolist()
        different = elem0_last5 != elem1_last5
        print(f", elem[0]≠elem[1]: {different}")
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    
    # CRITICAL FIX: Extract from last VALID (non-padded) position for element 0
    # Using -1 would extract from padding when batch has different length sequences
    last_valid_pos = inputs['attention_mask'][0].sum() - 1  # Last non-padded token
    last_layer_last_pos = outputs.hidden_states[-1][0, last_valid_pos, :].cpu().clone()
    
    # Debug: verify we're extracting from correct position
    if batch_size == 1:
        print(f"  Extracting from position {last_valid_pos.item()} (last valid token)")
    
    del outputs
    del inputs
    torch.cuda.empty_cache()
    
    return last_layer_last_pos

# Setup
CACHE_DIR = '/workspace/huggingface_cache'
EXP_NUMBER = 6  # Fixed: extract from last valid position, not padding
model_name = "Qwen/Qwen2.5-7B-Instruct"

print(f"Loading {model_name} in BF16...")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    cache_dir=CACHE_DIR,
    low_cpu_mem_usage=True,
    device_map="auto"
)

# Use dummy_0 as base prompt - all sequences are similar length (~1100 tokens)
prompt = DUMMY_SEQUENCES[0]  # Technical/data-processing sequence as base

prompt_tokens = len(tokenizer.encode(prompt))
dummy_tokens = [len(tokenizer.encode(seq)) for seq in DUMMY_SEQUENCES]
print(f"\nAll sequence lengths:")
print(f"  dummy_0 (base): {dummy_tokens[0]} tokens")
print(f"  dummy_1: {dummy_tokens[1]} tokens")
print(f"  dummy_2: {dummy_tokens[2]} tokens")
print(f"  dummy_3: {dummy_tokens[3]} tokens")
print("CRITICAL: All sequences ~1100 tokens, minimal padding\n")

# Test batch sizes 1, 2, 4 with 10 repetitions each
batch_sizes = [1, 2, 4]
num_repetitions = 10
results = {}
all_activations = {}

print(f"{'='*60}")
print(f"Starting H100 BALANCED-LENGTH TEST at {datetime.now().isoformat()}")
print(f"Model: {model_name}")
print(f"Precision: BF16 (bfloat16)")
print(f"Base prompt tokens: {prompt_tokens}")
print(f"Operation: Single forward pass (prefill only)")
print(f"CRITICAL: Element 0 input identical, dummies are similar length")
print(f"Repetitions per batch size: {num_repetitions}")
print(f"{'='*60}\n")

for bs in batch_sizes:
    print(f"Collecting batch_size={bs} ({num_repetitions} repetitions)...")
    if bs == 1:
        print(f"  Batch: [dummy_0]")
    elif bs == 2:
        print(f"  Batch: [dummy_0, dummy_1] (extracting from elem 0)")
    else:
        print(f"  Batch: [dummy_0, dummy_1, dummy_2, dummy_3] (extracting from elem 0)")
    
    runs = []
    for rep in range(num_repetitions):
        activation = collect_activations_parallel_batch(model, tokenizer, prompt, batch_size=bs, device="cuda")
        runs.append(activation)
        if rep == 0:
            print(f"  Rep 0: norm={torch.norm(activation).item():.6f}, first_val={activation[0].item():.6f}")
        if (rep + 1) % 3 == 0:
            print(f"  Completed {rep + 1}/{num_repetitions} repetitions")
    
    # Check repeatability
    first_rep = runs[0]
    all_identical = all(torch.equal(first_rep, runs[i]) for i in range(1, num_repetitions))
    if all_identical:
        print(f"  ✓ All {num_repetitions} repetitions identical (expected)")
    else:
        print(f"  ⚠ Repetitions vary (unexpected!)")
    
    results[bs] = torch.stack(runs)
    all_activations[f"batch_size_{bs}"] = results[bs].float().numpy().tolist()
    
    mean_activation = results[bs].mean(dim=0)
    deviations = torch.stack([torch.norm(results[bs][i] - mean_activation) for i in range(num_repetitions)])
    std_noise = deviations.std().item()
    mean_noise = deviations.mean().item()
    
    print(f"  Statistical noise: mean={mean_noise:.6f}, std={std_noise:.6f}")
    print(f"  Activation norm: {torch.norm(mean_activation).item():.2f}\n")
    
    torch.cuda.empty_cache()

# Compare systematic deviations
print("\n" + "="*60)
print("=== SYSTEMATIC DEVIATION MATRIX ===")
print("="*60)
print("     ", end="")
for bs in batch_sizes:
    print(f"bs={bs:2d}  ", end="")
print()

systematic_deviations = {}
for bs1 in batch_sizes:
    print(f"bs={bs1:2d} ", end="")
    for bs2 in batch_sizes:
        if bs1 == bs2:
            print("  -    ", end="")
        else:
            mean1 = results[bs1].mean(dim=0)
            mean2 = results[bs2].mean(dim=0)
            l2 = torch.norm(mean1 - mean2).item()
            systematic_deviations[f"bs{bs1}_vs_bs{bs2}"] = l2
            print(f"{l2:6.3f} ", end="")
    print()

print("\n" + "="*60)
print("=== ACTIVATION ANALYSIS ===")
print("="*60)

bs1_mean = results[1].mean(dim=0)
bs2_mean = results[2].mean(dim=0)

print(f"Dimension: {bs1_mean.shape[0]}")
print(f"bs=1 norm: {torch.norm(bs1_mean).item():.2f}")
print(f"bs=2 norm: {torch.norm(bs2_mean).item():.2f}")
print(f"L2 distance (bs1 vs bs2): {torch.norm(bs1_mean - bs2_mean).item():.4f}")
if torch.norm(bs1_mean) > 0:
    print(f"Relative difference: {(torch.norm(bs1_mean - bs2_mean) / torch.norm(bs1_mean)).item():.6f}")

diff = (bs1_mean - bs2_mean).abs()
print(f"Max absolute diff: {diff.max().item():.6f}")
print(f"Dims with |diff| > 0.01: {(diff > 0.01).sum().item()}/{diff.shape[0]}")

bs1_vs_bs2_deviation = systematic_deviations.get("bs1_vs_bs2", 0)
print("\n" + "="*60)
print("=== VERDICT ===")
print("="*60)
print(f"Element 0 input: dummy_0 (IDENTICAL across batch sizes)")
print(f"All sequences: Similar length (~{dummy_tokens[0]} tokens)")
print(f"bs1 vs bs2 deviation: {bs1_vs_bs2_deviation:.6f}\n")

if bs1_vs_bs2_deviation > 0.1:
    print(f"✓ DETECTION VIABLE: L2={bs1_vs_bs2_deviation:.4f}")
    print(f"  Forensics can detect hidden batch capacity on H100")
else:
    print(f"⚠ WEAK SIGNAL: L2={bs1_vs_bs2_deviation:.6f}")
    print(f"  Marginal detection capability")

# Save results
output = {
    "experiment": "H100_balanced_length_FIXED_extraction_position",
    "timestamp": datetime.now().isoformat(),
    "model": model_name,
    "hardware": {
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "hostname": HOSTNAME,
        "container_id": CONTAINER_ID
    },
    "environment": env_vars,
    "config": {
        "batch_sizes": batch_sizes,
        "repetitions": num_repetitions,
        "operation": "single_forward_pass_prefill_only",
        "dtype": "bfloat16",
        "use_cache": False,
        "input_strategy": "dummy_0_is_base_all_sequences_similar_length",
        "extraction_method": "last_valid_position_from_attention_mask",
        "base_prompt_tokens": prompt_tokens,
        "all_sequence_tokens": dummy_tokens,
        "hidden_dim": int(bs1_mean.shape[0])
    },
    "statistical_noise": {
        f"batch_size_{bs}": {
            "mean": float(torch.stack([torch.norm(results[bs][i] - results[bs].mean(dim=0)) 
                                       for i in range(num_repetitions)]).mean()),
            "std": float(torch.stack([torch.norm(results[bs][i] - results[bs].mean(dim=0)) 
                                     for i in range(num_repetitions)]).std())
        }
        for bs in batch_sizes
    },
    "systematic_deviations": systematic_deviations,
    "activation_norms": {
        f"batch_size_{bs}": float(torch.norm(results[bs].mean(dim=0))) 
        for bs in batch_sizes
    },
    "forensics_result": {
        "bs1_vs_bs2_deviation": bs1_vs_bs2_deviation,
        "detection_viable": bs1_vs_bs2_deviation > 0.1
    },
    "raw_activations": all_activations
}

output_file = f"{torch.cuda.get_device_name(0).replace(' ', '_')}_balanced_exp{EXP_NUMBER}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
output_path = f"/workspace/{output_file}"

with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Results saved to {output_path}")
print(f"✓ File size: ~{len(json.dumps(output)) / 1024:.1f} KB")
print("\n" + "="*60)
print("BALANCED-LENGTH TEST COMPLETE")
print("="*60)