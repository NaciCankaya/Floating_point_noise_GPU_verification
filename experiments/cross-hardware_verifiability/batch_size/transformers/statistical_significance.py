import json
import numpy as np

# Load the verification results
with open('transformers_A100generation_H100verification_sdpa.json', 'r') as f:  # <- update filename
    data = json.load(f)

batch_sizes = data['metadata']['batch_sizes']
comparisons = data['comparisons']

# Extract diagonal vs off-diagonal for decode
diagonal_values = []
off_diagonal_values = []
off_diagonal_meaningful = []  # Excluding equivalent pairs

equivalent_pairs = set()
for p in data['metadata'].get('decode_equivalent_pairs', []):
    equivalent_pairs.add((p[0], p[1]))
    equivalent_pairs.add((p[1], p[0]))

for comp in comparisons:
    claimed = comp['claimed_batch_size']
    verify = comp['verify_batch_size']
    dist = comp['decode_distances']['key_vectors_mean']  # or 'logprobs_mean'
    
    if claimed == verify:
        diagonal_values.append(dist)
    else:
        off_diagonal_values.append(dist)
        if (claimed, verify) not in equivalent_pairs:
            off_diagonal_meaningful.append(dist)

diag = np.array(diagonal_values)
off = np.array(off_diagonal_values)
off_meaningful = np.array(off_diagonal_meaningful)

print("="*60)
print("DIAGONAL (same batch size, cross-hardware)")
print("="*60)
print(f"  n = {len(diag)}")
print(f"  mean = {diag.mean():.4f}")
print(f"  std  = {diag.std():.4f}")
print(f"  range = [{diag.min():.4f}, {diag.max():.4f}]")

print("\n" + "="*60)
print("OFF-DIAGONAL (different batch size, cross-hardware)")
print("="*60)
print(f"  n = {len(off)}")
print(f"  mean = {off.mean():.4f}")
print(f"  std  = {off.std():.4f}")
print(f"  range = [{off.min():.4f}, {off.max():.4f}]")

print("\n" + "="*60)
print("OFF-DIAGONAL MEANINGFUL (excluding equivalent pairs)")
print("="*60)
print(f"  n = {len(off_meaningful)}")
print(f"  mean = {off_meaningful.mean():.4f}")
print(f"  std  = {off_meaningful.std():.4f}")
print(f"  range = [{off_meaningful.min():.4f}, {off_meaningful.max():.4f}]")

print("\n" + "="*60)
print("SEPARATION ANALYSIS")
print("="*60)

# Check overlap
print(f"\nDiagonal max:        {diag.max():.4f}")
print(f"Off-diagonal min:    {off_meaningful.min():.4f}")

if diag.max() < off_meaningful.min():
    print("✓ PERFECT SEPARATION - no overlap")
else:
    overlap = diag.max() - off_meaningful.min()
    print(f"✗ OVERLAP of {overlap:.4f}")

# Cohen's d
pooled_std = np.sqrt((diag.std()**2 + off_meaningful.std()**2) / 2)
if pooled_std > 0:
    cohens_d = (off_meaningful.mean() - diag.mean()) / pooled_std
    print(f"\nCohen's d: {cohens_d:.2f}")
    if cohens_d > 0.8:
        print("  → Large effect size")
    elif cohens_d > 0.5:
        print("  → Medium effect size")
    else:
        print("  → Small effect size")

# SNR with std
snr = off_meaningful.mean() / diag.mean() if diag.mean() > 0 else float('inf')
print(f"\nSNR: {snr:.2f}x")

# t-test
from scipy import stats
t_stat, p_value = stats.ttest_ind(off_meaningful, diag)
print(f"\nt-test: t={t_stat:.2f}, p={p_value:.2e}")
if p_value < 0.05:
    print("  → Statistically significant (p < 0.05)")
else:
    print("  → NOT statistically significant")

# Percentile analysis
print("\n" + "="*60)
print("DISTRIBUTION OVERLAP")
print("="*60)
# What fraction of off-diagonal is above diagonal max?
frac_above = (off_meaningful > diag.max()).mean()
print(f"Off-diagonal values above diagonal max: {frac_above*100:.1f}%")

# What fraction of diagonal is below off-diagonal min?
frac_below = (diag < off_meaningful.min()).mean()
print(f"Diagonal values below off-diagonal min: {frac_below*100:.1f}%")