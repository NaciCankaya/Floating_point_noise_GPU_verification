# RunPod Setup Guide for Cross-Hardware GPU Experiments

This guide helps you efficiently run experiments across different RunPod datacenters with various GPU types.

## The Challenge

Network volumes in RunPod are datacenter-specific, but our cross-hardware experiments require access to different GPU types (A100, H100, RTX 4090, etc.) across multiple datacenters. **Solution: Use GitHub as central storage.**

## Quick Start

### Option 1: Using Jupyter (Recommended for Interactive Work)

1. **Start a new RunPod instance** with Jupyter template
2. **Clone this repository** (if starting fresh):
   ```bash
   git clone https://github.com/NaciCankaya/Floating_point_noise_GPU_verification.git
   cd Floating_point_noise_GPU_verification
   ```
3. **Open and run** `runpod_setup.ipynb`
4. **Run your experiments**
5. **Commit and push results** before terminating

### Option 2: Using Bash Script

1. **Start a new RunPod instance**
2. **Clone this repository** (if starting fresh):
   ```bash
   git clone https://github.com/NaciCankaya/Floating_point_noise_GPU_verification.git
   cd Floating_point_noise_GPU_verification
   ```
3. **Run the setup script**:
   ```bash
   ./setup_runpod.sh
   ```
4. **Run your experiments**
5. **Commit and push results** before terminating

## First-Time Git Configuration

On your first RunPod instance, configure git credentials:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Authentication Options

**Option A: HTTPS with Personal Access Token (Easiest)**
1. Create a token at: https://github.com/settings/tokens
2. When pushing, use token as password
3. Cache credentials: `git config --global credential.helper store`

**Option B: SSH Keys (More Secure)**
```bash
# Generate key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Display public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: Settings → SSH and GPG keys
```

## Workflow for Each New Pod

```bash
# 1. Clone repo (or pull if already present)
git clone https://github.com/NaciCankaya/Floating_point_noise_GPU_verification.git
cd Floating_point_noise_GPU_verification

# OR if repo already exists:
git pull origin main

# 2. Run setup (installs dependencies)
./setup_runpod.sh
# OR open runpod_setup.ipynb in Jupyter

# 3. Run your experiment
# ... your code here ...

# 4. Commit results with GPU metadata
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
git add results/
git commit -m "Experiment results from ${GPU_TYPE}"

# 5. Push to GitHub
git push origin main

# 6. Terminate pod (no data loss!)
```

## Creating a Custom RunPod Template (Advanced)

To automate even more, create a custom RunPod template:

1. **In RunPod Dashboard**: Templates → Create Template
2. **Container Image**: Use a PyTorch/CUDA base image
3. **Docker Command**: Add startup script:
   ```bash
   git clone https://github.com/NaciCankaya/Floating_point_noise_GPU_verification.git && \
   cd Floating_point_noise_GPU_verification && \
   pip install -r requirements.txt
   ```
4. **Environment Variables**: Add your git credentials securely

Now each new pod will automatically:
- Clone the latest code
- Install all dependencies
- Be ready to run experiments

## Dependencies

All dependencies are listed in `requirements.txt`:
- transformers
- hf_transfer
- accelerate
- ninja, packaging, wheel
- flash-attn (takes 5-10 min to compile)

## Tips

- ✅ **Always push before terminating** - RunPod pods are ephemeral!
- 📊 **Label commits with GPU type** - helps track which hardware produced which results
- 🚀 **Use `-q` flag** for pip to reduce output noise: `pip install -q package`
- ⏱️ **flash-attn takes time** - plan for 5-10 minutes during setup
- 🔄 **Small repo = fast cloning** - Our 54MB repo clones in under a minute

## File Structure

```
.
├── requirements.txt          # Python dependencies
├── setup_runpod.sh          # Bash setup script
├── runpod_setup.ipynb       # Jupyter setup notebook
├── RUNPOD_GUIDE.md          # This file
└── experiments/             # Your experiment notebooks/scripts
```

## Troubleshooting

**Q: Git push asks for password every time**
A: Use credential caching: `git config --global credential.helper store`

**Q: flash-attn installation fails**
A: Ensure you have enough memory and the correct CUDA version. Try with `--no-build-isolation` flag.

**Q: Lost my changes when pod terminated**
A: This is why we commit and push! Always push before terminating.

**Q: Different datacenter = different setup?**
A: No! This workflow is datacenter-agnostic. Just clone and run setup.

## Need Help?

Open an issue on GitHub or check the main README.md for project-specific information.
