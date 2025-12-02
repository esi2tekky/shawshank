# vLLM Setup Guide

This guide covers setting up vLLM and GPU access for running local models (Tulu 3 suite) in the Shawshank framework.

**Note**: This project uses **Google Cloud Platform (GCP)** with remote GPU instances. You can use remote GCP instances instead of physical GPUs.

## Table of Contents

1. [GCP Setup (Recommended)](#gcp-setup-recommended)
2. [System Requirements](#system-requirements)
3. [Installing Dependencies](#installing-dependencies)
4. [GPU Setup and Verification](#gpu-setup-and-verification)
5. [Hugging Face Configuration](#hugging-face-configuration)
6. [Testing the Setup](#testing-the-setup)
7. [Troubleshooting](#troubleshooting)

---

## GCP Setup (Recommended)

### Overview

This project uses **Google Cloud Platform (GCP)** with GPU-enabled compute instances. You can create and use remote GPU instances instead of requiring physical GPUs on your local machine.

### Step 1: Create GCP Project and Enable APIs

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the following APIs:
   - Compute Engine API
   - Cloud Resource Manager API

### Step 2: Request GPU Quota

1. Navigate to [IAM & Admin > Quotas](https://console.cloud.google.com/iam-admin/quotas)
2. Filter by "GPU" and your region
3. Request quota increase for:
   - **NVIDIA T4 GPUs**: Request at least 1 (for testing)
   - **NVIDIA A100 GPUs**: Request 1+ (recommended for Tulu 3 8B models, 40GB+ VRAM)
   - **NVIDIA L4 GPUs**: Alternative option (24GB VRAM)

**Note**: Quota approval may take 24-48 hours.

### Step 3: Create GPU Instance

#### Option A: Using gcloud CLI

```bash
# Install Google Cloud SDK if not already installed
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Create GPU instance (example with A100)
gcloud compute instances create vllm-gpu-instance \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-a100,count=1 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --metadata=install-nvidia-driver=True
```

#### Option B: Using GCP Console

1. Go to [Compute Engine > VM instances](https://console.cloud.google.com/compute/instances)
2. Click "Create Instance"
3. Configure:
   - **Name**: `vllm-gpu-instance`
   - **Region/Zone**: Choose a zone with GPU availability (e.g., `us-central1-a`)
   - **Machine type**: `n1-standard-8` or higher (8+ vCPUs recommended)
   - **GPU**: 
     - Type: `NVIDIA A100` (recommended) or `NVIDIA T4` (budget option)
     - Count: `1`
   - **Boot disk**: 
     - OS: Ubuntu 22.04 LTS
     - Size: 200GB+ (SSD recommended)
   - **Firewall**: Allow HTTP/HTTPS traffic (if needed)
4. Under "Advanced options > Management":
   - Check "Install NVIDIA GPU driver automatically"
5. Click "Create"

### Step 4: Connect to Instance

```bash
# SSH into the instance
gcloud compute ssh vllm-gpu-instance --zone=us-central1-a

# Or use SSH from console (click "SSH" button in GCP Console)
```

### Step 5: Verify GPU on Instance

Once connected to the instance:

```bash
# Check GPU
nvidia-smi

# You should see your GPU listed (e.g., A100, T4, L4)
```

### Step 6: Install Dependencies on GCP Instance

Follow the [Installing Dependencies](#installing-dependencies) section below, but note:

- The instance already has NVIDIA drivers (if you enabled auto-install)
- You may need to install CUDA toolkit separately
- Use the same Python/pip installation steps

### Step 7: Transfer Code to Instance

#### Option A: Clone Repository on Instance

```bash
# On the GCP instance
git clone https://github.com/esi2tekky/shawshank.git
cd shawshank
```

#### Option B: Use gcloud to Transfer Files

```bash
# From your local machine
gcloud compute scp --recurse ./shawshank vllm-gpu-instance:~/ --zone=us-central1-a
```

### Step 8: Set Up Environment Variables

On the GCP instance, set up your environment:

```bash
# Set OpenAI API key (for judge)
export OPENAI_API_KEY="your-key-here"

# Login to Hugging Face
huggingface-cli login
```

### GCP Cost Considerations

- **A100 GPU**: ~$3-4/hour (40GB VRAM, recommended)
- **T4 GPU**: ~$0.35-0.50/hour (16GB VRAM, budget option)
- **L4 GPU**: ~$0.75-1.00/hour (24GB VRAM, good balance)
- **Storage**: ~$0.17/GB/month for persistent disk
- **Network**: Egress charges apply for data transfer

**Cost-saving tips**:
- Stop instances when not in use: `gcloud compute instances stop vllm-gpu-instance --zone=us-central1-a`
- Use preemptible instances for lower cost (with risk of termination)
- Delete instances when done: `gcloud compute instances delete vllm-gpu-instance --zone=us-central1-a`

### GCP Instance Management

```bash
# Start instance
gcloud compute instances start vllm-gpu-instance --zone=us-central1-a

# Stop instance (preserves disk)
gcloud compute instances stop vllm-gpu-instance --zone=us-central1-a

# Delete instance (deletes disk unless using persistent disk)
gcloud compute instances delete vllm-gpu-instance --zone=us-central1-a

# List instances
gcloud compute instances list
```

---

## System Requirements

### Hardware Requirements

**Option 1: GCP Remote Instance (Recommended)**
- **GPU Instance**: NVIDIA A100 (40GB), L4 (24GB), or T4 (16GB)
  - A100 recommended for Tulu 3 8B models
  - T4 minimum for testing (may have memory constraints)
- **Machine Type**: n1-standard-8 or higher (8+ vCPUs, 30GB+ RAM)
- **Storage**: 200GB+ SSD boot disk
- **Network**: Stable internet connection for model downloads

**Option 2: Local Physical GPU**
- **GPU**: NVIDIA GPU with CUDA support (minimum 8GB VRAM recommended)
  - For Tulu 3 8B models: 16GB+ VRAM recommended
  - Check your GPU: `nvidia-smi`
- **RAM**: 16GB+ system RAM recommended
- **Storage**: ~20GB free space per model (for model weights)

### Software Requirements

- **Python**: 3.8 - 3.11 (Python 3.10 recommended)
- **CUDA**: 11.8 or 12.1+ (check compatibility with your GPU)
- **cuDNN**: Compatible version for your CUDA installation
- **NVIDIA Drivers**: Latest stable drivers for your GPU

---

## Installing Dependencies

### Step 1: Verify Python Environment

**On GCP Instance or Local Machine:**

```bash
python3 --version  # Should be 3.8-3.11
```

If you don't have a virtual environment, create one:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Note**: GCP instances typically come with Python 3.10 pre-installed.

### Step 2: Install PyTorch with CUDA Support

First, install PyTorch with CUDA. Check [PyTorch installation guide](https://pytorch.org/get-started/locally/) for the correct command for your system.

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU-only (not recommended for vLLM):**
```bash
pip install torch torchvision torchaudio
```

### Step 3: Install vLLM and Other Dependencies

Install vLLM and other required packages:

```bash
pip install vllm
```

Or install all project dependencies at once:

```bash
pip install -r requirements.txt
```

**Note**: vLLM installation may take several minutes as it compiles CUDA kernels.

### Step 4: Verify vLLM Installation

```bash
python3 -c "import vllm; print('vLLM installed successfully')"
```

If you encounter import errors, see the [Troubleshooting](#troubleshooting) section.

---

## GPU Setup and Verification

### Step 1: Check NVIDIA Drivers

**On GCP Instance:**
```bash
nvidia-smi
```

**On Local Machine:**
```bash
nvidia-smi
```

You should see output showing your GPU(s), driver version, and CUDA version. 

**GCP Note**: If you enabled "Install NVIDIA GPU driver automatically" when creating the instance, drivers should already be installed. If `nvidia-smi` fails, you may need to install drivers manually or restart the instance.

### Step 2: Verify CUDA Installation

Check CUDA version:

```bash
nvcc --version
```

Or check via PyTorch:

```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### Step 3: Test GPU Access from Python

Run this test script to verify GPU access:

```bash
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    print('WARNING: CUDA not available. vLLM requires GPU support.')
"
```

**Expected output** (if GPU is available):
```
PyTorch version: 2.x.x+cu118
CUDA available: True
CUDA version: 11.8
GPU device: NVIDIA GeForce RTX 3090
GPU memory: 24.00 GB
```

---

## Hugging Face Configuration

### Step 1: Create Hugging Face Account

1. Go to [huggingface.co](https://huggingface.co) and create an account
2. Navigate to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Create a new token with "Read" permissions

### Step 2: Login to Hugging Face

**On GCP Instance or Local Machine:**

Install the Hugging Face CLI:

```bash
pip install huggingface_hub
```

Login with your token:

```bash
huggingface-cli login
```

Enter your token when prompted. This will save your credentials for model downloads.

**Note**: On GCP instances, you'll need to login once per instance. Credentials are stored in `~/.cache/huggingface/`.

### Step 3: Verify Access

Test downloading a small model to verify access:

```bash
python3 -c "
from huggingface_hub import snapshot_download
print('Testing Hugging Face access...')
snapshot_download('allenai/Llama-3.1-Tulu-3-8B-SFT', cache_dir='./test_cache', local_files_only=False)
print('Access verified!')
"
```

**Note**: This will download ~16GB, so ensure you have sufficient disk space and bandwidth.

---

## Testing the Setup

### Step 1: Test Target Factory

Verify that the target factory can detect your setup:

```bash
python3 -c "
from src.target.target_factory import load_target
print('Testing target factory...')
# This should work without loading a model
print('Factory imported successfully')
"
```

### Step 2: Test vLLM Target (Dry Run)

Test loading a small model or verify the vLLM target class:

```bash
python3 -c "
from src.target.vllm_target import VLLMTarget
print('vLLM target class imported successfully')
print('Note: Actual model loading requires GPU and ~16GB VRAM')
"
```

### Step 3: Run Comparative Scan (Optional)

Test loading models without running full experiments:

```bash
python src/experiments/run_comparative_scan.py --targets tulu_sft
```

**Warning**: This will download and load the full model (~16GB download, requires GPU memory).

### Step 4: Test Full Pipeline

Once everything is set up, test with a small subset:

```bash
# Create a small test file
head -n 3 data/human_baseline.jsonl > data/test_baseline.jsonl

# Run on one model
python src/experiments/run_human_baseline_multi.py \
  --seeds data/test_baseline.jsonl \
  --models tulu_sft
```

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Symptoms**: Error message about CUDA out of memory when loading models.

**Solutions**:
1. Reduce `gpu_memory_utilization` in `vllm_target.py` (default 0.9, try 0.7 or 0.5)
2. **On GCP**: Upgrade to a larger GPU instance (A100 40GB instead of T4 16GB)
3. **On Local**: Close other GPU-intensive applications
4. Use a smaller model if available
5. Ensure you have sufficient VRAM (16GB+ recommended for Tulu 3 8B)
6. **On GCP**: Check instance GPU type with `nvidia-smi` - T4 may be insufficient for Tulu 3 8B

### Issue: "ModuleNotFoundError: No module named 'vllm'"

**Solutions**:
1. Ensure virtual environment is activated: `source venv/bin/activate`
2. Reinstall vLLM: `pip install --upgrade vllm`
3. Check Python version compatibility (vLLM requires Python 3.8-3.11)

### Issue: "CUDA not available" or "torch.cuda.is_available() returns False"

**Solutions**:
1. **On GCP**: Verify NVIDIA drivers: `nvidia-smi` - if it fails, the instance may need driver installation or restart
2. **On GCP**: Ensure you selected "Install NVIDIA GPU driver automatically" when creating the instance
3. **On Local**: Verify NVIDIA drivers: `nvidia-smi`
4. Reinstall PyTorch with CUDA support (see [Installing Dependencies](#installing-dependencies))
5. Check CUDA version compatibility
6. **On GCP**: Restart the instance: `sudo reboot`
7. **On Local**: Restart your system after driver installation

### Issue: "Hugging Face authentication failed"

**Solutions**:
1. Re-login: `huggingface-cli login`
2. Check token permissions (needs "Read" access)
3. Verify token is valid in [Hugging Face settings](https://huggingface.co/settings/tokens)

### Issue: "Model download timeout" or "Connection error"

**Solutions**:
1. Check internet connection
2. Models are large (~16GB), ensure sufficient bandwidth
3. Use Hugging Face mirror if in restricted region
4. Download models manually and use local path

### Issue: "vLLM compilation errors"

**Symptoms**: Errors during vLLM installation about CUDA kernels.

**Solutions**:
1. Ensure CUDA toolkit is installed: `nvcc --version`
2. Install compatible CUDA version (11.8 or 12.1+)
3. Try installing from source: `pip install git+https://github.com/vllm-project/vllm.git`
4. Check vLLM GitHub issues for your specific error

### Issue: "GPU memory not released between models"

**Symptoms**: Second model fails to load due to insufficient memory.

**Solutions**:
1. The `run_human_baseline_multi.py` script includes automatic cleanup
2. Manually clear cache if needed:
   ```python
   import gc
   import torch
   gc.collect()
   torch.cuda.empty_cache()
   ```
3. Ensure models are properly deleted before loading the next one

### Issue: "Model loading is very slow"

**Solutions**:
1. First load is slow (model download and initialization)
2. Subsequent loads should be faster (cached weights)
3. Ensure you're using SSD storage, not HDD
4. Check GPU compute capability (older GPUs may be slower)

---

## Quick Reference

### Essential Commands

**On GCP Instance or Local Machine:**

```bash
# Check GPU status
nvidia-smi

# Verify CUDA in Python
python3 -c "import torch; print(torch.cuda.is_available())"

# Test vLLM import
python3 -c "import vllm; print('OK')"

# Login to Hugging Face
huggingface-cli login

# Test target factory
python3 -c "from src.target.target_factory import load_target; print('OK')"

# Run comparative scan (downloads models)
python src/experiments/run_comparative_scan.py --targets tulu_sft
```

**GCP-Specific Commands:**

```bash
# Connect to instance
gcloud compute ssh vllm-gpu-instance --zone=us-central1-a

# Start instance
gcloud compute instances start vllm-gpu-instance --zone=us-central1-a

# Stop instance (to save costs)
gcloud compute instances stop vllm-gpu-instance --zone=us-central1-a

# Transfer files to instance
gcloud compute scp file.txt vllm-gpu-instance:~/ --zone=us-central1-a
```

### System Information Script

Create a file `check_setup.py`:

```python
#!/usr/bin/env python3
"""Quick setup verification script."""

import sys

print("=" * 60)
print("Shawshank vLLM Setup Check")
print("=" * 60)

# Python version
print(f"\nPython version: {sys.version}")

# PyTorch
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
except ImportError:
    print("ERROR: PyTorch not installed")

# vLLM
try:
    import vllm
    print(f"\nvLLM version: {vllm.__version__}")
except ImportError:
    print("\nERROR: vLLM not installed")
except AttributeError:
    print("\nvLLM installed (version not available)")

# Hugging Face
try:
    from huggingface_hub import whoami
    user = whoami()
    print(f"\nHugging Face: Logged in as {user.get('name', 'unknown')}")
except Exception as e:
    print(f"\nHugging Face: Not logged in or error ({e})")

# Project imports
try:
    from src.target.target_factory import load_target
    from src.target.vllm_target import VLLMTarget
    from src.config import OPEN_SOURCE_MODELS
    print("\nProject imports: OK")
    print(f"Available Tulu models: {list(OPEN_SOURCE_MODELS.keys())}")
except ImportError as e:
    print(f"\nERROR: Project import failed: {e}")

print("\n" + "=" * 60)
```

Run it:
```bash
python3 check_setup.py
```

---

## Next Steps

Once setup is complete:

1. **Test with small dataset**: Run human baseline on a few prompts first
2. **Monitor GPU usage**: Use `nvidia-smi -l 1` to watch GPU memory
3. **Run full experiments**: Execute `run_human_baseline_multi.py` on all models
4. **Check results**: Verify output files in `results/human_baseline/`

For questions or issues, refer to:
- [vLLM Documentation](https://docs.vllm.ai/)
- [PyTorch CUDA Guide](https://pytorch.org/get-started/locally/)
- [Hugging Face Documentation](https://huggingface.co/docs)

