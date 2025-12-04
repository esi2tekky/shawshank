# vLLM Setup Guide

This guide covers setting up vLLM and GPU access for running local models (Tulu 3 suite) in the Shawshank framework.

**Note**: This project uses **Amazon Web Services (AWS)** with remote GPU instances. You can use remote AWS EC2 instances instead of physical GPUs.

## Table of Contents

1. [AWS Setup (Recommended)](#aws-setup-recommended)
2. [System Requirements](#system-requirements)
3. [Installing Dependencies](#installing-dependencies)
4. [GPU Setup and Verification](#gpu-setup-and-verification)
5. [Hugging Face Configuration](#hugging-face-configuration)
6. [Testing the Setup](#testing-the-setup)
7. [Troubleshooting](#troubleshooting)

---

## AWS Setup (Recommended)

### Overview

This project uses **Amazon Web Services (AWS)** with GPU-enabled EC2 instances. You can create and use remote EC2 instances instead of requiring physical GPUs on your local machine.

### Step 1: Create AWS Account and Configure Access

1. Go to [AWS Console](https://console.aws.amazon.com/)
2. Sign in or create a new AWS account

#### Option A: Using AWS CLI with Access Keys (Standard Method)

3. Install AWS CLI (if not already installed):
   ```bash
   # macOS
   brew install awscli
   
   # Linux
   curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
   unzip awscliv2.zip
   sudo ./aws/install
   ```

4. Create Access Keys:
   - Go to [IAM Console > Users](https://console.aws.amazon.com/iam/) > Your User > Security Credentials
   - Click "Create access key"
   - Download or copy the Access Key ID and Secret Access Key

5. Configure AWS CLI:
   ```bash
   aws configure
   # Enter your Access Key ID, Secret Access Key, region (e.g., us-east-1), and output format (json)
   ```

#### Option B: Using AWS SSO (If Your Organization Uses SSO)

3. Install AWS CLI (same as above)

4. Configure SSO:
   ```bash
   aws configure sso
   # Follow prompts to set up SSO profile
   ```

5. Login to SSO:
   ```bash
   aws sso login --profile your-profile-name
   # This opens a browser for authentication
   ```

**Note**: If you're only using the AWS Console web interface (not CLI), you don't need to configure anything - just log in via the browser.

#### Verify Your Configuration

After configuring, verify your credentials are set up correctly:

```bash
# Check if credentials are configured
aws sts get-caller-identity

# If this works, you should see your AWS account ID and user ARN
# If it fails, you need to configure credentials (see above)
```

**Troubleshooting "Unable to locate credentials" error:**

1. **Check if you've run `aws configure`**:
   ```bash
   # Check if credentials file exists
   cat ~/.aws/credentials
   # Should show [default] section with aws_access_key_id and aws_secret_access_key
   ```

2. **If using SSO, make sure you've logged in**:
   ```bash
   aws sso login --profile your-profile-name
   ```

3. **Set default region in config** (if not already set):
   ```bash
   aws configure set region us-west-2
   ```

4. **Verify your configuration**:
   ```bash
   aws configure list
   # Should show your access key, secret key, and region
   ```

### Step 2: Request Service Limit Increase for GPU Instances

1. Navigate to [Service Quotas Console](https://console.aws.amazon.com/servicequotas/)
2. Search for "EC2" and select "Amazon Elastic Compute Cloud (Amazon EC2)"
3. Request quotas for your instance types:

   **For Current Work (Tulu 3 8B models):**
   - **Running On-Demand G instances**: Request quota for `g5.xlarge` (NVIDIA A10G, 24GB VRAM) - recommended
   - **All G and VT Spot Instance Requests**: You already have quota (8) - good for cost savings!

   **For Future RL Workloads:**
   - **Running On-Demand P instances**: Request quota for `p3.2xlarge` (V100) or `p4d.24xlarge` (A100) if needed
   - **All P Spot Instance Requests**: Request spot quota if planning to use spot instances for RL
   - Consider requesting higher G instance quota if you'll run multiple RL experiments in parallel

4. Click on each limit and request an increase
5. Fill out the request form with:
   - Desired limit (e.g., 1-4 instances for on-demand, higher for spot)
   - Use case description (mention ML research, model training/evaluation)
   - Contact information

**Note**: 
- Service limit approval typically takes 24-48 hours
- You can use **Spot Instances** immediately with your existing G quota (8) for significant cost savings
- For RL workloads, you may want 2-4 on-demand instances to avoid interruption during training

### Step 3: Create GPU Instance

#### Option A: Using AWS CLI

```bash
# Create a security group for SSH access (if not exists)
aws ec2 create-security-group \
  --group-name vllm-gpu-sg \
  --description "Security group for vLLM GPU instance" \
  --region us-west-2

# Add SSH rule to security group
aws ec2 authorize-security-group-ingress \
  --group-name vllm-gpu-sg \
  --protocol tcp \
  --port 22 \
  --cidr 0.0.0.0/0 \
  --region us-west-2

# Create GPU instance (g5.xlarge - A10G GPU)
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type g5.xlarge \
  --key-name 229project \
  --security-groups vllm-gpu-sg \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
  --region us-west-2
```

**Note**: 
- **Key pair**: `229project`
- **Region**: `us-west-2`
- Replace the AMI ID with the actual Deep Learning AMI ID for us-west-2 region (find it in EC2 Console > Launch Instance > AMI search)

#### Option B: Using AWS Console

1. Go to [EC2 Console > Instances](https://console.aws.amazon.com/ec2/)
2. Click "Launch Instance"
3. Configure:
   - **Name**: `vllm-gpu-instance`
   - **AMI**: Choose one of:
     - **Deep Learning Base AMI (Amazon Linux 2023)** - Recommended (includes CUDA, NVIDIA drivers pre-installed)
     - **Deep Learning AMI (Ubuntu)** - Alternative (includes CUDA, cuDNN, PyTorch pre-installed)
     - **Ubuntu Server 22.04 LTS** - Manual setup required (more control, more setup)
   - **Architecture**: Select **64-bit (x86)** (not Arm) for GPU support
   - **Instance type**: 
     - `g5.xlarge` (NVIDIA A10G, 24GB VRAM) - **recommended** (you have spot quota for this)
     - `g5.2xlarge` (NVIDIA A10G, 48GB VRAM) - if you need more VRAM
     - `p3.2xlarge` (NVIDIA V100, 16GB VRAM) - for RL if needed
     - `p4d.24xlarge` (NVIDIA A100, 40GB VRAM) - high performance for RL
   - **Purchase option**: 
     - **On-Demand** - guaranteed availability (request quota if needed)
     - **Spot Instances** - up to 90% savings (you have quota for G instances)
   - **Key pair**: Select or create a key pair for SSH access - called 229project
   - **Network settings**: 
     - Create or select a security group
     - Allow SSH (port 22) from your IP
   - **Configure storage**: 
     - 200GB+ gp3 SSD (default is usually sufficient)
4. Click "Launch Instance"

**Note**: If using Deep Learning AMI, NVIDIA drivers and CUDA are pre-installed. If using Ubuntu AMI, you'll need to install drivers manually (see Step 6).

### Step 4: Connect to Instance

```bash
# Find your instance's public IP (us-west-2 region)
aws ec2 describe-instances \
  --filters "Name=instance-state-name,Values=running" \
  --query "Reservations[*].Instances[*].[InstanceId,PublicIpAddress]" \
  --output table \
  --region us-west-2

# SSH into the instance using your key
ssh -i ~/.ssh/229project.pem ec2-user@YOUR_INSTANCE_IP

# Or use EC2 Instance Connect from AWS Console (click "Connect" button)
```

**Note**: 
- **Key pair**: `229project`
- **Region**: `us-west-2`
- For **Deep Learning AMI (Amazon Linux 2023)**: Default user is `ec2-user`
- For **Deep Learning AMI (Ubuntu)**: Default user is `ubuntu`
- For **Ubuntu Server**: Default user is `ubuntu`
- Make sure your key file has correct permissions: `chmod 400 ~/.ssh/229project.pem`

### Step 5: Verify GPU on Instance

Once connected to the instance:

```bash
# Check GPU
nvidia-smi

# You should see your GPU listed (e.g., A100, T4, L4)
```

### Step 6: Install NVIDIA Drivers and Dependencies on AWS Instance

**Important**: If you used **Deep Learning AMI** (Amazon Linux or Ubuntu), drivers are pre-installed. Skip to "Verify Drivers" below.

If you used **Ubuntu Server AMI** (not Deep Learning), follow these steps:

#### Install NVIDIA Drivers (Ubuntu Server AMI only)

Once connected to your instance via SSH:

**For Ubuntu:**
```bash
# Update system packages
sudo apt-get update
sudo apt-get upgrade -y

# Install NVIDIA driver (detects and installs appropriate version)
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall

# Or install specific version
sudo apt-get install -y nvidia-driver-535

# Reboot the instance to load the drivers
sudo reboot
```

**For Amazon Linux 2023** (if not using Deep Learning AMI):
```bash
# Update system packages
sudo yum update -y

# Install NVIDIA driver
sudo yum install -y nvidia-driver

# Reboot the instance to load the drivers
sudo reboot
```

#### Verify Drivers (All AMIs)

After connecting (and rebooting if needed):

```bash
nvidia-smi
```

You should see your GPU listed with driver information. 

**Note**: 
- **Deep Learning AMI** (Amazon Linux or Ubuntu): Drivers should work immediately
- **Amazon Linux 2023**: Uses `yum` package manager (not `apt-get`)
- **Ubuntu Server**: Uses `apt-get` package manager

#### Install CUDA Toolkit (if needed)

Some applications may require the CUDA toolkit:

```bash
# Install CUDA toolkit 11.8 (or 12.1 depending on your needs)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

Or use the package manager:

```bash
# For CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-8
```

#### Install Python Dependencies

Follow the [Installing Dependencies](#installing-dependencies) section below for Python, PyTorch, and vLLM installation.

### Step 7: Transfer Code to Instance

#### Option A: Clone Repository on Instance

```bash
# On the AWS instance
git clone https://github.com/esi2tekky/shawshank.git
cd shawshank
```

#### Option B: Use SCP to Transfer Files

```bash
# From your local machine
scp -i ~/.ssh/229project.pem -r ./shawshank ec2-user@YOUR_INSTANCE_IP:~/
```

#### Option C: Use AWS Systems Manager Session Manager

If you've configured IAM roles and Systems Manager:

```bash
# Start a session
aws ssm start-session --target i-YOUR_INSTANCE_ID

# Then use SCP or git clone within the session
```

### Step 8: Set Up Environment Variables

On the AWS instance, set up your environment:

```bash
# Set OpenAI API key (for judge)
export OPENAI_API_KEY="your-key-here"

# Login to Hugging Face
huggingface-cli login
```

### AWS Cost Considerations

- **g5.xlarge** (A10G 24GB): ~$1.00-1.20/hour (recommended for Tulu 3 8B)
- **p3.2xlarge** (V100 16GB): ~$3.06/hour (budget option, older GPU)
- **p4d.24xlarge** (A100 40GB): ~$32.77/hour (high performance)
- **Storage**: ~$0.10/GB/month for gp3 SSD
- **Data transfer**: First 100GB/month free, then ~$0.09/GB

**Cost-saving tips**:
- **Use Spot Instances** for up to 90% savings (you have G instance spot quota):
  - Select "Spot instance" in Launch Instance wizard
  - Or use: `aws ec2 request-spot-instances`
  - **Note**: Spot instances can be interrupted, but fine for batch experiments
- **Stop instances** when not in use (only pay for storage): 
  ```bash
  aws ec2 stop-instances --instance-ids i-YOUR_INSTANCE_ID
  ```
- **Terminate instances** when done (deletes instance and root volume):
  ```bash
  aws ec2 terminate-instances --instance-ids i-YOUR_INSTANCE_ID
  ```

**For RL Workloads**: Consider requesting on-demand G or P instance quotas (1-4 instances) to avoid interruption during long training runs, while using spot instances for shorter experiments.

### AWS Instance Management

```bash
# Start instance
aws ec2 start-instances --instance-ids i-YOUR_INSTANCE_ID

# Stop instance (preserves EBS volume)
aws ec2 stop-instances --instance-ids i-YOUR_INSTANCE_ID

# Terminate instance (deletes instance and root volume)
aws ec2 terminate-instances --instance-ids i-YOUR_INSTANCE_ID

# List running instances
aws ec2 describe-instances \
  --filters "Name=instance-state-name,Values=running" \
  --query "Reservations[*].Instances[*].[InstanceId,InstanceType,PublicIpAddress]" \
  --output table
```

---

## System Requirements

### Hardware Requirements

**Option 1: AWS Remote Instance (Recommended)**
- **GPU Instance**: 
  - **g5.xlarge** (NVIDIA A10G, 24GB VRAM) - **recommended** for Tulu 3 8B (you have spot quota)
  - **g5.2xlarge** (NVIDIA A10G, 48GB VRAM) - if you need more VRAM
  - **p3.2xlarge** (NVIDIA V100, 16GB VRAM) - for RL workloads if needed
  - **p4d.24xlarge** (NVIDIA A100, 40GB VRAM) - high performance for RL
- **Instance Type**: g5.xlarge or higher (4+ vCPUs, 16GB+ RAM)
- **Storage**: 200GB+ gp3 SSD EBS volume
- **Network**: Stable internet connection for model downloads
- **AMI**: Deep Learning AMI (Ubuntu) recommended (pre-configured with CUDA)
- **Purchase Option**: Spot instances recommended for cost savings (you have G instance spot quota)

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

**On AWS Instance or Local Machine:**

```bash
python3 --version  # Should be 3.8-3.11
```

If you don't have a virtual environment, create one:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Note**: 
- **Deep Learning AMI** (Amazon Linux or Ubuntu): Comes with Python 3.10+ and many ML libraries pre-installed
- **Ubuntu Server AMI**: Includes Python 3.10
- **Amazon Linux 2023**: Uses `yum` instead of `apt-get` for package management

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

**On AWS Instance:**
```bash
nvidia-smi
```

**On Local Machine:**
```bash
nvidia-smi
```

You should see output showing your GPU(s), driver version, and CUDA version. 

**AWS Note**: If using Deep Learning AMI, drivers are pre-installed. If using Ubuntu AMI, install drivers manually (see Step 6 in AWS Setup section). If `nvidia-smi` fails, you need to install the drivers first.

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

**On AWS Instance or Local Machine:**

Install the Hugging Face CLI:

```bash
pip install huggingface_hub
```

Login with your token:

```bash
huggingface-cli login
```

Enter your token when prompted. This will save your credentials for model downloads.

**Note**: On AWS instances, you'll need to login once per instance. Credentials are stored in `~/.cache/huggingface/`.

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
2. **On AWS**: Upgrade to a larger GPU instance (g5.xlarge A10G 24GB or p4d.24xlarge A100 40GB)
3. **On Local**: Close other GPU-intensive applications
4. Use a smaller model if available
5. Ensure you have sufficient VRAM (16GB+ recommended for Tulu 3 8B)
6. **On AWS**: Check instance GPU type with `nvidia-smi` - p3.2xlarge V100 may be insufficient for Tulu 3 8B

### Issue: "ModuleNotFoundError: No module named 'vllm'"

**Solutions**:
1. Ensure virtual environment is activated: `source venv/bin/activate`
2. Reinstall vLLM: `pip install --upgrade vllm`
3. Check Python version compatibility (vLLM requires Python 3.8-3.11)

### Issue: "CUDA not available" or "torch.cuda.is_available() returns False"

**Solutions**:
1. **On AWS**: Verify NVIDIA drivers: `nvidia-smi` - if it fails, install drivers using Step 6 in the AWS Setup section
2. **On AWS**: If using Ubuntu Server AMI (not Deep Learning AMI), install NVIDIA drivers manually:
   ```bash
   sudo apt-get update
   sudo apt-get install -y ubuntu-drivers-common
   sudo ubuntu-drivers autoinstall
   sudo reboot
   ```
3. **On AWS**: If using Amazon Linux 2023 (not Deep Learning AMI):
   ```bash
   sudo yum update -y
   sudo yum install -y nvidia-driver
   sudo reboot
   ```
4. **On AWS**: If using Deep Learning AMI (Amazon Linux or Ubuntu), drivers should be pre-installed. Try: `sudo reboot`
5. **On Local**: Verify NVIDIA drivers: `nvidia-smi`
6. Reinstall PyTorch with CUDA support (see [Installing Dependencies](#installing-dependencies))
7. Check CUDA version compatibility
8. **On AWS**: After installing drivers, restart the instance: `sudo reboot`
9. **On Local**: Restart your system after driver installation

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

**On AWS Instance or Local Machine:**

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

**AWS-Specific Commands:**

```bash
# Connect to instance
ssh -i ~/.ssh/229project.pem ec2-user@YOUR_INSTANCE_IP

# Start instance
aws ec2 start-instances --instance-ids i-YOUR_INSTANCE_ID --region us-west-2

# Stop instance (to save costs, preserves EBS volume)
aws ec2 stop-instances --instance-ids i-YOUR_INSTANCE_ID --region us-west-2

# Transfer files to instance
scp -i ~/.ssh/229project.pem file.txt ec2-user@YOUR_INSTANCE_IP:~/

# List running instances
aws ec2 describe-instances \
  --filters "Name=instance-state-name,Values=running" \
  --query "Reservations[*].Instances[*].[InstanceId,InstanceType,PublicIpAddress]" \
  --output table \
  --region us-west-2
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

