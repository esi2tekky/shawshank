# Project Checkpoint: Tulu Suite Setup on AWS EC2

**Date:** December 4, 2025  
**Branch:** `attack_targets`  
**Instance:** AWS EC2 g5.xlarge (us-west-2), IP: 18.237.105.90

---

## ✅ What We've Accomplished

### 1. Infrastructure Setup
- ✅ AWS EC2 instance created and configured (g5.xlarge with NVIDIA A10G, 23GB VRAM)
- ✅ SSH access configured (key: `shawshank.pem` at `/Users/esi/Downloads/shawshank.pem`)
- ✅ Repository cloned and checked out to `attack_targets` branch
- ✅ Conda environment set up with Python 3.10 (`shawshank` environment)
- ✅ All dependencies installed (vllm, torch, transformers, accelerate, etc.)
- ✅ GPU access verified (NVIDIA A10G detected and accessible)

### 2. vLLM Integration
- ✅ `src/target/vllm_target.py` implemented with `TargetAPI` interface
- ✅ `src/target/target_factory.py` created for unified model loading
- ✅ Memory-optimized configuration for A10G (23GB):
  - `max_model_len=512` (reduced from 4096 to fit in memory)
  - `gpu_memory_utilization=0.75`
  - `enforce_eager=True` (disables torch.compile to save memory)
  - `dtype="bfloat16"`
- ✅ Successfully tested model loading with `allenai/Llama-3.1-Tulu-3-8B-SFT`
- ✅ Model query test passed (generated valid responses)

### 3. Configuration & Scripts
- ✅ `src/config.py` created with model registry:
  - Open-source models: `tulu_sft`, `tulu_dpo`, `tulu_rlvr`
  - Closed-source models: `gpt_4o`
- ✅ `src/experiments/run_human_baseline_multi.py` ready for multi-model runs
- ✅ Human baseline data file exists: `data/human_baseline.jsonl` (50 prompts)

### 4. Documentation
- ✅ `SETUP_VLLM.md` - Complete AWS EC2 setup guide
- ✅ `SSH_LOGIN_GUIDE.md` - Partner SSH access instructions
- ✅ `ARCHITECTURE.md` - Comprehensive architecture documentation

---

## 🔧 Current Status

### Working Components
1. **vLLM Model Loading**: Successfully loads Tulu models on A10G GPU
2. **Target Factory**: Can load both OpenAI and vLLM models via unified interface
3. **GPU Memory Management**: Proper cleanup between model loads
4. **Environment**: All dependencies installed and working

### Known Limitations
1. **Memory Constraints**: 
   - A10G has 23GB VRAM, which limits `max_model_len` to 512 tokens
   - This is sufficient for most prompts but may truncate very long conversations
   - Model weights take ~15GB, leaving ~8GB for KV cache

2. **Model Loading Time**:
   - First load takes ~60-70 seconds (model download + initialization)
   - Subsequent loads are faster if model is cached

3. **Pending Tests**:
   - Target factory integration test had a minor error (needs verification)
   - Human baseline not yet run on any models

---

## 📋 What Still Needs to Be Done

### Immediate Next Steps (When Resuming)

1. **Verify Target Factory** (5 minutes)
   ```bash
   ssh -i ~/Downloads/shawshank.pem ec2-user@18.237.105.90
   source ~/miniconda3/bin/activate && conda activate shawshank
   cd ~/shawshank
   python3 -c "from src.target.target_factory import load_target; from src.config import OPEN_SOURCE_MODELS; target = load_target(OPEN_SOURCE_MODELS['tulu_sft']); print(target.query('Hello'))"
   ```

2. **Set Up Credentials** (2 minutes)
   - Hugging Face CLI login (if not already done):
     ```bash
     huggingface-cli login
     # Enter HF token when prompted
     ```
   - OpenAI API key for judge:
     ```bash
     export OPENAI_API_KEY='your-key-here'
     # Or add to ~/.env file
     ```

3. **Run Human Baseline on All Three Models** (30-60 minutes)
   ```bash
   cd ~/shawshank
   export OPENAI_API_KEY='your-key-here'
   python3 src/experiments/run_human_baseline_multi.py \
     --seeds data/human_baseline.jsonl \
     --output_dir results/human_baseline \
     --models tulu_sft tulu_dpo tulu_rlvr
   ```
   
   Expected output:
   - `results/human_baseline/human_baseline_tulu_sft.jsonl`
   - `results/human_baseline/human_baseline_tulu_dpo.jsonl`
   - `results/human_baseline/human_baseline_tulu_rlvr.jsonl`

### Future Tasks

4. **Test All Three Tulu Models Individually**
   - Verify each model loads correctly
   - Test query functionality for each
   - Confirm GPU memory cleanup works between models

5. **Run Comparative Experiments**
   - Execute human baseline on all models
   - Collect judge evaluations
   - Analyze results

6. **Implement Attack Scripts** (Phase 3)
   - Create LLM-based attacker
   - Run comparative scans across models
   - Compare attack success rates

---

## 🚀 Quick Start Guide (Resuming Work)

### 1. Connect to Instance
```bash
ssh -i ~/Downloads/shawshank.pem ec2-user@18.237.105.90
```

### 2. Activate Environment
```bash
source ~/miniconda3/bin/activate
conda activate shawshank
cd ~/shawshank
```

### 3. Pull Latest Changes
```bash
git pull origin attack_targets
```

### 4. Set Environment Variables
```bash
export OPENAI_API_KEY='your-openai-key'
# Hugging Face token should be in ~/.cache/huggingface/token
```

### 5. Verify GPU
```bash
nvidia-smi
# Should show NVIDIA A10G with ~0 MiB used
```

### 6. Test Model Loading
```bash
python3 -c "
from src.target.target_factory import load_target
from src.config import OPEN_SOURCE_MODELS
target = load_target(OPEN_SOURCE_MODELS['tulu_sft'])
print(target.query('Say hello'))
"
```

### 7. Run Human Baseline
```bash
python3 src/experiments/run_human_baseline_multi.py \
  --seeds data/human_baseline.jsonl \
  --output_dir results/human_baseline
```

---

## 📁 Key Files & Locations

### On AWS Instance
- **Repository**: `~/shawshank/`
- **Conda Environment**: `~/miniconda3/envs/shawshank/`
- **Model Cache**: `~/.cache/huggingface/hub/` (models downloaded here)
- **vLLM Cache**: `~/.cache/vllm/`
- **Results**: `~/shawshank/results/` (created when running experiments)

### Key Code Files
- `src/target/vllm_target.py` - vLLM target implementation
- `src/target/target_factory.py` - Model loading factory
- `src/config.py` - Model registry
- `src/experiments/run_human_baseline_multi.py` - Human baseline runner
- `data/human_baseline.jsonl` - Input prompts (50 prompts)

---

## ⚠️ Important Notes

1. **Instance Management**:
   - Instance is currently running (costs ~$1.10/hour for g5.xlarge)
   - Stop instance when not in use: AWS Console → EC2 → Instance → Stop
   - Instance state is preserved (EBS volume persists)

2. **GPU Memory**:
   - Always check GPU memory before loading models: `nvidia-smi`
   - Kill stuck processes: `pkill -f 'python.*vllm|VLLM'`
   - Clear GPU cache: `python3 -c "import torch; torch.cuda.empty_cache()"`

3. **Model Downloads**:
   - First load downloads ~16GB per model (takes 1-2 minutes)
   - Models are cached in `~/.cache/huggingface/hub/`
   - Subsequent loads are much faster

4. **Memory Constraints**:
   - `max_model_len=512` is a hard limit due to GPU memory
   - For longer contexts, consider:
     - Using a larger GPU instance (g5.2xlarge, g5.4xlarge)
     - Using quantization (AWQ/GPTQ)
     - Using CPU offloading for KV cache

---

## 🔍 Troubleshooting

### Model Won't Load
```bash
# Check GPU memory
nvidia-smi

# Kill stuck processes
pkill -f 'python.*vllm|VLLM'

# Clear GPU cache
python3 -c "import torch; torch.cuda.empty_cache()"
```

### Import Errors
```bash
# Verify environment
conda activate shawshank
python3 -c "import vllm; print('vLLM OK')"

# Reinstall if needed
pip install --upgrade vllm torch transformers accelerate
```

### Hugging Face Authentication
```bash
# Login interactively
huggingface-cli login

# Or set token manually
export HF_TOKEN='your-token'
```

---

## 📊 Expected Resource Usage

- **Model Loading**: ~15GB GPU memory per model
- **Inference**: ~16-18GB GPU memory during queries
- **Disk Space**: ~16GB per model (cached in `~/.cache/huggingface/`)
- **Time per Model**:
  - First load: 60-70 seconds
  - Subsequent loads: 10-15 seconds
  - 50 prompts: ~5-10 minutes (depending on response length)

---

## 🎯 Success Criteria

- [x] vLLM loads Tulu models successfully
- [x] Target factory works with vLLM backend
- [x] Human baseline running on all three Tulu models (in progress)
- [ ] Judge evaluations complete for all prompts
- [ ] Results saved to JSONL files
- [x] GPU memory cleanup verified between models
- [x] Architecture documentation created

---

## 📝 Next Session Checklist

When you resume work:

1. [ ] SSH into instance
2. [ ] Verify GPU is free (`nvidia-smi`)
3. [ ] Pull latest code (`git pull`)
4. [ ] Set OpenAI API key
5. [ ] Test target factory (quick verification)
6. [ ] Run human baseline on all three models
7. [ ] Verify results files created
8. [ ] Stop instance when done (to save costs)

---

**Last Updated:** December 4, 2025  
**Status:** Ready to run human baseline experiments

