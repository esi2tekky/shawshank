#!/usr/bin/env python3
"""Quick setup verification script for vLLM and GPU access."""

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
    else:
        print("WARNING: CUDA not available. vLLM requires GPU support.")
except ImportError:
    print("ERROR: PyTorch not installed")

# vLLM
try:
    import vllm
    try:
        print(f"\nvLLM version: {vllm.__version__}")
    except AttributeError:
        print("\nvLLM installed (version attribute not available)")
except ImportError:
    print("\nERROR: vLLM not installed")

# Hugging Face
try:
    from huggingface_hub import whoami
    user = whoami()
    print(f"\nHugging Face: Logged in as {user.get('name', 'unknown')}")
except Exception as e:
    print(f"\nHugging Face: Not logged in or error ({e})")
    print("  Run: huggingface-cli login")

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
print("Setup check complete!")
print("=" * 60)

