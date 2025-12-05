#!/bin/bash
set -e

# Script to run LLM attacker on all 4 open-source models
# Uses GPT-4o-mini as judge (cost-effective)

cd "$(dirname "$0")"

# Activate conda environment if available
if [ -d ~/miniconda3 ]; then
    source ~/miniconda3/bin/activate
    conda activate shawshank 2>/dev/null || true
fi

# Load environment variables
if [ -f .env.local ]; then
    set -a
    source .env.local
    set +a
fi

export PYTHONPATH="$(pwd)"

# Model keys and their full IDs
declare -A MODELS=(
    ["llama_base"]="meta-llama/Llama-3.1-8B"
    ["tulu_sft"]="allenai/Llama-3.1-Tulu-3-8B-SFT"
    ["tulu_dpo"]="allenai/Llama-3.1-Tulu-3-8B-DPO"
    ["tulu_rlvr"]="allenai/Llama-3.1-Tulu-3-8B"
)

MODEL_KEYS=("llama_base" "tulu_sft" "tulu_dpo" "tulu_rlvr")
INPUT_CSV="data/gpt_baseline.csv"  # Assuming this exists or will be generated
OUTPUT_DIR="results/llm_attacker"

echo "=========================================="
echo "LLM Attacker on All 4 Open-Source Models"
echo "=========================================="
echo "Input file: $INPUT_CSV"
echo "Judge: GPT-4o-mini"
echo "=========================================="
echo ""

# Verify input file exists
if [ ! -f "$INPUT_CSV" ]; then
    echo "ERROR: Input file not found: $INPUT_CSV"
    echo "Please generate prompts first using: python -m src.attacker.gpt_attacker"
    exit 1
fi

# GPU cleanup function
cleanup_gpu() {
    echo "Cleaning GPU memory..."
    # Kill any vLLM processes
    pkill -9 -f 'VLLM|EngineCore|vllm' 2>/dev/null || true
    # Kill any stuck Python processes
    pkill -9 -f 'python.*llm_attacker' 2>/dev/null || true
    # Clear GPU cache
    python3 -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()" 2>/dev/null || true
    sleep 5
    # Verify GPU is free
    if command -v nvidia-smi &> /dev/null; then
        FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        echo "Free GPU memory: ${FREE_MEM}MB"
    fi
}

# Run LLM attacker on each model
for model_key in "${MODEL_KEYS[@]}"; do
    model_id="${MODELS[$model_key]}"

    echo ""
    echo "=========================================="
    echo "Starting LLM attacker on: $model_key"
    echo "Model ID: $model_id"
    echo "Time: $(date)"
    echo "=========================================="

    # Aggressive GPU cleanup before starting
    cleanup_gpu

    # Run LLM attacker
    python3 -m src.experiments.run_llm_attacker \
        --input "$INPUT_CSV" \
        --target_model "$model_id" \
        --output "$OUTPUT_DIR/${model_key}_results.csv" \
        2>&1 | tee "llm_attacker_${model_key}.log"

    # Aggressive cleanup after completion
    cleanup_gpu

    echo ""
    echo "Completed: $model_key at $(date)"
    echo "=========================================="
done

echo ""
echo "=========================================="
echo "ALL LLM ATTACKER EXPERIMENTS COMPLETE"
echo "Time: $(date)"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Check individual log files: llm_attacker_*.log"

