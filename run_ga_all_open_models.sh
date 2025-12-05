#!/bin/bash
set -e

# Script to run Genetic Algorithm attacks on all 4 open-source models
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
SEEDS_FILE="data/human_baseline.csv"
GENERATIONS=10
OUTPUT_DIR="results/ga_attacks"

echo "=========================================="
echo "GA Attack on All 4 Open-Source Models"
echo "=========================================="
echo "Seed file: $SEEDS_FILE"
echo "Generations: $GENERATIONS"
echo "Judge: GPT-4o-mini"
echo "Expected cost: ~\$0.78"
echo "=========================================="
echo ""

# Verify seed file exists
if [ ! -f "$SEEDS_FILE" ]; then
    echo "ERROR: Seed file not found: $SEEDS_FILE"
    exit 1
fi

# GPU cleanup function
cleanup_gpu() {
    echo "Cleaning GPU memory..."
    # Kill any vLLM processes
    pkill -9 -f 'VLLM|EngineCore|vllm' 2>/dev/null || true
    # Kill any stuck Python processes
    pkill -9 -f 'python.*ga_attacker' 2>/dev/null || true
    # Clear GPU cache
    python3 -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()" 2>/dev/null || true
    sleep 5
    # Verify GPU is free
    if command -v nvidia-smi &> /dev/null; then
        FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        echo "Free GPU memory: ${FREE_MEM}MB"
    fi
}

# Run GA on each model
for model_key in "${MODEL_KEYS[@]}"; do
    model_id="${MODELS[$model_key]}"

    echo ""
    echo "=========================================="
    echo "Starting GA attack on: $model_key"
    echo "Model ID: $model_id"
    echo "Time: $(date)"
    echo "=========================================="

    # Aggressive GPU cleanup before starting
    cleanup_gpu

    # Run GA attack (ensure env vars are loaded)
    source ~/miniconda3/bin/activate
    conda activate shawshank 2>/dev/null || true
    if [ -f .env.local ]; then
        set -a
        source .env.local
        set +a
    fi
    export PYTHONPATH=~/shawshank
    
    python3 -m src.experiments.run_ga_attacker \
        --target "$model_id" \
        --seeds "$SEEDS_FILE" \
        --generations "$GENERATIONS" \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee "ga_${model_key}.log"

    # Aggressive cleanup after completion
    cleanup_gpu

    echo ""
    echo "Completed: $model_key at $(date)"
    echo "=========================================="
done

echo ""
echo "=========================================="
echo "ALL GA ATTACKS COMPLETE"
echo "Time: $(date)"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Check individual log files: ga_*.log"

