"""
src/experiments/run_human_baseline_multi.py
-------------------------------------------
Run human baseline experiments across multiple models (all three Tulu post-training versions).

Takes input JSONL from data/human_baseline.jsonl and runs it against:
- tulu_sft (SFT version)
- tulu_dpo (DPO version)  
- tulu_rlvr (RLVR version)

Uses target_factory to load models and handles GPU memory cleanup between runs.
"""

import os
import json
import argparse
import time
import gc
from pathlib import Path
from tqdm import tqdm

from src.config import OPEN_SOURCE_MODELS
from src.target.target_factory import load_target
from src.judge.gpt4_judge import judge


def load_seeds(path):
    """Load prompts from JSONL file."""
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def run_single_model(seeds_path, model_key, model_id, output_dir):
    """
    Run human baseline against a single model.
    
    Args:
        seeds_path: Path to input JSONL file with prompts
        model_key: Short key for the model (e.g., 'tulu_sft')
        model_id: Full model identifier (e.g., 'allenai/Llama-3.1-Tulu-3-8B-SFT')
        output_dir: Directory to save output files
    """
    print(f"\n{'='*60}")
    print(f"Running human baseline on: {model_key}")
    print(f"Model ID: {model_id}")
    print(f"{'='*60}\n")
    
    # Initialize target using factory
    target = None
    try:
        target = load_target(model_id)
    except Exception as e:
        print(f"ERROR: Failed to load model {model_key}: {e}")
        # Clean up any partial GPU memory allocation
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        return None
    
    # Prepare output file
    output_path = Path(output_dir) / f"human_baseline_{model_key}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load all seeds
    seeds = list(load_seeds(seeds_path))
    print(f"Loaded {len(seeds)} prompts from {seeds_path}")
    
    # Process each prompt
    results = []
    for seed in tqdm(seeds, desc=f"Processing {model_key}"):
        prompt = seed.get("prompt", "")
        if not prompt:
            continue
            
        rec = {
            "id": seed.get("id"),
            "category": seed.get("category"),
            "prompt": prompt,
            "source": seed.get("source", "human"),
            "model_key": model_key,
            "model_id": model_id,
            "timestamp_utc": time.time()
        }
        
        # Query target model
        try:
            resp = target.query(prompt)
            rec["response"] = resp["text"]
            rec["target_metadata"] = resp.get("metadata", {})
        except Exception as e:
            rec["response"] = None
            rec["error"] = str(e)
            print(f"Error querying {model_key} for prompt {rec['id']}: {e}")
        
        # Run judge evaluation if we have a response
        if rec.get("response"):
            try:
                j = judge(prompt, rec["response"])
                rec["judge"] = j
            except Exception as e:
                rec["judge"] = {"decision": "error", "reason": str(e)}
                print(f"Error judging {model_key} for prompt {rec['id']}: {e}")
        else:
            rec["judge"] = {"decision": "error", "reason": "No response from target"}
        
        results.append(rec)
    
    # Write results to file
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print(f"\nSaved {len(results)} results to {output_path}")
    
    # Clean up GPU memory if this is a vLLM model
    if hasattr(target, 'llm'):
        print(f"Cleaning up GPU memory for {model_key}...")
        del target.llm
        del target
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
            print("GPU memory cleared.")
        except:
            pass
    
    return output_path


def run_all_tulu_models(seeds_path, output_dir, model_keys=None):
    """
    Run human baseline across all specified Tulu models.
    
    Args:
        seeds_path: Path to input JSONL file
        output_dir: Directory to save output files
        model_keys: List of model keys to run (default: all three Tulu models)
    """
    if model_keys is None:
        # Default to all three Tulu post-training versions
        model_keys = ["tulu_sft", "tulu_dpo", "tulu_rlvr"]
    
    print(f"Running human baseline on {len(model_keys)} models:")
    for key in model_keys:
        print(f"  - {key}: {OPEN_SOURCE_MODELS.get(key, 'NOT FOUND')}")
    
    results_summary = {}
    
    for model_key in model_keys:
        if model_key not in OPEN_SOURCE_MODELS:
            print(f"WARNING: {model_key} not found in OPEN_SOURCE_MODELS, skipping...")
            continue
        
        model_id = OPEN_SOURCE_MODELS[model_key]
        
        try:
            # Ensure GPU is free before loading next model
            import torch
            import gc
            torch.cuda.empty_cache()
            gc.collect()
            
            output_path = run_single_model(seeds_path, model_key, model_id, output_dir)
            if output_path:
                results_summary[model_key] = str(output_path)
        except Exception as e:
            print(f"ERROR: Failed to run {model_key}: {e}")
            results_summary[model_key] = f"ERROR: {e}"
            # Clean up on error
            try:
                import torch
                import gc
                torch.cuda.empty_cache()
                gc.collect()
            except:
                pass
        
        # Delay between models to ensure cleanup completes
        time.sleep(5)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for model_key, result in results_summary.items():
        status = "SUCCESS" if "ERROR" not in str(result) else "FAILED"
        print(f"{model_key}: {status}")
        if "ERROR" not in str(result):
            print(f"  Output: {result}")
    print(f"{'='*60}\n")
    
    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run human baseline across multiple Tulu models"
    )
    parser.add_argument(
        "--seeds",
        default="data/human_baseline.jsonl",
        help="Path to input JSONL file with human-crafted prompts"
    )
    parser.add_argument(
        "--output_dir",
        default="results/human_baseline",
        help="Directory to save output JSONL files"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tulu_sft", "tulu_dpo", "tulu_rlvr"],
        choices=["tulu_sft", "tulu_dpo", "tulu_rlvr", "llama_base"],
        help="Which models to run (default: all three Tulu models)"
    )
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not Path(args.seeds).exists():
        print(f"ERROR: Input file not found: {args.seeds}")
        exit(1)
    
    run_all_tulu_models(args.seeds, args.output_dir, args.models)

