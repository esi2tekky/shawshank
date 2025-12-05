"""
src/experiments/run_ga_all_models.py
------------------------------------
Run Structured GA across all models defined in config.py.

Iterates through: llama_base → tulu_sft → tulu_dpo → tulu_rlvr
Cleans GPU memory between models.

Usage:
    # Run all open source models
    python -m src.experiments.run_ga_all_models --generations 10

    # Run specific models only
    python -m src.experiments.run_ga_all_models --models tulu_sft tulu_dpo --generations 10

    # Include GPT-4o comparison
    python -m src.experiments.run_ga_all_models --models tulu_sft tulu_dpo gpt_4o --generations 10
"""

import gc
import json
import argparse
from pathlib import Path
from datetime import datetime

from src.config import OPEN_SOURCE_MODELS, CLOSED_SOURCE_MODELS, ALL_MODELS
from src.target.target_factory import load_target
from src.judge.gpt4_judge_continuous import judge_continuous
from src.experiments.run_structured_ga import run_structured_ga_experiment


def cleanup_gpu():
    """Clean up GPU memory between model runs."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("  GPU memory cleared.")
    except ImportError:
        pass


def run_ga_on_model(
    model_key: str,
    model_path: str,
    output_base: Path,
    timestamp: str,
) -> dict:
    """Run GA experiment on a single model."""

    print(f"\n{'='*60}")
    print(f"MODEL: {model_key}")
    print(f"PATH: {model_path}")
    print(f"{'='*60}")

    # Create model-specific output directory
    model_output = output_base / f"{model_key}_{timestamp}"
    model_output.mkdir(parents=True, exist_ok=True)

    # Load target
    try:
        target = load_target(model_path)
    except Exception as e:
        print(f"ERROR: Failed to load {model_key}: {e}")
        return {"model": model_key, "status": "failed", "error": str(e)}

    # Run GA experiment
    # Equivalent to: python -m src.experiments.run_structured_ga --target <model> --population 20 --generations 10 --trials-per-gen 10
    start_time = datetime.now()
    try:
        ga = run_structured_ga_experiment(
            target=target,
            judge_fn=judge_continuous,
            population_size=20,
            generations=10,
            trials_per_gen=10,
            output_dir=str(model_output),
        )

        # Extract results from returned dict
        # ga returns: {"asr": float, "best_ever": dict, "final_strategies": list}
        final_strategies = ga.get("final_strategies", [])
        asr = ga.get("asr", 0)
        best_ever = ga.get("best_ever", {})

        result = {
            "model": model_key,
            "model_path": model_path,
            "status": "success",
            "population": 20,
            "generations": 10,
            "trials_per_gen": 10,
            "duration_seconds": (datetime.now() - start_time).total_seconds(),
            "asr": asr,
            "best_ever": best_ever,
            "top_strategy": final_strategies[0] if final_strategies else None,
            "final_strategies": final_strategies,
            "output_dir": str(model_output),
        }

    except Exception as e:
        result = {
            "model": model_key,
            "status": "error",
            "error": str(e),
            "duration_seconds": (datetime.now() - start_time).total_seconds(),
        }
        print(f"ERROR during GA: {e}")

    # Clean up GPU memory
    print("\n  Cleaning up GPU memory...")
    if hasattr(target, 'llm'):
        del target.llm
    del target
    cleanup_gpu()

    return result


def run_all_models(
    model_keys: list,
    output_dir: str = "results/model_comparison",
):
    """Run GA experiments across multiple models.

    Each model runs: python -m src.experiments.run_structured_ga --target <model> --population 20 --generations 10 --trials-per-gen 10
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("MULTI-MODEL GA EXPERIMENT")
    print(f"{'='*60}")
    print(f"Timestamp: {timestamp}")
    print(f"Models: {model_keys}")
    print(f"GA params: population=20, generations=10, trials_per_gen=10")
    print(f"Output: {output_base}")
    print(f"{'='*60}")

    # Validate model keys
    valid_keys = []
    for key in model_keys:
        if key in ALL_MODELS:
            valid_keys.append(key)
        else:
            print(f"WARNING: Unknown model key '{key}', skipping.")

    if not valid_keys:
        print("ERROR: No valid model keys provided.")
        return

    # Run each model
    all_results = []
    for i, model_key in enumerate(valid_keys, 1):
        print(f"\n[{i}/{len(valid_keys)}] Processing {model_key}...")
        model_path = ALL_MODELS[model_key]

        result = run_ga_on_model(
            model_key=model_key,
            model_path=model_path,
            output_base=output_base,
            timestamp=timestamp,
        )
        all_results.append(result)

        # Save intermediate results
        with open(output_base / f"progress_{timestamp}.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # === FINAL COMPARISON REPORT ===
    print(f"\n{'='*60}")
    print("COMPARISON REPORT")
    print(f"{'='*60}")

    # Table header
    print(f"\n{'Model':<15} {'Status':<10} {'Top Strategy':<20} {'ASR':<10} {'Duration':<10}")
    print("-" * 70)

    for r in all_results:
        model = r["model"]
        status = r["status"]

        if status == "success":
            top_strat = r["top_strategy"]["strategy"] if r.get("top_strategy") else "N/A"
            asr = f"{r['asr']:.1%}"
            duration = f"{r['duration_seconds']:.0f}s"
        else:
            top_strat = "N/A"
            asr = "N/A"
            duration = "N/A"

        print(f"{model:<15} {status:<10} {top_strat:<20} {asr:<10} {duration:<10}")

    # Strategy comparison across models
    print(f"\n{'='*60}")
    print("STRATEGY EFFECTIVENESS BY MODEL")
    print(f"{'='*60}")

    # Collect all strategies
    all_strategies = set()
    for r in all_results:
        if r["status"] == "success" and r.get("final_strategies"):
            for s in r["final_strategies"]:
                all_strategies.add(s["strategy"])

    # Print strategy comparison
    if all_strategies:
        print(f"\n{'Strategy':<20}", end="")
        for r in all_results:
            if r["status"] == "success":
                print(f"{r['model']:<12}", end="")
        print()
        print("-" * (20 + 12 * sum(1 for r in all_results if r["status"] == "success")))

        for strat in sorted(all_strategies):
            print(f"{strat:<20}", end="")
            for r in all_results:
                if r["status"] == "success":
                    strat_data = next(
                        (s for s in r.get("final_strategies", []) if s["strategy"] == strat),
                        None
                    )
                    if strat_data:
                        print(f"{strat_data['success_rate']:.1%}".ljust(12), end="")
                    else:
                        print("-".ljust(12), end="")
            print()

    # Save final comparison
    comparison_file = output_base / f"comparison_{timestamp}.json"
    with open(comparison_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "models_tested": valid_keys,
            "ga_params": {"population": 20, "generations": 10, "trials_per_gen": 10},
            "results": all_results,
        }, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_base}")
    print(f"Comparison file: {comparison_file}")
    print(f"{'='*60}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Structured GA across multiple models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama_base", "tulu_sft", "tulu_dpo", "tulu_rlvr"],
        help="Model keys from config.py (default: all open source models)"
    )
    parser.add_argument(
        "--output",
        default="results/model_comparison",
        help="Output directory for results"
    )

    args = parser.parse_args()

    print("\nAvailable models:")
    print("  Open Source:", list(OPEN_SOURCE_MODELS.keys()))
    print("  Closed Source:", list(CLOSED_SOURCE_MODELS.keys()))
    print("\nGA params (hardcoded): --population 20 --generations 10 --trials-per-gen 10")

    run_all_models(
        model_keys=args.models,
        output_dir=args.output,
    )
