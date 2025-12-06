"""
src/experiments/run_llm_attacker.py
------------------------------------
Run LLM-generated adversarial prompts through target model and judge.

Takes input CSV from gpt_attacker.py (with columns: id, intent_category, intent_action, vague_action, category, prompt)
Adds columns: response, judge_label, judge_reason

Output: *_attack.csv (appends "_attack" to input filename)
"""

import os
import csv
import json
import argparse
from pathlib import Path
from tqdm import tqdm

from src.target.target_factory import load_target
from src.judge.gpt4_judge import judge
import gc
import torch


def run_llm_baseline(
    input_csv: str,
    output_csv: str = None,
    target_model: str = None,  # Can be model key (e.g., "tulu_sft") or full ID
    max_tokens: int = 300
):
    """
    Run LLM-generated prompts through target model and judge.

    Args:
        input_csv: Path to CSV from gpt_attacker.py (e.g., data/gpt_baseline.csv)
        output_csv: Output path (default: auto-generate from input_csv)
        target_model: Model identifier (e.g., "tulu_sft", "gpt-4o", or full HF path)
        max_tokens: Maximum tokens for target model response (default: 300)
    """
    input_path = Path(input_csv)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    if target_model is None:
        print("Error: --target_model is required")
        return

    # Auto-generate output filename if not provided
    if output_csv is None:
        # Use model name in output filename
        model_safe = target_model.replace("/", "_").replace(":", "_")
        output_csv = str(input_path.parent / f"{input_path.stem}_{model_safe}_results.csv")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize target using factory (supports both OpenAI and vLLM)
    print(f"Loading target model: {target_model}")
    target = load_target(target_model)

    # Read input CSV
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} prompts from {input_path}")

    # Prepare output CSV
    output_fieldnames = list(rows[0].keys()) + [
        "response",
        "judge_label",
        "judge_reason",
    ]

    with open(output_path, "w", newline='', encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=output_fieldnames)
        writer.writeheader()

        for row in tqdm(rows, desc="Testing prompts"):
            prompt_id = row.get("id", "unknown")
            prompt = row.get("prompt", "")

            if not prompt:
                print(f"⚠️ Skipping {prompt_id}: empty prompt")
                continue


            # Query target model
            try:
                resp = target.query(prompt)
                row["response"] = resp["text"]
            except Exception as e:
                row["response"] = None

            # Run judge if we have a response
            if row.get("response"):
                try:
                    judge_result = judge(prompt, row["response"])
                    row["judge_label"] = judge_result.get("decision", "error")
                    row["judge_reason"] = judge_result.get("reason", "")
                except Exception as e:
                    row["judge_label"] = "error"
                    row["judge_reason"] = str(e)
                    print(f"❌ Error judging {prompt_id}: {e}")
            else:
                row["judge_label"] = "error"
                row["judge_reason"] = "No response from target"

            # Write row to output
            writer.writerow(row)
            out_f.flush()


    print(f"✅ Results saved to {output_path}")

    # Quick stats
    with open(output_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        results = list(reader)

    total = len(results)
    noncompliant = sum(1 for r in results if r["judge_label"] == "noncompliant")
    compliant = sum(1 for r in results if r["judge_label"] == "compliant")
    errors = sum(1 for r in results if r["judge_label"] == "error")

    asr = (noncompliant / total * 100) if total > 0 else 0

    print(f"\n📊 Quick Stats:")
    print(f"   Total: {total}")
    print(f"   Noncompliant (successful attacks): {noncompliant} ({asr:.1f}%)")
    print(f"   Compliant (blocked): {compliant}")
    print(f"   Errors: {errors}")
    
    # Cleanup GPU memory if using vLLM
    if hasattr(target, 'llm'):
        print("Cleaning up GPU memory...")
        if hasattr(target, 'cleanup'):
            target.cleanup()
        else:
            del target.llm
            del target
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM-generated prompts through target model and judge"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV from gpt_attacker.py (e.g., data/gpt_baseline.csv)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: auto-generate with '_attack.csv' suffix)"
    )
    parser.add_argument(
        "--target_model",
        required=True,
        help="Target model identifier (e.g., 'tulu_sft', 'gpt-4o', or full HF path like 'allenai/Llama-3.1-Tulu-3-8B-SFT')"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=300,
        help="Maximum tokens for target model response (default: 300)"
    )

    args = parser.parse_args()

    run_llm_baseline(
        input_csv=args.input,
        output_csv=args.output,
        target_model=args.target_model,
        max_tokens=args.max_tokens
    )
