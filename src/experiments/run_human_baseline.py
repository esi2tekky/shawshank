# src/experiments/run_human_baseline.py
import os, json, argparse, time, sys
from pathlib import Path
from tqdm import tqdm
# import local proxy and openai target classes
from src.target.local_proxy import LocalProxy
from src.judge.gpt4_judge import judge

TARGETS = {
    "local": LocalProxy,
    "openai": None  # filled below to avoid import error if openai not installed
}

def load_seeds(path):
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)

def run(seeds_path, out_path, target_name="local", openai_model="gpt-4"):
    # init target
    if target_name == "openai":
        from src.target.openai_target import OpenAITarget
        target = OpenAITarget(model=openai_model)
    else:
        target = LocalProxy()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(out_path, "w")
    seeds = list(load_seeds(seeds_path))
    for s in tqdm(seeds, desc="Running prompts"):
        #print(f"Running prompt {s.get('id')} ({s.get('category')}):", s['prompt'][:60], "...")

        prompt = s["prompt"]
        rec = {
            "id": s.get("id"),
            "category": s.get("category"),
            "prompt": prompt,
            "source": s.get("source"),
            "timestamp_utc": time.time()
        }
        try:
            resp = target.query(prompt)
            rec["response"] = resp["text"]
            rec["target_metadata"] = resp.get("metadata", {})
        except Exception as e:
            rec["response"] = None
            rec["error"] = str(e)
            print("Error while querying:", e)

        # run judge only if we have a response
        if rec.get("response"):
            try:
                j = judge(prompt, rec["response"])
            except Exception as e:
                j = {"decision": "error", "reason": str(e)}
            rec["judge"] = j

        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()

        # safety throttle: small sleep to avoid rate bursts
        time.sleep(0.5)

    out_f.close()
    print("Saved results to", out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="data/human_baseline.jsonl")
    p.add_argument("--out", default="results/human_baseline/run.jsonl")
    p.add_argument("--target", choices=["local","openai"], default="local")
    p.add_argument("--openai_model", default="gpt-4")
    args = p.parse_args()
    run(args.seeds, args.out, target_name=args.target, openai_model=args.openai_model)
