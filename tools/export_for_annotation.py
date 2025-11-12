import json, csv, argparse
from pathlib import Path

def jsonl_to_csv(jsonl_path, csv_path):
    jsonl_path, csv_path = Path(jsonl_path), Path(csv_path)
    if not jsonl_path.exists():
        print(f"❌ Input file not found: {jsonl_path}")
        return
    lines = list(jsonl_path.open())
    print(f"🔍 Found {len(lines)} lines in {jsonl_path}")

    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception as e:
            print(f"⚠️ JSON parse error at line {i+1}: {e}")
            continue
        rows.append({
            "id": obj.get("id"),
            "category": obj.get("category"),
            "prompt": obj.get("prompt"),
            "response": obj.get("response"),
            "judge_label": obj.get("judge", {}).get("decision"),
            "judge_reason": obj.get("judge", {}).get("reason")
        })
    print(f"✅ Parsed {len(rows)} valid records")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline='', encoding="utf-8") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=[
                "id","category","prompt","response",
                "judge_label","judge_reason"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Wrote {len(rows)} rows to {csv_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", dest="outfile", required=True)
    args = p.parse_args()
    jsonl_to_csv(args.infile, args.outfile)
