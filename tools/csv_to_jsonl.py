#!/usr/bin/env python3
import csv, json, argparse, sys
from pathlib import Path

def gen_id(n): return f"h{n:03d}"

def convert(infile, outfile):
    infile = Path(infile)
    outfile = Path(outfile)
    if not infile.exists():
        print("❌ Input file not found:", infile)
        sys.exit(1)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    print("🔍 Reading from", infile)
    with infile.open(newline='', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        print("🧾 Detected columns:", reader.fieldnames)
        if not reader.fieldnames:
            print("❌ CSV appears empty or not comma-separated. Check delimiter.")
            sys.exit(1)
        if "prompt" not in [h.lower() for h in reader.fieldnames]:
            print("⚠️ No 'prompt' column found — please rename your header to 'prompt'.")
            sys.exit(1)

        with outfile.open("w", encoding="utf-8") as f_out:
            count = 0
            for i, row in enumerate(reader, 1):
                if not any(row.values()):
                    continue
                # normalize keys
                norm = {k.lower().strip(): (v.strip() if isinstance(v, str) else v)
                        for k, v in row.items()}
                if not norm.get("prompt"):
                    print(f"⚠️ Skipping line {i}: empty prompt")
                    continue
                obj = {
                    "id": norm.get("id", gen_id(i)),
                    "category": norm.get("category", "unspecified"),
                    "prompt": norm["prompt"],
                    "source": norm.get("source", "human"),
                    "author": norm.get("author", ""),
                    "notes": norm.get("notes", "")
                }
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1
                if count <= 3:
                    print("✅ Sample output line:", obj)
            print(f"✅ Wrote {count} records to {outfile}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", dest="outfile", required=True)
    args = p.parse_args()
    convert(args.infile, args.outfile)
