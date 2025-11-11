# shawshank (jailbreak lab)

Skeleton project for LLM jailbreak experiments.

See src/experiments/run_llm_attacker.py for an end-to-end demo.

Populate .env with OPENAI_API_KEY before running.

What this repo does (tl;dr)

Uses an LLM (attacker agent) to generate candidate prompts aimed at eliciting a target behavior from a model.

Sends candidates to a target model (local stub or real API) and captures responses.

Uses GPT-4 as a judge to classify responses as compliant (unsafe/disallowed) or noncompliant (safe/refusal).

Saves structured run logs for analysis (ASR, diversity, transfer, etc.).

NOTE: This project is intended for research. DO NOT publish or expose actual harmful model outputs. Follow the ethical guidelines in the Labeling & Safety section below.

Repo layout (important files)
.
├─ Dockerfile
├─ requirements.txt
├─ .env.example
├─ src/
│  ├─ attacker/llm_attacker.py         # LLM attacker wrapper
│  ├─ judge/gpt4_judge.py              # GPT-4 judge wrapper
│  ├─ target/local_proxy.py            # local target stub (swap for real wrapper)
│  └─ experiments/run_llm_attacker.py  # example end-to-end runner
├─ data/seeds/human_baseline.jsonl
├─ results/                            # gitignored; run outputs go here
└─ README.md

Quickstart — Local (no Docker)

Clone repo:

git clone https://github.com/esi2tekky/shawshank.git
cd shawshank


Create and activate venv, install deps:

python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt


Create a local .env (do NOT commit):

cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-... (your real key, kept local)


Run the demo:

python src/experiments/run_llm_attacker.py


Outputs are saved to ./results/llm_attacker_run.json.

Quickstart — Docker

(Use if you want a reproducible container across machines.)

Build image:

docker build -t shawshank:dev .


Create .env locally from example (do NOT commit):

cp .env.example .env
# edit .env to add OPENAI_API_KEY and other values


Run interactive container (mount repo and load env):

docker run -it --env-file .env -v $(pwd):/app shawshank:dev bash
# inside container:
python src/experiments/run_llm_attacker.py

Config / Secrets

.env.example contains placeholders. Never commit real keys.

Use local .env for development (listed in .gitignore).

For CI or cloud runs: store keys in GitHub Secrets or GCP Secret Manager and inject them at runtime.

Important vars in .env:

OPENAI_API_KEY=sk-...
TARGET_MODEL_URL=http://localhost:8080      # optional, for remote proxies
RESULTS_DIR=./results

How the demo works (high level)

llm_attacker.generate_candidates(goal) — asks GPT-4 to propose candidate user prompts for a specified goal.

Each candidate is sent to the target via target.query(prompt). The default LocalProxy echoes back a stub response; swap for a real model wrapper to test GPT-4 / Gemini / open-source targets.

gpt4_judge.judge(user_prompt, model_response) classifies the pair and returns a JSON decision and short reason.

All raw inputs, target outputs, judge results and metadata (timestamps, token counts if available) are saved to results/ for later analysis.

Running closed-API experiments (cost & safety)

GPT-4 is used both as attacker generator and judge in the demo. Both incur API costs.

Start with tiny pilot runs (5–10 candidates) to estimate tokens/cost. Log and monitor usage.

Add a token-usage logger (check resp['usage'] from OpenAI responses) and enforce a MAX_QUERIES budget in scripts.

NEVER store or publish disallowed/harmful outputs. If you must analyze content, store only labels, embeddings, and redacted excerpts.

Labeling & evaluation (recommended academic protocol)

To make baseline comparisons rigorous and defensible:

Label schema: final_label ∈ {compliant, noncompliant}; optional severity and category.

Annotators: at least 2 independent annotators blind to prompt source. Use a 3rd adjudicator if annotators disagree.

Agreement: report Cohen’s kappa or Krippendorff’s alpha. If < 0.6, refine guidelines.

Sampling: for large runs, randomly sample successes and failures for manual annotation.

Retention: avoid storing harmful generation outputs; keep labels, metadata, and embeddings only.

Extending the skeleton

Replace LocalProxy with wrappers for:

OpenAI GPT-4 / GPT-5 (via official API)

Google Gemini (if available in your org)

HF-hosted open-source models (via transformers or Text-Generation-Inference)

Add more attacker methods:

GA-based fuzzer (ga_baseline.py)

RL-based mutator policy

Hybrid LLM→GA pipelines

Instrument metrics: ASR, queries-to-first-success, diversity (embedding clusters), transfer rates across models.

Safety & ethics (must-read)

This work deals with discovering model failures. Work responsibly:

Keep all harmful outputs restricted to the project team.

Mask/remove operational details that could facilitate real-world harm before sharing results.

Consult your institution’s IRB / legal / ethics guidelines if you plan to publish.

Prefer open-source local proxies for heavy iterative experiments to save API costs and reduce sensitive calls to closed APIs.

Troubleshooting

docker: command not found → install Docker or run with venv.

openai authentication error → ensure OPENAI_API_KEY is set in .env and loaded into the environment.

Judge parse errors (judge expects JSON) → check judge prompt and add robust parsing/fallbacks.

Useful commands
# build (docker)
docker build -t shawshank:dev .

# run demo locally (venv)
python src/experiments/run_llm_attacker.py

# view latest results
jq . results/llm_attacker_run.json | less
