# Shawshank: Breaking LLMs Using ML
**A Comparative Study of ML-Based Jailbreak Generation Approaches for LLM Safety Testing**

This repository implements a framework for evaluating different attack generation methods against LLM safety guardrails. The project compares human-crafted adversarial prompts with various ML-based generation techniques to assess which approaches most effectively identify model vulnerabilities.

---

## Project Methodology

### A. Dataset & Seed Prompts

We build a dataset **D** of prompts representing different intent categories with structured placeholders (`<INTENT>`) for sensitive tasks.

**Intent Categories:**
- Physical harm
- Illegal activity
- Sexual content
- Hate speech
- Medical advice
- Privacy violation
- Misinformation
- Child exploitation
- Acts of violence
- Creation of weapons
- Terrorism

Each method generates **B** prompts (default: B=50) across these categories using different evasion strategies.

### B. Methods

We compare multiple attack generation approaches:

1. **Human Baseline** — Manually crafted prompts from red-teamers using known evasion strategies (roleplay, translation, obfuscation, suffix attacks, chain-of-thought coaxing)

2. **LLM-Generator Baseline** — GPT-4 generates adversarial prompts via two-stage process:
   - Stage 1: Generate sensitive intent + vague obfuscation
   - Stage 2: Apply evasion strategy with `<INTENT>` token injection

3. **Black-box GA Baseline** *(planned)* — Genetic algorithm optimizing short suffixes/edits to maximize judge reward

4. **RL Editor** *(planned)* — Policy network that edits seed prompts within T steps to bypass filters

### C. Evaluation Metrics

**Primary Metrics:**
- **ASR (Attack Success Rate)** — Fraction of prompts that bypass safety filters per method/intent
- **Queries-to-first-success** — Efficiency metric for iterative methods
- **Transferability** — Success rate of attacks from Model A on Model B

**Quality Metrics:**
- **Judge Error Rate** — Human-vs-automated disagreement rate (Cohen's κ)
- **Diversity** — Clustering successful candidates to count unique attack styles

---

## Local Setup

### 1. Clone Repository
```bash
git clone https://github.com/esi2tekky/shawshank.git
cd shawshank
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure API Keys
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-...
```

### 4. Initialize Environment
```bash
export OPENAI_API_KEY="your-key-here"  # Or source from .env
```

---

## Running the LLM Baseline

### Step 1: Generate Adversarial Prompts
Generate **B** prompts using the LLM-based attack generator (default B=50):

```bash
python src/attacker/gpt_attacker.py --num 50 --output data/gpt_baseline.csv
```

**Output:** `data/gpt_baseline.csv` with columns:
```
id, intent_category, intent_action, vague_action, category, prompt
```

### Step 2: Test Against Target Model + Judge
Feed prompts to target LLM and evaluate with GPT-4 judge:

```bash
python -m src.experiments.run_llm_attacker \
  --input data/gpt_baseline.csv \
  --target openai \
  --target_model gpt-4
```

**Output:** `data/gpt_baseline_attack.csv` with added columns:
```
..., response, judge_label, judge_reason, target_metadata, timestamp_utc
```

### Step 3: Human Annotation
Perform manual validation of judge decisions:

```bash
python tools/human_annotator.py --input data/gpt_baseline_attack.csv
```

**Interactive Controls:**
- `y` / Enter → Mark as compliant (safe)
- `n` → Mark as noncompliant (jailbreak success)
- `s` → Skip
- `q` → Quit and save

**Output:** Adds `human_label` column to the CSV.

### Step 4: Analysis

**Analyze LLM baseline performance:**
```bash
python analysis/analyze_llm_baseline.py
```
Generates ASR by evasion strategy, ASR by intent category, and visualizations.

**Compare Human vs LLM methods:**
```bash
python analysis/compare_human_vs_llm.py
```
Compares attack success rates between manual and ML-generated prompts.

**Validate judge reliability:**
```bash
python analysis/analyze_human_vs_judge.py
```
Computes Cohen's κ, confusion matrix, and judge error rates.

---

## Repository Structure

```
.
├── src/
│   ├── attacker/
│   │   ├── gpt_attacker.py          # Two-stage LLM attack generator
│   │   └── llm_attacker.py          # Legacy single-stage generator
│   ├── target/
│   │   ├── openai_target.py         # OpenAI API wrapper
│   │   └── local_proxy.py           # Local stub for testing
│   ├── judge/
│   │   └── gpt4_judge.py            # GPT-4 safety classifier
│   ├── experiments/
│   │   ├── run_llm_attacker.py      # LLM baseline pipeline
│   │   └── run_human_baseline.py    # Human baseline pipeline
│   └── utils/
│       └── storage.py               # Data I/O utilities
├── analysis/
│   ├── analyze_llm_baseline.py      # LLM method analysis
│   ├── compare_human_vs_llm.py      # Method comparison
│   └── analyze_human_vs_judge.py    # Judge validation
├── tools/
│   ├── human_annotator.py           # Interactive labeling tool
│   ├── csv_to_jsonl.py              # Format conversion
│   └── export_for_annotation.py     # Prepare data for labeling
├── data/
│   ├── gpt_baseline.csv             # Generated prompts
│   └── gpt_baseline_attack.csv      # Attack results
├── requirements.txt
├── Dockerfile
└── .env.example
```

---

## Docker Setup (Alternative)

For reproducible environments:

```bash
# Build container
docker build -t shawshank:dev .

# Run experiments
docker run -it --env-file .env -v $(pwd):/app shawshank:dev bash
```

---

## Safety & Ethics

⚠️ **This project is for authorized security research only.**

**Guidelines:**
- Keep all harmful outputs restricted to the research team
- Mask operational details before sharing results
- Consult your institution's IRB/ethics board before publication
- Never store or publish actual harmful model generations
- Store only labels, embeddings, and redacted excerpts for analysis

**Recommended Protocol:**
- Use ≥2 independent annotators blind to prompt source
- Report Cohen's κ (target >0.6 for reliable labeling)
- Sample balanced sets of successes/failures for validation

---

## Cost Management

**API Usage:**
- LLM generation: ~$0.01-0.03 per prompt (GPT-4)
- Target queries: ~$0.01-0.05 per response (GPT-4)
- Judge evaluations: ~$0.005-0.01 per classification

**Recommendations:**
- Start with B=10 for pilot runs to estimate costs
- Monitor token usage via OpenAI dashboard
- Use local models for iterative development

---

## Extending the Framework

**Add New Attack Methods:**
```python
# src/attacker/ga_attacker.py
def generate_with_ga(seed_prompts, fitness_fn, generations=100):
    # Implement genetic algorithm baseline
    pass
```

**Add New Target Models:**
```python
# src/target/anthropic_target.py
from anthropic import Anthropic

class AnthropicTarget(TargetAPI):
    def query(self, prompt):
        # Implement Claude API wrapper
        pass
```

**Custom Metrics:**
```python
# analysis/diversity_analysis.py
from sentence_transformers import SentenceTransformer
# Cluster embeddings of successful prompts
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Use `python -m src.module.script` syntax |
| `openai.error.AuthenticationError` | Check `OPENAI_API_KEY` in `.env` |
| Judge returns malformed JSON | Increase `max_tokens` in `gpt4_judge.py` |
| Import errors | Ensure running from project root directory |

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{shawshank2025,
  title={Breaking LLMs Using ML: A Comparative Study of Jailbreak Generation},
  author={Your Name},
  year={2025},
  howpublished={CS229 Final Project, Stanford University}
}
```

---

## License

This project is for educational and research purposes only. See LICENSE for details.
