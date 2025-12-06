# Shawshank

ML-based jailbreak generation for LLM safety testing. Compares human-crafted adversarial prompts against various ML approaches (LLM generation, genetic algorithms, contextual bandits, RL) to find which methods best identify model vulnerabilities.

## Setup

```bash
git clone https://github.com/esi2tekky/shawshank.git
cd shawshank
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add OPENAI_API_KEY
```

## Implementation

### Attackers (`src/attacker/`)

**gpt_attacker.py** - Two-stage LLM prompt generator
- Stage 1: Generate harmful intent + vague obfuscation
- Stage 2: Apply evasion strategy with `<INTENT>` token injection
```bash
python src/attacker/gpt_attacker.py --num 50 --output data/gpt_baseline.csv
```

**bandit_attacker.py** - Thompson Sampling contextual bandit
- Learns which (intent, strategy) combos work best
- Two variants: contextual (726 arms) and intent-agnostic (66 arms)
```bash
python -m src.attacker.bandit_attacker --target gpt-4o --trials 500 --intent-agnostic
```

**rl_attacker.py** - Actor-Critic policy gradient
- Multi-step episodes with adaptive strategy selection
- Learns to select effective attack strategies over time
```bash
python -m src.attacker.rl_attacker --target gpt-4o --episodes 100
```

**mutation_operators.py** - GA mutation/crossover ops
- Rule-based: roleplay prefix, hypothetical frame, suffix injection, authority appeal, etc.
- LLM-based: rephrase mutations, intelligent crossover

### Judge (`src/judge/`)

**gpt4_judge_continuous.py** - Scores responses 0.0 (blocked) to 1.0 (jailbroken)
- Returns `{score, reason, decision}` for each prompt/response pair

### Experiments (`src/experiments/`)

**run_llm_attacker.py** - Test generated prompts against target models
```bash
python -m src.experiments.run_llm_attacker --input data/gpt_baseline.csv --target openai --target_model gpt-4
```

**run_ga_attacker.py** - Genetic algorithm optimization
```bash
python -m src.experiments.run_ga_attacker --target gpt-4o --generations 50 --population 20
```

**run_structured_ga.py** - Structured GA with prefix/suffix evolution

**run_human_baseline.py** - Run human-crafted prompts through pipeline

### Analysis (`analysis/`)

- `analyze_llm_baseline.py` - LLM baseline metrics
- `analyze_baseline.py` - General baseline analysis
- `compare_human_vs_llm.py` - Compare human vs LLM attack effectiveness
- `analyze_human_vs_judge.py` - Judge agreement metrics
- `analyze_tulu_comparison.py` - Cross-model comparison

## Project Structure

```
src/
  attacker/      # Attack generation methods
  target/        # Model API wrappers (OpenAI, local)
  judge/         # Response evaluation
  experiments/   # Experiment runners
analysis/        # Results analysis scripts
data/            # Generated datasets and results
```

## Ethics

Research-only. Keep harmful outputs restricted, mask details before sharing, never publish actual model generations.
