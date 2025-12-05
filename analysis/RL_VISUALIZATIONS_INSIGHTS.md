# RL Attack Visualizations: Key Insights

**Generated:** December 5, 2025

This document explains the insights provided by the RL attack visualizations and how to interpret them for analysis.

---

## Visualization Overview

### 1. Learning Curves (`learning_curves.png`)

**What it shows:**
- **Top Left**: Reward over episodes (smoothed) - Shows if the agent is learning to get better rewards
- **Top Right**: Success rate over episodes (rolling average) - Shows if attack success improves over time
- **Bottom Left**: Episode length over training - Shows if agent learns to succeed faster (shorter episodes)
- **Bottom Right**: Policy loss over training (log scale) - Shows learning dynamics

**Key Insights:**
- **Upward trend in reward/success** = Agent is learning effective attacks
- **Downward trend in episode length** = Agent learns to succeed faster (more efficient)
- **Decreasing policy loss** = Policy is converging (learning is stabilizing)
- **Plateaus** = Agent has reached its learning limit or needs more exploration

**Research Questions:**
- Do all models show learning, or do some plateau early?
- Which models show the steepest learning curves (fastest improvement)?
- Is there overfitting (early improvement, then decline)?

---

### 2. Model Comparison (`model_comparison.png`)

**What it shows:**
- **Left**: Overall Attack Success Rate (ASR) by model
- **Middle**: Average reward per episode
- **Right**: Average episode length (steps to success)

**Key Insights:**
- **Higher ASR** = Model is more vulnerable to RL attacks
- **Higher reward** = Attacks are more successful (higher judge scores)
- **Shorter episodes** = Agent finds successful attacks faster (more efficient)

**Research Questions:**
- Which alignment stage is most vulnerable? (Base vs SFT vs DPO vs RLVR)
- Is there a correlation between reward and ASR?
- Do more vulnerable models require shorter or longer episodes?

**Expected Pattern:**
- If alignment creates vulnerabilities: DPO/RLVR > Base/SFT
- If alignment improves safety: Base > SFT > DPO > RLVR

---

### 3. Intent Vulnerability Heatmap (`intent_vulnerability_heatmap.png`)

**What it shows:**
- Success rate for each intent category (rows) across all models (columns)
- Color intensity = vulnerability (darker = more vulnerable)

**Key Insights:**
- **Dark rows** = Intent categories that are universally vulnerable
- **Dark columns** = Models that are vulnerable across many categories
- **Patterns** = Some intents may be inherently harder/easier to attack
- **Model-specific vulnerabilities** = Some models fail on specific categories

**Research Questions:**
- Which intent categories are most vulnerable across all models?
- Do different alignment stages have different failure modes?
- Are there categories where alignment actually helps (lighter colors for aligned models)?

**Expected Patterns:**
- Categories like "hacking", "fraud" might be more vulnerable
- Categories like "self_harm" might be better defended
- Alignment might help on some categories but hurt on others

---

### 4. Strategy Effectiveness (`strategy_effectiveness.png`)

**What it shows:**
- Top 15 strategy combinations ranked by success rate
- Shows which strategy combinations the RL agent learned to use most effectively

**Key Insights:**
- **High success rate strategies** = What the agent learned works best
- **High attempt count** = Agent frequently uses these strategies (exploitation)
- **Low attempt count** = Rare but effective strategies (exploration success)

**Research Questions:**
- Which strategy combinations are most effective?
- Does the agent learn novel combinations not in the initial taxonomy?
- Are single strategies or combinations more effective?

**Expected Patterns:**
- Combinations like "roleplay + chain_of_thought" might be highly effective
- Some strategies might work better on specific models
- Agent might discover unexpected effective combinations

---

### 5. Early vs Late Training (`early_vs_late.png`)

**What it shows:**
- **Left**: ASR in first half vs second half of training
- **Right**: Average reward in first half vs second half

**Key Insights:**
- **Improvement (late > early)** = Agent is learning and improving
- **No improvement** = Agent reached learning limit early or needs more training
- **Decline** = Possible overfitting or exploration exhaustion

**Research Questions:**
- Do all models show learning progress?
- Which models show the most improvement?
- Is 200 episodes sufficient for learning, or do we need more?

**Expected Patterns:**
- Models with steeper improvement curves = more learnable attack surface
- Models that plateau early = either well-defended or agent needs different approach
- Significant improvement suggests RL is finding novel attack strategies

---

### 6. Reward Distribution (`reward_distribution.png`)

**What it shows:**
- Histogram of total rewards per episode for each model
- Shows the distribution of attack success (not just average)

**Key Insights:**
- **Right-skewed** = Most episodes fail (low rewards), but some succeed (high rewards)
- **Normal distribution** = Consistent performance
- **Bimodal** = Two distinct modes (successful vs unsuccessful attacks)
- **Mean vs Median** = Shows if distribution is skewed

**Research Questions:**
- Is success rare but high-reward, or frequent but low-reward?
- Do models have different reward distributions (different attack profiles)?
- Are there "lucky" high-reward episodes or consistent performance?

**Expected Patterns:**
- Models with right-skewed distributions = Hard to attack, but some successful attacks
- Models with normal distributions = Consistent vulnerability
- Bimodal distributions = Clear separation between successful and failed attacks

---

## How to Use These Visualizations

### For Paper Writing:

1. **Introduction**: Use learning curves to show RL is effective
2. **Results**: Use model comparison to show vulnerability differences
3. **Analysis**: Use intent heatmap to show category-specific vulnerabilities
4. **Discussion**: Use strategy effectiveness to show what attacks work
5. **Limitations**: Use early vs late to discuss training sufficiency

### For Understanding Model Vulnerabilities:

1. **Overall**: Model comparison shows which models are most vulnerable
2. **Specific**: Intent heatmap shows which categories are vulnerable
3. **Mechanism**: Strategy effectiveness shows how attacks succeed
4. **Learning**: Learning curves show if RL discovers new attacks

### For Comparing Methods:

1. **RL vs Human**: Compare RL ASR (from model comparison) with human baseline ASR
2. **RL vs GA**: Compare RL success rates with GA fitness scores
3. **RL Efficiency**: Episode length shows how many attempts RL needs vs fixed attempts for other methods

---

## Key Research Questions These Visualizations Answer

1. **Does RL learn effective attacks?**
   - Answer: Learning curves (improvement over time)

2. **Which models are most vulnerable?**
   - Answer: Model comparison (ASR ranking)

3. **What types of attacks work best?**
   - Answer: Strategy effectiveness (top strategies)

4. **Are there category-specific vulnerabilities?**
   - Answer: Intent heatmap (category patterns)

5. **Does alignment help or hurt?**
   - Answer: Model comparison (Base vs Aligned models)

6. **Is RL learning or just exploring?**
   - Answer: Early vs late (improvement over time)

7. **How efficient is RL?**
   - Answer: Episode length (steps to success)

---

## Interpretation Guidelines

### Good Signs (RL is working):
- ✅ Upward trend in learning curves
- ✅ Late > Early in training comparison
- ✅ High success rates on vulnerable categories
- ✅ Agent discovers effective strategy combinations
- ✅ Decreasing episode length (learning efficiency)

### Concerning Signs:
- ⚠️ Flat learning curves (no improvement)
- ⚠️ Late < Early (overfitting or exploration exhaustion)
- ⚠️ Very low success rates across all categories
- ⚠️ No clear strategy patterns (random exploration)
- ⚠️ Increasing episode length (getting worse)

### Expected vs Unexpected:
- **Expected**: Some models more vulnerable than others
- **Expected**: Some categories more vulnerable than others
- **Unexpected**: Aligned models more vulnerable than base (alignment paradox)
- **Unexpected**: RL discovers novel effective strategies
- **Unexpected**: RL improves significantly over training

---

## Next Steps for Analysis

1. **Quantitative Analysis**: Extract specific numbers from visualizations
2. **Statistical Testing**: Compare models statistically (t-tests, ANOVA)
3. **Correlation Analysis**: Find correlations between metrics
4. **Case Studies**: Examine specific high-reward episodes
5. **Comparison with Other Methods**: Compare RL results with GA, Human baseline

---

## File Locations

All visualizations are saved in: `analysis/figures/rl_analysis/`

- `learning_curves.png`
- `model_comparison.png`
- `intent_vulnerability_heatmap.png`
- `strategy_effectiveness.png`
- `early_vs_late.png`
- `reward_distribution.png`

To regenerate: `python analysis/visualize_rl_results.py`

