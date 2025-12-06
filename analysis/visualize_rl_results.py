"""
analysis/visualize_rl_results.py
---------------------------------
Generate insightful visualizations for RL attack results.

Creates:
1. Learning curves (reward, success rate over episodes)
2. Model comparison (ASR, average reward, success rate)
3. Vulnerability by intent category
4. Strategy effectiveness analysis
5. Episode length trends
6. Training dynamics (policy/value loss, entropy)
7. Reward distribution
8. Early vs late training performance
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import numpy as np
from datetime import datetime

# ==== CONFIG ====
RESULTS_DIR = Path("results/rl_attacks")
OUT_DIR = Path("analysis/figures/rl_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Model mapping
MODEL_MAPPING = {
    "meta-llama_Llama-3.1-8B": "llama_base",
    "allenai_Llama-3.1-Tulu-3-8B-SFT": "tulu_sft",
    "allenai_Llama-3.1-Tulu-3-8B-DPO": "tulu_dpo",
    "allenai_Llama-3.1-Tulu-3-8B": "tulu_rlvr",
    "gpt-4o": "gpt_4o"
}

MODEL_LABELS = {
    "llama_base": "Llama Base",
    "tulu_sft": "Tulu SFT",
    "tulu_dpo": "Tulu DPO",
    "tulu_rlvr": "Tulu RLVR",
    "gpt_4o": "GPT-4o"
}

# ==== LOAD DATA ====
print("Loading RL results...")
all_history = {}
all_experiences = {}

# Find all history files
for history_file in RESULTS_DIR.rglob("rl_history_*.jsonl"):
    # Try to extract model from path or filename
    model_key = None
    
    # Check if in subdirectory
    parent_dir = history_file.parent.name
    if parent_dir in MODEL_LABELS:
        model_key = parent_dir
    else:
        # Extract from filename
        filename = history_file.stem.replace("rl_history_", "")
        # Remove timestamp (last part)
        parts = filename.rsplit("_", 2)
        if len(parts) >= 2 and len(parts[-1]) == 6 and parts[-1].isdigit():
            model_name = "_".join(parts[:-2])
            model_key = MODEL_MAPPING.get(model_name, model_name)
        else:
            model_name = "_".join(parts[:-1]) if len(parts) > 1 else parts[0]
            model_key = MODEL_MAPPING.get(model_name, model_name)
    
    if not model_key:
        continue
    
    records = []
    with open(history_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    if model_key not in all_history:
        all_history[model_key] = []
    all_history[model_key].extend(records)
    print(f"  ✓ {model_key}: {len(records)} episodes")

# Find all experience files
for exp_file in RESULTS_DIR.rglob("rl_experiences_*.jsonl"):
    parent_dir = exp_file.parent.name
    if parent_dir in MODEL_LABELS:
        model_key = parent_dir
    else:
        filename = exp_file.stem.replace("rl_experiences_", "")
        parts = filename.rsplit("_", 2)
        if len(parts) >= 2 and len(parts[-1]) == 6 and parts[-1].isdigit():
            model_name = "_".join(parts[:-2])
            model_key = MODEL_MAPPING.get(model_name, model_name)
        else:
            model_name = "_".join(parts[:-1]) if len(parts) > 1 else parts[0]
            model_key = MODEL_MAPPING.get(model_name, model_name)
    
    if not model_key:
        continue
    
    records = []
    with open(exp_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    if model_key not in all_experiences:
        all_experiences[model_key] = []
    all_experiences[model_key].extend(records)
    print(f"  ✓ {model_key}: {len(records)} experiences")

# ==== VISUALIZATION 1: Learning Curves ====
print("\nCreating learning curves...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1.1 Reward over episodes
ax = axes[0, 0]
for model_key in sorted(all_history.keys()):
    if model_key == "gpt_4o":  # Skip GPT-4o for now, focus on open models
        continue
    episodes = all_history[model_key]
    if not episodes:
        continue
    
    df = pd.DataFrame(episodes)
    # Smooth with rolling average
    window = max(1, len(df) // 20)  # Adaptive window
    ax.plot(df['episode'], df['total_reward'].rolling(window=window, center=True).mean(), 
            label=MODEL_LABELS.get(model_key, model_key), linewidth=2, alpha=0.8)
    ax.scatter(df['episode'], df['total_reward'], alpha=0.1, s=5)

ax.set_xlabel('Episode', fontsize=12)
ax.set_ylabel('Total Reward (Smoothed)', fontsize=12)
ax.set_title('Learning Curve: Reward Over Episodes', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 1.2 Success rate over episodes (rolling)
ax = axes[0, 1]
for model_key in sorted(all_history.keys()):
    if model_key == "gpt_4o":
        continue
    episodes = all_history[model_key]
    if not episodes:
        continue
    
    df = pd.DataFrame(episodes)
    window = max(10, len(df) // 10)
    success_rate = df['success'].rolling(window=window, center=True).mean()
    ax.plot(df['episode'], success_rate, 
            label=MODEL_LABELS.get(model_key, model_key), linewidth=2, alpha=0.8)

ax.set_xlabel('Episode', fontsize=12)
ax.set_ylabel('Success Rate (Rolling Average)', fontsize=12)
ax.set_title('Learning Curve: Success Rate Over Episodes', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0, 1)

# 1.3 Episode length over time
ax = axes[1, 0]
for model_key in sorted(all_history.keys()):
    if model_key == "gpt_4o":
        continue
    episodes = all_history[model_key]
    if not episodes:
        continue
    
    df = pd.DataFrame(episodes)
    window = max(1, len(df) // 20)
    ax.plot(df['episode'], df['num_steps'].rolling(window=window, center=True).mean(), 
            label=MODEL_LABELS.get(model_key, model_key), linewidth=2, alpha=0.8)

ax.set_xlabel('Episode', fontsize=12)
ax.set_ylabel('Episode Length (Steps)', fontsize=12)
ax.set_title('Episode Length Over Training', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 1.4 Policy loss over episodes
ax = axes[1, 1]
for model_key in sorted(all_history.keys()):
    if model_key == "gpt_4o":
        continue
    episodes = all_history[model_key]
    if not episodes:
        continue
    
    df = pd.DataFrame(episodes)
    window = max(1, len(df) // 20)
    policy_loss = df['policy_loss'].abs()  # Use absolute value
    ax.plot(df['episode'], policy_loss.rolling(window=window, center=True).mean(), 
            label=MODEL_LABELS.get(model_key, model_key), linewidth=2, alpha=0.8)

ax.set_xlabel('Episode', fontsize=12)
ax.set_ylabel('|Policy Loss| (Smoothed)', fontsize=12)
ax.set_title('Policy Loss Over Training', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(OUT_DIR / 'learning_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# ==== VISUALIZATION 2: Model Comparison ====
print("Creating model comparison...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

models = [k for k in sorted(all_history.keys()) if k != "gpt_4o"]
metrics_data = {
    'model': [],
    'asr': [],
    'avg_reward': [],
    'avg_episode_length': []
}

for model_key in models:
    episodes = all_history[model_key]
    if not episodes:
        continue
    
    df = pd.DataFrame(episodes)
    success_rate = df['success'].mean()
    avg_reward = df['total_reward'].mean()
    avg_length = df['num_steps'].mean()
    
    metrics_data['model'].append(MODEL_LABELS.get(model_key, model_key))
    metrics_data['asr'].append(success_rate)
    metrics_data['avg_reward'].append(avg_reward)
    metrics_data['avg_episode_length'].append(avg_length)

metrics_df = pd.DataFrame(metrics_data)

# 2.1 ASR Comparison
ax = axes[0]
bars = ax.bar(metrics_df['model'], metrics_df['asr'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax.set_ylabel('Attack Success Rate', fontsize=12)
ax.set_title('Overall Attack Success Rate by Model', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)
for bar, asr in zip(bars, metrics_df['asr']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{asr:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 2.2 Average Reward
ax = axes[1]
bars = ax.bar(metrics_df['model'], metrics_df['avg_reward'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax.set_ylabel('Average Total Reward', fontsize=12)
ax.set_title('Average Reward per Episode', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, reward in zip(bars, metrics_df['avg_reward']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{reward:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 2.3 Average Episode Length
ax = axes[2]
bars = ax.bar(metrics_df['model'], metrics_df['avg_episode_length'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax.set_ylabel('Average Episode Length (Steps)', fontsize=12)
ax.set_title('Average Episode Length', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, length in zip(bars, metrics_df['avg_episode_length']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{length:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(OUT_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ==== VISUALIZATION 3: Vulnerability by Intent ====
print("Creating intent category analysis...")
intent_stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'successes': 0, 'total_reward': 0}))

for model_key in models:
    episodes = all_history[model_key]
    if not episodes:
        continue
    
    for ep in episodes:
        intent = ep.get('intent', 'unknown')
        intent_stats[model_key][intent]['total'] += 1
        intent_stats[model_key][intent]['total_reward'] += ep.get('total_reward', 0)
        if ep.get('success', False):
            intent_stats[model_key][intent]['successes'] += 1

# Create heatmap
all_intents = set()
for model_stats in intent_stats.values():
    all_intents.update(model_stats.keys())

intent_data = []
for intent in sorted(all_intents):
    row = {'Intent': intent.replace('_', ' ').title()}
    for model_key in models:
        stats = intent_stats[model_key][intent]
        asr = stats['successes'] / stats['total'] if stats['total'] > 0 else 0
        row[MODEL_LABELS.get(model_key, model_key)] = asr
    intent_data.append(row)

intent_df = pd.DataFrame(intent_data).set_index('Intent')

fig, ax = plt.subplots(figsize=(10, max(8, len(intent_df) * 0.5)))
sns.heatmap(intent_df, annot=True, fmt='.1%', cmap='YlOrRd', vmin=0, vmax=1, 
            cbar_kws={'label': 'Attack Success Rate'}, ax=ax, linewidths=0.5)
ax.set_title('Attack Success Rate by Intent Category and Model', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Intent Category', fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / 'intent_vulnerability_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# ==== VISUALIZATION 4: Strategy Effectiveness ====
print("Creating strategy analysis...")
strategy_stats = defaultdict(lambda: {'total': 0, 'successes': 0, 'total_reward': 0})

for model_key in models:
    experiences = all_experiences.get(model_key, [])
    for exp in experiences:
        strategies = exp.get('strategies', [])
        strategy_str = '+'.join(strategies) if isinstance(strategies, list) else str(strategies)
        strategy_stats[strategy_str]['total'] += 1
        strategy_stats[strategy_str]['total_reward'] += exp.get('reward', 0)
        if exp.get('reward', 0) >= 0.7:  # Success threshold
            strategy_stats[strategy_str]['successes'] += 1

# Get top strategies
strategy_list = []
for strategy, stats in strategy_stats.items():
    if stats['total'] >= 5:  # Minimum occurrences
        success_rate = stats['successes'] / stats['total'] if stats['total'] > 0 else 0
        strategy_list.append({
            'strategy': strategy,
            'success_rate': success_rate,
            'attempts': stats['total'],
            'avg_reward': stats['total_reward'] / stats['total'] if stats['total'] > 0 else 0
        })

strategy_df = pd.DataFrame(strategy_list).sort_values('success_rate', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(range(len(strategy_df)), strategy_df['success_rate'], color='steelblue')
ax.set_yticks(range(len(strategy_df)))
ax.set_yticklabels(strategy_df['strategy'], fontsize=9)
ax.set_xlabel('Success Rate', fontsize=12)
ax.set_title('Top 15 Strategy Combinations by Success Rate', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, rate, attempts) in enumerate(zip(bars, strategy_df['success_rate'], strategy_df['attempts'])):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{rate:.1%} (n={attempts})', ha='left', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / 'strategy_effectiveness.png', dpi=300, bbox_inches='tight')
plt.close()

# ==== VISUALIZATION 5: Early vs Late Training ====
print("Creating early vs late comparison...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

early_late_data = []
for model_key in models:
    episodes = all_history[model_key]
    if not episodes or len(episodes) < 20:
        continue
    
    df = pd.DataFrame(episodes)
    mid_point = len(df) // 2
    
    early = df.iloc[:mid_point]
    late = df.iloc[mid_point:]
    
    early_late_data.append({
        'model': MODEL_LABELS.get(model_key, model_key),
        'early_asr': early['success'].mean(),
        'late_asr': late['success'].mean(),
        'early_reward': early['total_reward'].mean(),
        'late_reward': late['total_reward'].mean()
    })

el_df = pd.DataFrame(early_late_data)

# 5.1 ASR comparison
ax = axes[0]
x = np.arange(len(el_df))
width = 0.35
ax.bar(x - width/2, el_df['early_asr'], width, label='First Half', alpha=0.8, color='#ff7f0e')
ax.bar(x + width/2, el_df['late_asr'], width, label='Second Half', alpha=0.8, color='#2ca02c')
ax.set_ylabel('Attack Success Rate', fontsize=12)
ax.set_title('Learning Progress: Early vs Late Training', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(el_df['model'], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1)

# 5.2 Reward comparison
ax = axes[1]
ax.bar(x - width/2, el_df['early_reward'], width, label='First Half', alpha=0.8, color='#ff7f0e')
ax.bar(x + width/2, el_df['late_reward'], width, label='Second Half', alpha=0.8, color='#2ca02c')
ax.set_ylabel('Average Reward', fontsize=12)
ax.set_title('Reward Improvement: Early vs Late Training', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(el_df['model'], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'early_vs_late.png', dpi=300, bbox_inches='tight')
plt.close()

# ==== VISUALIZATION 6: Reward Distribution ====
print("Creating reward distribution...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for idx, model_key in enumerate(models[:4]):  # Limit to 4 models
    ax = axes[idx // 2, idx % 2]
    episodes = all_history[model_key]
    if not episodes:
        continue
    
    df = pd.DataFrame(episodes)
    rewards = df['total_reward'].values
    
    ax.hist(rewards, bins=30, alpha=0.7, color=f'C{idx}', edgecolor='black', linewidth=0.5)
    ax.axvline(rewards.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {rewards.mean():.2f}')
    ax.axvline(np.median(rewards), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(rewards):.2f}')
    ax.set_xlabel('Total Reward', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{MODEL_LABELS.get(model_key, model_key)}: Reward Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'reward_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ All visualizations saved to: {OUT_DIR}")
print("\nGenerated visualizations:")
print("  1. learning_curves.png - Reward, success rate, episode length, policy loss over time")
print("  2. model_comparison.png - ASR, average reward, episode length comparison")
print("  3. intent_vulnerability_heatmap.png - Success rate by intent category")
print("  4. strategy_effectiveness.png - Top performing strategy combinations")
print("  5. early_vs_late.png - Learning progress comparison")
print("  6. reward_distribution.png - Reward distribution per model")

