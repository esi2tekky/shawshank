# GA Time Optimization Analysis

## Current Configuration
- **Seeds**: 50 prompts
- **Population**: 100 individuals (50 seeds × 2: 50 elites + 50 mutations)
- **Generations**: 10
- **Total evaluations**: 4 models × 1,000 = **4,000 evaluations**
- **Estimated time**: 6-10 hours

## Time Reduction Options

### Option 1: Reduce Generations (RECOMMENDED)
**Change**: 10 → 5 generations
- **Evaluations**: 4 models × 500 = **2,000 evaluations**
- **Time**: ~3-5 hours (50% reduction)
- **Rigor impact**: Low - Most GA improvement happens in early generations
- **Rationale**: Diminishing returns after generation 5-7

### Option 2: Reduce Population Size
**Change**: 100 → 60 individuals (50 elites + 10 mutations)
- **Evaluations**: 4 models × 600 = **2,400 evaluations**
- **Time**: ~3.5-6 hours (40% reduction)
- **Rigor impact**: Medium - Reduces diversity and exploration
- **Rationale**: Less mutation diversity, but still maintains all seed topics

### Option 3: Reduce Both (Balanced)
**Change**: 10 → 6 generations, 100 → 75 individuals
- **Evaluations**: 4 models × 450 = **1,800 evaluations**
- **Time**: ~2.5-4 hours (60% reduction)
- **Rigor impact**: Low-Medium
- **Rationale**: Good balance between time and exploration

### Option 4: Aggressive Reduction
**Change**: 10 → 5 generations, 100 → 50 individuals (elites only, no mutations)
- **Evaluations**: 4 models × 250 = **1,000 evaluations**
- **Time**: ~1.5-2.5 hours (75% reduction)
- **Rigor impact**: High - No evolution, just evaluation
- **Rationale**: Fastest but loses GA benefits

## Recommendation: Option 1 (5 Generations)

**Why:**
1. **Maintains diversity**: Still evaluates all 50 seed topics with mutations
2. **Preserves GA benefits**: Evolution still occurs, just fewer cycles
3. **Research shows**: Most improvement in GA happens in first 5-7 generations
4. **Time savings**: Cuts time in half (3-5 hours)
5. **Still rigorous**: 500 evaluations per model is substantial

**Implementation:**
- Change `--generations 10` to `--generations 5` in script
- Keep population at 100 (maintains diversity)
- All other parameters unchanged

## Alternative: Option 3 (Balanced)

If you need even faster results:
- 6 generations + 75 individuals
- Still maintains GA evolution
- Good compromise between speed and rigor

## Time Estimates by Option

| Option | Generations | Population | Eval/Model | Total Time | Rigor |
|--------|-------------|------------|------------|------------|-------|
| Current | 10 | 100 | 1,000 | 6-10 hours | High |
| Option 1 | 5 | 100 | 500 | **3-5 hours** | **High** |
| Option 2 | 10 | 60 | 600 | 3.5-6 hours | Medium |
| Option 3 | 6 | 75 | 450 | 2.5-4 hours | Medium-High |
| Option 4 | 5 | 50 | 250 | 1.5-2.5 hours | Low |

## Cost Impact

All options maintain the same cost structure:
- **Judge calls**: Proportional to evaluations
- **Option 1**: ~$0.39 (50% of original $0.78)
- **Option 3**: ~$0.35 (45% of original)

Cost is minimal regardless of option chosen.

