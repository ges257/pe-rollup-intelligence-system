# Challenges: Cold-Start Scarcity in Private Equity AI

> For the full data generation pipeline, see [causal-synth-engine](https://github.com/ges257/causal-synth-engine)

## The Problem

Private Equity vendor consolidation data has three fatal characteristics:

1. **Fragmented**: Each portfolio company tracks vendors differently
2. **Private**: Contracts, pricing, and switching decisions are confidential
3. **Sparse**: A typical PE fund might see 10-20 vendor transitions per year

**Result**: Training a Graph Neural Network on real PE data leads to massive overfitting. The model memorizes the few examples rather than learning generalizable patterns.

---

## The Naive Solution (And Why It Fails)

The obvious approach: generate random synthetic data.

```python
# Naive approach
vendor_switch = random.choice([True, False], p=[0.05, 0.95])
```

**Problem**: Random data has no structure. A model trained on random switches learns nothing about *why* practices switch vendors. When deployed, it performs no better than guessing.

---

## The Innovation: Mechanism-First Data Generation

I inverted the typical ML workflow:

| Traditional ML | This Project |
|---------------|--------------|
| Collect data → Find patterns | Define mechanisms → Generate data |
| Model discovers structure | Model must rediscover known structure |
| Overfits to noise | Learns causal relationships |

### Step 1: Define the Causal Mechanisms

Before generating any data, I encoded the business logic of vendor switching:

```python
# The "Why" of vendor switching
P_switch = base_rate × integration_multiplier × fatigue_multiplier
```

**Integration Quality** (the key driver):
- `0 (none)`: Manual data entry, constant friction → 2x more likely to switch
- `1 (partial_csv)`: File-based workflows, moderate friction → 1.3x
- `2 (full_api)`: Real-time sync, low friction → 0.7x (sticky)

**Change Fatigue** (the timing constraint):
- Recently switched (<12 months) → 0.3x (exhausted, won't switch again)
- Mid-tenure (12-24 months) → 0.7x (hesitant)
- Long tenure (>24 months) → 1.0x (ready for change)

### Step 2: Ground in Real-World Knowledge (LLM Research)

Random vendor names and features would be meaningless. I used LLMs to research actual dental industry vendors:

**Research Process**:
1. Identified 20 real vendors across 7 categories (Lab, RCM, Telephony, etc.)
2. Prompted Claude/GPT to extract structured information from official websites
3. Classified each vendor's EHR integrations as full_api, partial_csv, or none

**Example Output** (from `data/llm_research/batch1_lab_vendors.md`):
```
V002: Glidewell
- Category: Lab
- EHR Integration: partial_csv (portal-based STL upload)
- DSO Partnerships: Heartland Dental (confirmed)
- Scale: 80,000+ dentists annually
```

**Key Insight Discovered**: All dental labs use portal-based ordering (partial_csv), not real-time APIs. This is a structural characteristic of the industry, not vendor laziness.

### Step 3: Generate Data That Obeys the Mechanisms

With mechanisms defined and vendors researched, I built a 6-step Python pipeline:

```
generate_sites.py         → 100 dental practices
generate_vendors.py       → 20 vendors (real characteristics)
generate_integration_matrix.py → Site-vendor integration quality
generate_initial_state.py → 2019 baseline contracts
simulate_switches.py      → Apply causal switching logic
generate_kpis.py          → Days A/R, Denial Rate (vendor-affected)
```

**Result**: 866 contracts with realistic switching patterns, 7,200 monthly KPI records.

---

## Validation: Did the Model Learn the Mechanism?

The acid test: **ablation studies**.

If the model truly learned that integration quality drives switching, removing that feature should devastate performance.

| Experiment | PR-AUC | Delta |
|------------|--------|-------|
| Full model | 0.9407 | - |
| Remove `integration_quality` | 0.6852 | **-25.5%** |
| Remove `vendor_tier` | 0.9368 | -0.4% |
| Remove `site_region` | 0.9260 | -1.6% |

**The model correctly identified integration quality as the dominant feature.** This validates that:
1. The synthetic data preserved the causal structure
2. The R-GCN learned the mechanism, not arbitrary patterns
3. The model will generalize to real data with the same structure

---

## Trade-offs and Limitations

### What This Approach Enables
- Reproducible research (full dataset included)
- Controlled ablation experiments
- Confident deployment (known causal structure)

### What This Approach Cannot Do
- Capture idiosyncratic real-world factors (politics, relationships)
- Model rare black swan events
- Replace domain expertise for mechanism design

### If I Had More Time
1. Add more sophisticated fatigue modeling (cumulative stress)
2. Incorporate geographic clustering effects
3. Model vendor-to-vendor competition dynamics

---

## References

- **Full data generation pipeline:** [causal-synth-engine](https://github.com/ges257/causal-synth-engine)
- LLM research samples: [data/llm_research/](data/llm_research/)
- Methodology documentation: [docs/methodology.md](docs/methodology.md)
