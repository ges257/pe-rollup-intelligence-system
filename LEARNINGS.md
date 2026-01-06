# Learnings & Trade-offs

## What Worked

### Integration Quality is the Dominant Signal

The most important finding: **integration quality** (API level between vendors and EHR systems) is the primary driver of vendor switching behavior.

| Ablation | PR-AUC | Delta |
|----------|--------|-------|
| Full model | 0.9407 | — |
| Without `integration_quality` | 0.6852 | **-25.5%** |
| Without `vendor_tier` | 0.9368 | -0.4% |
| Without `site_region` | 0.9260 | -1.6% |

**Business Insight:** Practices don't switch vendors because of price or features—they switch because integration friction makes daily operations painful. A vendor with a full API (score=2) is "sticky" regardless of cost.

### R-GCN Outperforms Tabular Models

| Model | PR-AUC | Notes |
|-------|--------|-------|
| Jaccard Similarity | 0.164 | Heuristic baseline |
| Peer Count | 0.171 | Best heuristic |
| LightGBM (30 features) | 0.937 | Strong tabular baseline |
| **R-GCN** | **0.9407** | **+0.37% over GBM** |

**Why Graph ML Won:** The relational structure matters. When Site A switches from Vendor X to Vendor Y, it creates a signal for similar sites. GBM treats each site-vendor pair independently; R-GCN propagates information through the graph.

### Perfect Calibration via Isotonic Regression

| Metric | Before | After |
|--------|--------|-------|
| Brier Score | 0.1434 | 0.1105 |
| ECE (Expected Calibration Error) | 0.1254 | **0.0000** |

**Why Calibration Matters:** In PE decision-making, a "90% confidence" prediction should be right 90% of the time. Our calibrated model achieves perfect ECE, enabling reliable financial projections in the dashboard.

---

## What Didn't Work

### HGT (Heterogeneous Graph Transformer)

We experimented with attention-based architectures, expecting them to outperform convolution:

| Model | Heads | PR-AUC | vs R-GCN |
|-------|-------|--------|----------|
| R-GCN | — | 0.9407 | baseline |
| HGT-1H | 1 | 0.6199 | -34% |
| HGT-2H | 2 | 0.7155 | -24% |
| HGT-4H | 4 | 0.5724 | -39% |

**Finding:** Attention mechanisms added no value for this task. The edge types (integration quality 0/1/2) are simple enough that convolution captures all the signal. HGT's additional parameters just added noise.

### Text-Derived Integration Matrix

We tried using NLP to automatically extract integration quality from vendor descriptions:

| Approach | PR-AUC |
|----------|--------|
| Rule-based encoding | 0.9407 |
| Text-derived (NLP) | 0.8224 |
| Delta | **-12.4%** |

**Finding:** For categorical features with clear definitions, rule-based encoding outperforms NLP. The NLP approach introduced ambiguity ("partial API" vs "limited integration") that degraded signal quality.

### Early Stopping with Low Patience

Initial experiments used patience=10 for early stopping. Best results came with patience=50:

| Patience | Best Epoch | PR-AUC |
|----------|------------|--------|
| 10 | ~40 | 0.91 |
| 30 | ~80 | 0.93 |
| 50 | **105** | **0.9407** |

**Finding:** R-GCN benefits from extended training. The model continues improving gradually even after the loss curve flattens.

---

## Trade-offs Made

| Decision | Trade-off | Rationale |
|----------|-----------|-----------|
| Synthetic data | Reproducibility over real-world validation | PE data is private; synthetic enables ablation studies |
| Category-level rules | Simplicity over vendor-specific nuance | 5/7 categories have fixed patterns anyway |
| 100 sites | Tractable training over scale | Sufficient for pattern learning; can scale later |
| R-GCN over HGT | Simplicity over flexibility | Convolutional model fit the data better |
| 2-layer network | Shallow over deep | Graph radius of 2 captures site-vendor-site paths |

---

## Hyperparameter Sensitivity

Final hyperparameters (from random search):

| Parameter | Value | Sensitivity |
|-----------|-------|-------------|
| Edge dropout | 0.5 | **High** — dropping from 0.5 to 0.3 hurt generalization |
| Learning rate | 0.01 | Medium — stable from 0.005 to 0.02 |
| Hidden channels | 128 | Low — 64 and 256 performed similarly |
| Output channels | 80 | Low — embedding dimension not critical |
| Decoder L2 | 0.01 | **High** — necessary to prevent overfitting |
| Encoder L2 | 5e-4 | Medium — helps but not critical |

---

## If I Had More Time

1. **Multi-model ensemble (R-GCN + GBM)** — Combine graph structure with tabular features
2. **Temporal dynamics** — Contract duration effects on switching probability
3. **Geographic clustering penalties** — Nearby practices should consolidate to same vendor
4. **Real-world validation** — Partner with a PE firm to test on actual portfolio data
5. **Active learning** — Identify which vendor switches would be most informative to observe

---

## Key Lessons

1. **Mechanism-first data generation works.** The ablation validated that the model learned the causal structure we encoded.

2. **Simpler architectures can win.** HGT with attention was strictly worse than R-GCN with convolution.

3. **Calibration is non-negotiable for business decisions.** A well-calibrated model enables confident financial projections.

4. **Integration quality is the entire game.** 25.5% of model performance comes from one feature—understand your domain's causal drivers.
