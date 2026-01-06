# Implementation Challenges & Debugging

This document captures the key challenges overcome during the development of the PE Rollup Intelligence Platform.

---

## Challenge 1: R-GCN Underfitting (PR-AUC 0.834 → 0.9407)

**Problem:** Initial R-GCN achieved only PR-AUC 0.834, substantially below LightGBM baseline (0.937).

**Diagnosis:**
- Learning rate too low (0.001) — slow convergence
- No edge dropout — model memorizing training edges
- Embedding dimensions too small — insufficient capacity

**Solution:**
1. **Paper-based tuning:** Increased LR to 0.01, added edge dropout 0.4, expanded hidden channels 64→128
2. **Random search:** 25-trial search identified edge_dropout=0.5 as optimal

**Result:** PR-AUC 0.834 → 0.9407 (+10.7 pp)

---

## Challenge 2: SAGEConv Ignores Edge Types (PR-AUC 0.687)

**Problem:** SAGEConv achieved only PR-AUC 0.687, far below baselines.

**Diagnosis:** SAGEConv aggregates neighbor messages uniformly without considering edge types. Edges representing "no integration" (0), "partial integration" (1), and "full API integration" (2) are treated identically, discarding the primary predictive signal.

**Solution:** Switched to R-GCN which explicitly models relation types via separate weight matrices per edge type.

**Result:** +25.4% PR-AUC gain from architecture change alone.

---

## Challenge 3: TGN Architecture Mismatch (PR-AUC 0.557)

**Problem:** Temporal Graph Networks performed only marginally above random baseline (0.557).

**Diagnosis:** TGN is optimized for continuous-time event streams (e.g., social media interactions). The PE rollup data consists of persistent contracts (2019-2024) rather than discrete events. TGN's memory module expects high-frequency state updates, but vendor contracts change infrequently.

**Solution:** Abandoned TGN in favor of static R-GCN.

---

## Challenge 4: Class Imbalance (85% Negative)

**Problem:** Early experiments used accuracy. Models achieved 90%+ accuracy by predicting "no adoption" for all pairs.

**Diagnosis:** Class distribution is ~85% negative, ~15% positive. A model predicting all negatives achieves 85% accuracy but 0% recall.

**Solution:**
- Switched primary metric from accuracy to PR-AUC
- Used class weights in loss function
- Added ROC-AUC, Recall@K, and Brier score for comprehensive evaluation

---

## Challenge 5: Calibration Drift (ECE → 0.0000)

**Problem:** Early R-GCN produced well-ranked predictions but poorly calibrated probabilities. Predictions with ŷ=0.8 had empirical adoption rates of only 0.6.

**Solution:** Applied isotonic regression calibration on validation set.

**Result:** Final model achieves ECE ≈ 0.0000 (perfect calibration).

---

## Key Lessons Learned

| Lesson | Impact |
|--------|--------|
| **Architecture selection matters** | SAGEConv → R-GCN improved PR-AUC by +0.254 |
| **Literature-informed tuning** | Paper-based defaults before exhaustive search saved time |
| **Small graphs prefer explicit relation modeling** | R-GCN outperformed attention-based HGT by 22.5% |
| **Metric selection critical** | PR-AUC essential for imbalanced classification |
| **Calibration essential** | ECE 0.00 enables business decision thresholds |

---

> See [methodology.md](methodology.md) for full data generation and pipeline documentation.
