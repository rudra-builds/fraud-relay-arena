# Veritas AI Hackathon 2026 — Track 4.2.2 Project Report

**Project Title:** RelayGuard: Cost-Calibrated Triage Pipelines for Credit Card Fraud  
**Track:** 4.2.2 – Credit Card Fraud Detection (Round 1 Blind Review)  
**Code Repository:** [github.com/rudra-builds/fraud-relay-arena](https://github.com/rudra-builds/fraud-relay-arena)  

---

## 1. Project Selection & "What We Added"

### Suggested Project Chosen
We selected the **Credit Card Fraud Detection** track (Track 4.2.2), based on the Kaggle ULB dataset of 284,807 transactions.

### What We Added (Beyond the Standard Prompt)
1. **Asymmetric Cost Matrix & Fair Threshold Tuning:** Instead of assuming symmetric error costs, we modeled real banking economics ($500 per missed fraud vs. $10 per false-alarm card freeze). We tuned decision thresholds ($T$) for both baseline and tree models on a validation slice rather than using an arbitrary 0.50 cutoff.
2. **Dual Baseline Benchmarking:** We benchmarked models against both a **Naive "Always Legitimate" Baseline** (exposing the 99.87% accuracy trap) and the official **Balanced Logistic Regression Baseline**.
3. **3-Tier Analyst Triage Funnel:** Created a human-in-the-loop workflow: Tier 1 Auto-Pass (<15%), Tier 2 Analyst Review Queue with Reason Codes (15%–65%), and Tier 3 Auto-Block (≥65%).
4. **2-Day Temporal Concept Drift Analysis (Own Question):** Created a reproducible experiment (`scripts/reproduce_drift.py`) comparing Day 1 training vs. Day 2 evaluation to study feature drift across the 48-hour recording window.
5. **Architectural Guardrails:** Enforced strict chronological splitting (no shuffling) verified by cryptographic SHA-256 state handoffs and automated Python AST syntax audits to guarantee zero test-set leakage.

---

## 2. Methodology & Leakage Prevention

- **Dataset:** 284,807 transactions (492 frauds, 0.17% prevalence).
- **Split:** Strictly chronological based on `Time`: **80% train (227,845 txs, 417 frauds)** and **20% test (56,962 txs, 75 frauds)**.
- **Reproducibility & Fallback:** A realistic synthetic generator is included so the repository never crashes on cold clones. Primary benchmark telemetry was 100% evaluated on the Kaggle dataset.

---

## 3. Results & Honest Statistical Analysis

Evaluated on the 56,962 unseen test transactions (75 frauds):

| Pipeline / Model | Threshold ($T$) | PR-AUC | Precision | Recall | Bank Cost ($) | Inference (µs/tx) |
|---|---|---|---|---|---|---|
| **Naive Baseline (Always Legitimate)** | 0.50 | 0.0013 | 0.0000 | 0.0000 | $37,500.00 | 0.000 |
| **Code 0: Logistic Regression (Default)** | 0.50 | 0.7841 | 0.0708 | **0.8933** | $12,790.00 | **0.014** |
| **Code 0: Logistic Regression (Cost-Tuned)** | 0.90 | 0.7841 | 0.2667 | 0.8533 | **$7,260.00** | **0.014** |
| **Lane 1: Velocity & HistGBDT** | 0.50 | 0.7964 | 0.8889 | 0.7467 | $9,570.00 | 0.388 |
| **Lane 2: Anomaly + Forest Hybrid** | 0.50 | **0.8079** | **0.9818** | 0.7200 | $10,510.00 | 0.442 |
| **Lane 3: Pruned HistGBDT (Cost-Tuned)** | **0.35** | 0.7921 | 0.5882 | 0.8000 | **$7,920.00** | 0.328 |

> **Honest Statistical Note:** With only 75 frauds in the test slice, differences between PR-AUC 0.79 and 0.81 fall within sampling noise. The primary operational gain is not algorithmic supremacy, but moving from default symmetric thresholds to cost-calibrated boundaries ($35%–90%), which cuts banking losses by over 38% while reducing false alarms.

---

## 4. "Own Question": 2-Day Temporal Concept Drift

The dataset spans 48 hours (Day 1: 144,786 txs; Day 2: 140,021 txs). Reproducible via `python3 scripts/reproduce_drift.py`:
- **Balanced Logistic Regression:** Maintained **0.7757 PR-AUC** (linear boundaries generalized stably across PCA distributions).
- **HistGradientBoosting:** Dropped from **~0.80 to 0.4924 PR-AUC** when trained purely on Day 1.
- **Takeaway:** Complex tree ensembles overfit to Day 1's local PCA split points. In production, payment gateways require daily online retraining or regularized linear fallbacks.

---

## 5. Honest Failure Analysis & Hypotheses

Because features $V_1$–$V_{28}$ are anonymized PCA projections, all behavioral interpretations are **analyst hypotheses**:
1. **Missed Fraud (Test Row #1882):** A **$2.22** micro-charge. While $V_{14}$ showed a negative deviation ($-4.943$), $V_{17}$ was anomalous in the positive direction ($+3.702$). We hypothesize this represents a card-testing micro-charge designed to evade negative-centroid detection.
2. **False Alarm (Test Row #660):** A benign **$0.77** transaction with extreme PCA variance ($V_{14} = -7.537$) overlapping the fraud distribution.

---

## 6. AI Tools Used & Mandatory Disclaimer

### AI Tools Citation
In compliance with Rule 4:
- **Google Antigravity IDE (Gemini 2.5 Flash / Pro):** Used for AST leakage-visitor implementation, Streamlit layout, and code modularization.
- **Anthropic Claude Code & OpenAI Codex CLI:** Used to test cross-agent compatibility of the Agent Skill.
- **Adobe Podcast AI:** Used for voice audio post-processing.
*Every line of code and mathematical calculation was verified and can be fully explained by the author.*

> [!IMPORTANT]
> **Mandatory Disclaimer:** This project is a prototype and is not financial advice.
