---
name: fraud-relay-auditor
description: >
  Executes a 3-lane, 6-agent Sprint Relay for Credit Card Fraud Detection
  (Veritas AI Track 4.2.2). Benchmarks three two-stage ML pipelines against a
  chronological LogisticRegression baseline, audits code for AST data leakage
  and Bandit security flaws, and outputs PR-AUC, latency telemetry, cost
  analysis, and plain-English fraud explanations.
---

# Fraud Relay Auditor Skill

> ⚠️ **Disclaimer:** This project is a prototype and is not financial advice.

## Overview

Orchestrates 6 specialised ML sub-agents across 3 parallel relay lanes to solve
highly imbalanced credit-card fraud detection while enforcing:

- **Zero test-set leakage** (chronological Time-based split, AST verification)
- **Sub-millisecond inference latency** (per-transaction µs benchmarks)
- **Static security verification** (Bandit SAST + banned-call detection)
- **Banking cost analysis** (\$500/missed fraud, \$10/false alarm)

## Architecture

```
Baseline (Code 0) ──► LogisticRegression(class_weight='balanced')
Lane 1 (Speed)    ──► Agent 1A (Temporal Features) → Agent 1B (HistGBDT)
Lane 2 (Hybrid)   ──► Agent 2A (IsolationForest)   → Agent 2B (RandomForest)
Lane 3 (Explain)  ──► Agent 3A (Feature Pruning)    → Agent 3B (Cost-Threshold)
                      ───────────── ▼ ─────────────
                   Track Official Scrutineering (AST + Bandit)
                      ───────────── ▼ ─────────────
                   Streamlit Pit-Wall Dashboard (app.py)
```

## Prerequisites

```bash
pip install -r requirements.txt
```

The pipeline auto-generates a 25,000-row synthetic dataset if
`data/creditcard.csv` is not present, so you can run it immediately.
For full results, download the Kaggle dataset:
```bash
# Requires kaggle CLI: pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
```

## Workflow Steps

### Step 1 – Run the 3-Lane Sprint Relay + Baseline

Execute the auditor pipeline using the authoritative script:

```bash
python3 .agents/skills/fraud-relay-auditor/scripts/scrutineer.py --data <csv_path> --target <target_col>
```

#### CLI Options
- `--data <path>`: Path to transaction CSV (default: `data/creditcard.csv`). Automatically generates a 25,000-row synthetic fallback if missing.
- `--target <col>`: Label column (default: `Class`).
- `--time <col>`: Timestamp column for chronological ordering (default: `Time`). If not present, maintains existing row order for 80/20 split without crashing.
- `--fraud-cost <float>`: Cost per missed fraud / false negative (default: `500.0`).
- `--fp-cost <float>`: Cost per false alarm / false positive (default: `10.0`).

#### Default Run
```bash
python3 .agents/skills/fraud-relay-auditor/scripts/scrutineer.py
```

This will:
1. Load and chronologically split the dataset (80% train / 20% test by `Time`)
2. Train the **Code 0** LogisticRegression baseline
3. Run **Lane 1** (Velocity + HistGBDT), **Lane 2** (IsolationForest + RandomForest),
   and **Lane 3** (Feature Pruning + Cost-Threshold) relay pipelines
4. Perform AST data-leakage inspection and Bandit security scanning
5. Save all metrics, confusion matrices, and failure cases to `relay_telemetry.json`

### Step 2 – Launch the Trackside Dashboard

```bash
streamlit run app.py
```

The dashboard provides:
- **🏆 Leaderboard** – PR-AUC, Precision, Recall, Bank Cost, Latency comparison
- **🔍 Live Inspector** – Interactive fraud scoring with plain-English reason codes
- **💀 Failure Analysis** – False negatives (missed frauds) and false positives
- **🛡️ Security Audit** – AST + Bandit certification badges

### Step 3 – Review Results

The console output prints a summary table you can copy directly into your
Veritas AI write-up. The `relay_telemetry.json` file contains all raw data
for further analysis.

## Trigger Conditions

Activate this skill when:
- Building or evaluating a credit-card fraud detection pipeline
- Comparing multiple ML approaches with baseline benchmarking
- Needing AST-level data-leakage verification for ML code
- Preparing a Veritas AI hackathon submission (Track 4.2.2)

## Output Files

| File | Description |
|------|-------------|
| `relay_telemetry.json` | Full metrics, confusion matrices, failure cases, audit results |
| `app.py` | Streamlit dashboard (reads telemetry JSON) |
