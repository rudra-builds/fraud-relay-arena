# 🏁 Fraud Relay Arena

**Track:** Veritas AI International High School Hackathon 2026 – Track 4.2.2: Credit Card Fraud Detection (Round 1 Blind Evaluation)  

> ⚠️ **Disclaimer:** This project is a prototype and is not financial advice.

---

## 1. Problem Statement & Target User

### Problem Statement
Credit card fraud detection is a rare-event classification problem characterized by extreme class imbalance (~0.17% fraud in real-world banking streams). Standard classification metrics like raw accuracy are actively deceptive: a naive model that marks 100% of transactions as legitimate achieves 99.83% accuracy while missing every single fraud event. We treat fraud detection as a financial optimization problem evaluated strictly on **Precision-Recall AUC (PR-AUC)** and **dollar-denominated banking costs**.

### Target User & Exact Moment of Use
- **Specific User Persona:** *Elena Rostova, Senior Payment-Gateway Fraud Operations Analyst.*
- **Exact Moment of Use:** Elena uses the **RelayGuard Pit-Wall Dashboard** during real-time fraud escalation triage. When an incoming transaction triggers risk flags during the 30-second pre-settlement authorization window, Elena reviews the live inspector's automated reason codes (e.g., negative $V_{14}$/$V_{17}$ anomalies vs. benign $V_{12}$ readings) and compares the bank cost tradeoff (\$500 chargeback loss vs. \$10 customer friction) to make an informed, transparent card-freeze decision.

---

## 2. Architecture: 3-Lane Sprint Relay

We benchmark **three distinct, two-stage ML relay pipelines** against an official baseline, enforcing strict data separation between feature engineering (Leg 1) and classification (Leg 2):

| Lane | Strategy | Leg 1 (Feature Agent) | Leg 2 (Classifier Agent) |
|---|---|---|---|
| **Code 0** | Baseline | RobustScaler | LogisticRegression (`class_weight='balanced'`) |
| **Lane 1** | Speed & Velocity | Cyclical Time Encoding + Log-Amount | Weighted HistGradientBoosting (`sub-ms` inference) |
| **Lane 2** | Anomaly Hybrid | Unsupervised Isolation Forest Anomaly Scoring | Balanced Random Forest (90 trees) |
| **Lane 3** | Explainable Cost | Top-12 Correlation Pruning | Cost-Threshold Sweep HistGBDT |

### Key Engineering Guarantees
- **Strict Chronological Split:** Data is ordered strictly by `Time` and partitioned 80/20. Models train solely on historical transactions and are evaluated on unseen future transactions—preventing time-travel data leakage.
- **Relay Baton Handoff Contract:** The `@dataclass RelayBaton` contract guarantees dimension parity, checks for `NaN`/`Inf` corruptions, and generates a SHA-256 hash verifying that Leg 1 transformations were fitted exclusively on training data.
- **Track Official Scrutineering:** Python AST static analysis verifies zero test-data leakage (`fit`/`fit_transform` never called on test arrays), bans unsafe dynamic execution (`eval`, `exec`), and runs Bandit SAST for automated vulnerability scanning.

---

## 3. "Make It Your Own" — Banking Cost Optimization

Lane 3 directly addresses the rubric's "Make It Your Own" criterion by substituting generic statistical thresholds with banking loss minimization:

- **Asymmetric Cost Matrix:**
  $$\text{Total Bank Cost} = (\text{False Negatives} \times \$500) + (\text{False Positives} \times \$10)$$
  - Missed fraud (\$500): Chargeback loss, merchant reimbursement, recovery overhead.
  - False alarm (\$10): Customer friction, notification SMS, manual review overhead.
- **Validation-Slice Threshold Sweep:** Evaluates 17 decision thresholds strictly on the final 15% validation slice of the training partition (never touching the test set) to identify the global cost-minimizing threshold.
- **Human-Readable Reason Codes:** Translates latent PCA feature deviations into natural-language explanations for operations analysts.

---

## 4. How to Run

### Quick Start (Standard Execution)
The pipeline automatically generates a deterministic 25,000-row synthetic dataset if `data/creditcard.csv` is not present, so you can run it immediately after cloning:

```bash
git clone https://github.com/rudra-builds/fraud-relay-arena.git
cd fraud-relay-arena
pip install -r requirements.txt

# Run the 3-Lane Sprint Relay benchmark & AST audit
python3 .agents/skills/fraud-relay-auditor/scripts/scrutineer.py

# Launch the interactive pit-wall dashboard
streamlit run app.py
```

### Full Kaggle Dataset Execution (284,807 Transactions)
```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
python3 .agents/skills/fraud-relay-auditor/scripts/scrutineer.py
streamlit run app.py
```

### Running with Custom Datasets & CLI Options
`scrutineer.py` accepts custom transaction schemas and cost structures:

```bash
python3 .agents/skills/fraud-relay-auditor/scripts/scrutineer.py \
  --data path/to/my_transactions.csv \
  --target is_fraud \
  --time timestamp \
  --fraud-cost 750.0 \
  --fp-cost 15.0
```
- If `--time` is omitted or not found in the CSV, the pipeline maintains the existing row order for the 80/20 chronological split without crashing.
- Custom datasets can also be uploaded directly through the Streamlit sidebar UI (`Upload Custom CSV` + `Run Relay Benchmark`).

### Universal Agent Skill Compatibility (OpenAI Codex · Claude Code · Antigravity)

The `fraud-relay-auditor` skill adheres to the Agent Skills standard and is natively discoverable across major agent environments without duplicating files:

- **OpenAI Codex CLI:** Automatically reads repository skill roots (`.agents/skills/`). Trigger directly:  
  `$fraud-relay-auditor` or prompt *“Run the fraud-relay-auditor skill on data/creditcard.csv”*.
- **Claude Code:** Discovered via the `.claude/skills/fraud-relay-auditor` relative symlink. Trigger using:  
  `/fraud-relay-auditor` or prompt *“Use the fraud-relay-auditor skill to benchmark the 3 lanes.”*
- **Google Antigravity:** Discovered via `.agents/skills/fraud-relay-auditor/SKILL.md`. Trigger naturally:  
  *“Audit the credit card fraud relay pipelines and launch the pit wall.”*

All runtimes share the exact same authoritative `SKILL.md` runbook and `scripts/` runtime.

---

## 5. Benchmark Results & Honest Baselines

Generated on the full **284,807-transaction Kaggle dataset** with an untouched **80/20 chronological split** (Train: 227,845 transactions, 417 frauds | Test: 56,962 transactions, 75 frauds):

| Pipeline / Lane | PR-AUC | Precision | Recall | Bank Cost (\$) | Latency (µs/tx) | Composite Score |
|---|---|---|---|---|---|---|
| **Naive Baseline (Always Legitimate)** | 0.0013 | 0.0000 | 0.0000 | \$37,500.00 | 0.000 | 0.00 |
| **Baseline (Code 0: Balanced LogisticReg)** | 0.7841 | 0.0708 | 0.8933 | \$12,790.00 | 0.009 | 55.28 |
| **Lane 1: Velocity & HistGBDT** | 0.7964 | 0.8889 | 0.7467 | \$9,570.00 | 0.449 | 64.65 |
| **Lane 2: Anomaly + Forest Hybrid** | 0.8079 | 0.9818 | 0.7200 | \$10,510.00 | 0.443 | 62.59 |
| **Lane 3: Pruned Explainable Cost-Tuner** | 0.7921 | 0.5882 | 0.8000 | \$7,920.00 | 0.554 | 68.97 |

```
Composite Score = (45% × PR-AUC) + (35% × Cost Reduction vs Baseline) + (20% × Latency SLA ≤ 5.0 µs/tx)
Track Security Audit: ✅ scrutineer (18 functions verified) | ✅ baton_contract (3 functions verified)
```

### Why the Baselines Fail & Why the Relay Lanes Win
1. **The Trap of the Naive Baseline:** The naive "always predict legitimate" baseline hits **99.87% accuracy**, but fails catastrophically in banking reality: **0.0013 PR-AUC, 0% recall, and \$37,500.00 in unstopped fraud losses**.
2. **The Flaw of Symmetric Logistic Regression (Code 0):** Balanced Logistic Regression achieves high recall (89.33%), but floods the desk with **879 false alarms** (only 7.08% precision). At \$10 per customer block, that generates \$8,790 in customer friction overhead alone, costing \$12,790.00 total.
3. **How the Relay Lanes Genuinely Outperform:**
   - **Lane 1 (Velocity & HistGBDT):** Slashes false alarms from 879 to just 7 (88.89% precision) while preserving 0.7964 PR-AUC and cutting bank cost to \$9,570.00.
   - **Lane 2 (Anomaly Hybrid):** Highest PR-AUC (**0.8079**) and **98.18% precision** (only 1 false alarm out of 56,887 legitimate transactions).
   - **Lane 3 (Cost-Tuned Winner):** Tunes the decision threshold to $T = 0.35$ for the asymmetric \$500/\$10 loss matrix, achieving the lowest total bank loss of **\$7,920.00** (**38.1% savings vs Code 0, and 78.9% savings vs Naive!**).
- **Sub-Microsecond Speed:** All four pipelines achieve inference latency between **0.009 µs and 0.554 µs per transaction**, well within payment switch SLAs (< 5.0 µs).

### Core System Novelty (Beyond "Three Models")
Judges emphasize that running algorithms in isolation is trivial. RelayGuard’s technical novelty lies in:
1. **Cost-Calibrated Decision Threshold ($T = 0.35$):** Shifting decision boundaries from an arbitrary 0.50 cutoff to mathematically minimize asymmetric \$500 missed-fraud losses against \$10 false alarms.
2. **3-Tier Human-in-the-Loop Analyst Workflow:**
   - *Tier 1 (<15% risk):* Auto-approved (frictionless swipe).
   - *Tier 2 (15%–65% risk):* Routed to the fraud analyst’s review queue with plain-English reason codes.
   - *Tier 3 (>65% risk):* Automated instant hard block at the payment switch.
3. **Multivariate Reason Code Explanation Interface:** Decomposing multivariate PCA deviations ($V_{14}, V_{17}, V_{12}$) into plain-English flags (e.g. "anomalous positive V17 inversion + low amount suggests card-testing micro-charge").
4. **Programmatic Data Leakage Defense:** SHA-256 state handoffs via `RelayBaton` and automated Python `ast` syntax inspection ensuring zero `.fit()` calls on test data.

### Dataset & Synthetic Fallback Transparency
- **Primary Empirical Benchmark:** Evaluated on the full 284,807 transactions of the real Kaggle Credit Card Fraud Detection dataset using an untouched chronological 80/20 train/test split (227,845 train / 56,962 test).
- **Synthetic Fallback Disclosure:** `scrutineer.py` contains a built-in deterministic 25,000-row synthetic fallback generator. This fallback is **strictly a defensive resilience mechanism** so that judges or developers cloning the repo without immediate Kaggle credentials can run the pipeline without a crash. The benchmark numbers reported in this paper and dashboard are 100% computed on the real Kaggle dataset.

---

## 6. Honest Failure Analysis (Concrete Test Examples)

In accordance with Section 3 of the Veritas AI project guidelines, we highlight real mistakes our models make on the unseen Kaggle test set:

### Example 1: Missed Fraud / False Negative (Test Row 1882)
- **Ground Truth:** Fraud (`Class = 1`) | **Lane 1 Score:** `29.22%` | **Lane 3 Score:** `9.26%` (Threshold = `0.35`)
- **Transaction Amount:** **\$2.22** (A micro-charge!)
- **Feature Readings:** $V_{14} = -4.943$ (fraud-like), $V_{17} = +3.702$ (inverted/positive), $V_{12} = -0.741$ (near-zero), $V_{10} = -0.628$.
- **Why the Model Failed:** This transaction represents a real-world card-testing micro-charge of just **\$2.22**. The fraudster charged a nominal fee to verify the stolen card's validity. While $V_{14}$ exhibited a classic negative fraud deviation ($-4.943$), $V_{17}$ was anomalous in the positive direction ($+3.702$), and $V_{12}$ remained near zero. The tree-based ensembles routed this conjunction down majority-legitimate branches because of the tiny amount and mixed principal components. Under our cost matrix, this escaped fraud cost the bank **\$500.00**.

### Example 2: False Alarm / False Positive (Test Row 660)
- **Ground Truth:** Legitimate (`Class = 0`) | **Lane 1 Score:** `74.78%` | **Lane 3 Score:** `85.36%`
- **Transaction Amount:** **\$0.77**
- **Feature Readings:** $V_{14} = -7.537$, $V_{17} = +5.609$, $V_{12} = -2.037$, $V_4 = +3.586$.
- **Why the Model Failed:** This legitimate cardholder had a transaction with extreme PCA variance: $V_{14}$ dropped to $-7.537$ and $V_{12}$ dropped to $-2.037$, heavily overlapping with the multivariate fraud centroid. Both Lane 1 and Lane 3 triggered false alerts, costing **\$10.00** in cardholder friction and SMS verification overhead.

---

## 7. What We'd Do Next

If given more time and computational resources, we would expand RelayGuard along four architectural axes:

1. **Sequential Transaction Graph / Temporal RNNs:** Current tabular models inspect transactions as isolated rows. Real card fraudsters generate bursts of velocity across multiple terminals. We would implement an LSTM or Graph Neural Network (GNN) over the preceding 5-minute sliding window per cardholder to catch micro-charges before secondary high-value purchases occur.
2. **Dynamic MCC-Stratified Cost Calibration:** Different merchant categories exhibit radically different chargeback dynamics (e.g., airline tickets vs. digital gaming). We would calibrate merchant-category-code (MCC) specific loss functions rather than a uniform \$500/\$10 matrix.
3. **Ultra-Low-Latency ONNX Edge Engine:** Export trained HistGBDT and Isolation Forest models to ONNX Runtime with quantized C++ bindings to reduce inference latency from $0.66\mu\text{s}$ to sub-$50\text{ns}$ for direct in-memory payment-switch deployment.
4. **Active Learning & Tier-2 Human Triage Queue:** Route transactions falling in the indeterminate band ($0.35 \le p \le 0.65$) to a human-in-the-loop escalation queue, feeding resolved analyst decisions back into continuous retraining.

---

## 8. Project Structure

```
fraud-relay-arena/
├── .agents/skills/fraud-relay-auditor/
│   ├── SKILL.md                     # Authoritative Agent skill specification
│   └── scripts/
│       ├── baton_contract.py        # Authoritative RelayBaton contract & splitter
│       └── scrutineer.py            # 3-Lane sprint orchestrator & AST auditor
├── .claude/skills/
│   └── fraud-relay-auditor          # Relative symlink -> ../../.agents/skills/fraud-relay-auditor
├── data/
│   ├── .gitkeep                     # Preserves data directory structure
│   └── creditcard.csv               # Kaggle dataset (gitignored)
├── lanes/
│   └── __init__.py                  # Package namespace
├── app.py                           # Streamlit pit-wall dashboard
├── requirements.txt                 # Pinned dependencies
├── .gitignore                       # Git ignore rules
└── README.md                        # Documentation & write-up
```

---

## 9. AI Tools Citation & References

### AI Tools Used
- **Google Antigravity IDE:** Used as an AI pair programmer for multi-agent architecture design, AST leakage-visitor implementation, Streamlit pit-wall dashboard layout, and documentation structuring in accordance with Section 2 (Rule 4) of the Veritas AI Hackathon guidelines.

### Direct Library & Dataset Links
- **Dataset:** [Kaggle Credit Card Fraud Detection (ULB Machine Learning Group)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Machine Learning:** [scikit-learn Documentation](https://scikit-learn.org/)
- **Dashboard UI:** [Streamlit Framework](https://streamlit.io/)
- **Security Auditing:** [Bandit Python SAST (PyCQA)](https://bandit.readthedocs.io/)
- **Data & Visualization:** [Plotly Express](https://plotly.com/python/), [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Competition Reference:** Veritas AI International High School Hackathon 2026 Project Briefs (Track 4.2.2)

## 10. Limitations & Financial Disclaimer

### Known Technical Limitations
1. **Isolated Tabular Perspective:** Current models inspect swipes as independent rows without sequential customer history across hours.
2. **Uniform Cost Matrix:** In real banking, airline chargebacks differ from coffee micro-charges; our uniform \$500 / \$10 matrix models average risk rather than MCC-tiered risk.
3. **Synthetic Fallback Context:** While our reported benchmark is 100% computed on the full 284,807 Kaggle dataset, offline cold-clones fall back to a 25,000-row synthetic generator until Kaggle data is pulled.

> [!IMPORTANT]
> **Mandatory Hackathon Disclaimer:** This project is an experimental research prototype developed for the Veritas AI Hackathon Track 4.2.2 and is **not financial advice** or production-ready banking software.

---

*Built with ❤️ for Veritas AI Track 4.2.2 · "This project is a prototype and is not financial advice."*
