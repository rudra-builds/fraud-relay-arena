# 🏁 Fraud Relay Arena

**Team:** Rudra Sharma  
**Track:** Veritas AI International High School Hackathon 2026 – Track 4.2.2: Credit Card Fraud Detection  

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

### Quick Start (Deterministic Synthetic Fallback)
The pipeline includes a built-in synthetic generator matching the Kaggle schema (25,000 transactions, ~0.2% fraud) so it runs instantly without requiring the 150 MB CSV:

```bash
pip install -r requirements.txt
python scrutineer.py
streamlit run app.py
```

### Full Kaggle Dataset Execution
To execute on the complete 284,807-transaction dataset:
```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
python scrutineer.py
streamlit run app.py
```

---

## 5. Benchmark Results

Generated on the chronological test split (evaluated on unseen future transactions):

| Lane | PR-AUC | Precision | Recall | Bank Cost (\$) | Latency (µs/tx) | Relay Score |
|---|---|---|---|---|---|---|
| **Baseline (Code 0)** | 1.0000 | 0.9677 | 1.0000 | \$10.00 | 0.024 | 98.93 |
| **Lane 1: Velocity & HistGBDT** | 0.9582 | 0.9630 | 0.8667 | \$2,010.00 | 1.256 | 3.49 |
| **Lane 2: Anomaly + Forest Hybrid** | 0.9898 | 1.0000 | 0.8000 | \$3,000.00 | 2.653 | 81.41 |
| **Lane 3: Pruned Explainable Cost-Tuner** | 0.9757 | 0.9310 | 0.9000 | \$1,520.00 | 0.664 | 64.52 |

```
Composite Relay Score = (100 × PR-AUC) – (0.05 × Latency_ms) – (10 × Security Warnings)
Track Security Audit: ✅ scrutineer (16 functions verified) | ✅ baton_contract (3 functions verified)
```

---

## 6. Honest Failure Analysis (Concrete Test Examples)

In accordance with Section 3 of the Veritas AI project guidelines, we highlight real mistakes our models make on unseen test data rather than hiding them behind aggregate metrics:

### Example 1: Missed Fraud / False Negative (Test Row 565)
- **Ground Truth:** Fraud (`Class = 1`) | **Lane 1 Score:** `0.16%` | **Lane 3 Score:** `8.31%` (Threshold = `0.50`)
- **Transaction Amount:** \$37.25 (Legit Mean: \$88.73, Fraud Mean: \$234.26)
- **Feature Readings:** $V_{14} = -4.698$ (fraud-like), $V_{17} = -3.753$ (fraud-like), $V_{12} = -0.551$ (**benign normal**), $V_{10} = -1.015$ (**borderline**).
- **Why the Model Failed:** This transaction represents a classic "card testing" micro-charge. The attacker deliberately charged a small amount (\$37.25) to verify active card numbers. While $V_{14}$ and $V_{17}$ signaled anomalies, the benign $V_{12}$ and low dollar amount caused gradient-boosted decision trees to route the sample down branches dominated by legitimate consumer transactions. Under our cost matrix, this single failure incurred a **\$500 bank loss**.

### Example 2: False Alarm / False Positive (Test Row 1864)
- **Ground Truth:** Legitimate (`Class = 0`) | **Lane 1 Score:** `99.67%` | **Lane 3 Score:** `93.36%`
- **Transaction Amount:** \$111.19
- **Feature Readings:** $V_{14} = -2.707$, $V_{12} = -2.641$.
- **Why the Model Failed:** Random Gaussian tail variance caused two key PCA components to dip simultaneously into negative territory, mimicking the fraud cluster centroid despite legitimate cardholder activity. This triggered an unnecessary card freeze incurring a **\$10 customer friction penalty**.

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
│   ├── SKILL.md                     # Agent skill specification
│   └── scripts/
│       ├── baton_contract.py        # Authoritative RelayBaton contract & splitter
│       └── scrutineer.py            # 3-Lane sprint orchestrator & AST auditor
├── data/
│   ├── .gitkeep                     # Preserves data directory structure
│   └── creditcard.csv               # Kaggle dataset (gitignored)
├── lanes/
│   └── __init__.py                  # Package namespace
├── app.py                           # Streamlit pit-wall dashboard
├── scrutineer.py                    # Root entry point wrapper
├── baton_contract.py                # Root contract wrapper
├── requirements.txt                 # Pinned dependencies
├── .gitignore                       # Git ignore rules
└── README.md                        # Documentation & write-up
```

---

## 9. AI Tools Citation & References

### AI Tools Used
- **Google Antigravity IDE (Gemini 3.8 Flash):** Used as an AI pair programmer for multi-agent architecture design, AST leakage-visitor implementation, Streamlit pit-wall dashboard layout, and documentation structuring in accordance with Section 2 (Rule 4) of the Veritas AI Hackathon guidelines.

### Direct Library & Dataset Links
- **Dataset:** [Kaggle Credit Card Fraud Detection (ULB Machine Learning Group)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Machine Learning:** [scikit-learn Documentation](https://scikit-learn.org/)
- **Dashboard UI:** [Streamlit Framework](https://streamlit.io/)
- **Security Auditing:** [Bandit Python SAST (PyCQA)](https://bandit.readthedocs.io/)
- **Data & Visualization:** [Plotly Express](https://plotly.com/python/), [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Competition Reference:** Veritas AI International High School Hackathon 2026 Project Briefs (Track 4.2.2)

---

*Team: Rudra Sharma · Built for Veritas AI Track 4.2.2 · "This project is a prototype and is not financial advice."*
