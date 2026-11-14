# 🏁 Fraud Relay Arena

**Veritas AI International High School Hackathon 2026 – Track 4.2.2: Credit Card Fraud Detection**

---

## 1. Problem Statement

Credit card fraud is a rare-event classification problem with severe class imbalance (~0.17% fraud). Traditional accuracy metrics are misleading — a model that predicts "legitimate" for every transaction achieves 99.83% accuracy while catching zero fraud. We use **Precision-Recall AUC** as the primary metric and optimize for **real banking costs**.

## 2. Approach: 3-Lane Sprint Relay Architecture

We benchmark **three independent ML pipelines** against an official baseline, each structured as a 2-agent relay race:

| Lane | Strategy | Leg 1 (Feature Agent) | Leg 2 (Classifier Agent) |
|------|----------|-----------------------|--------------------------|
| **Code 0** | Baseline | RobustScaler | LogisticRegression (balanced) |
| **Lane 1** | Speed & Velocity | Cyclical Time + Log Amount | Weighted HistGradientBoosting |
| **Lane 2** | Anomaly Hybrid | IsolationForest Scores | Balanced RandomForest |
| **Lane 3** | Explainable Cost | Top-12 Correlation Pruning | Cost-Threshold HistGBDT |

### Key Design Decisions

- **Chronological Split**: Data is sorted by `Time` and split 80/20 so the model trains on past transactions and is tested on future ones — never the reverse.
- **Baton Handoff Contract**: A `@dataclass` enforces dimension matching, NaN checks, and SHA-256 hashing between feature engineering and classification stages.
- **Track Official Scrutineering**: Python AST analysis verifies zero test-data leakage; Bandit SAST catches security vulnerabilities.

## 3. "Make It Your Own" — Banking Cost Optimization

Lane 3 directly addresses the rubric's "Make It Your Own" requirement:

- **Cost Matrix**: \$500 per missed fraud (false negative) vs. \$10 per false alarm (false positive)
- **Threshold Sweep**: Optimizes the decision boundary on a training validation slice to minimize total dollar loss, not just classification error
- **Plain-English Reason Codes**: Outputs analyst-readable explanations of why a transaction was flagged

## 4. How to Run

### Quick Start (No Kaggle CSV Needed)
```bash
pip install -r requirements.txt
python .agents/skills/fraud-relay-auditor/scripts/scrutineer.py
streamlit run app.py
```

### Full Dataset
```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
python .agents/skills/fraud-relay-auditor/scripts/scrutineer.py
streamlit run app.py
```

## 5. Results

Run `scrutineer.py` to generate the results table. Sample output:

```
Lane                                   PR-AUC   Prec    Recall  Cost($)    µs/tx    Score
────────────────────────────────────────────────────────────────────────────────────────────
Baseline (Code 0)                      0.XXXX   0.XXXX  0.XXXX    XXX.XX   X.XXX    XX.XX
Lane 1: Velocity & HistGBDT            0.XXXX   0.XXXX  0.XXXX    XXX.XX   X.XXX    XX.XX
Lane 2: Anomaly + Forest Hybrid        0.XXXX   0.XXXX  0.XXXX    XXX.XX   X.XXX    XX.XX
Lane 3: Pruned Explainable Cost-Tuner  0.XXXX   0.XXXX  0.XXXX    XXX.XX   X.XXX    XX.XX
```

## 6. Project Structure

```
fraud-relay-arena/
├── .agents/skills/fraud-relay-auditor/
│   ├── SKILL.md                     # Agent skill definition
│   └── scripts/
│       ├── baton_contract.py        # RelayBaton dataclass + data splitter
│       └── scrutineer.py            # Full relay orchestrator + audit
├── data/
│   ├── .gitkeep                     # Preserves data directory
│   └── creditcard.csv               # Kaggle dataset (gitignored)
├── lanes/
│   └── __init__.py                  # Package namespace
├── app.py                           # Streamlit dashboard
├── scrutineer.py                    # Root entry point for relay orchestrator
├── baton_contract.py                # Root entry point for baton contract
├── requirements.txt                 # Dependencies
├── .gitignore                       # Git ignore rules
└── README.md
```

## 7. Ethical Considerations

- **False Positives Impact Real People**: Every wrongly blocked card causes customer frustration and potential embarrassment. Our Honest Failure Analysis tab makes these costs visible.
- **Transparency**: We show exactly where our models fail, not just where they succeed.
- **No Accuracy Theater**: We never report misleading accuracy on imbalanced data.

## 8. References

- Kaggle Credit Card Fraud Dataset (ULB Machine Learning Group)
- Veritas AI International High School Hackathon 2026 Project Briefs
- scikit-learn documentation for all ML algorithms used

---

*Built for Veritas AI · Track 4.2.2 · Credit Card Fraud Detection*
