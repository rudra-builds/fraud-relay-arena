# Veritas AI Hackathon 2026 — Track 4.2.2 Submission Report

**Project Title:** RelayGuard: Cost-Calibrated Decision Pipelines for Credit Card Fraud Triage  
**Participant:** Rudra Sharma  
**Track:** 4.2.2 – Credit Card Fraud Detection  
**Repository:** [github.com/rudra-builds/fraud-relay-arena](https://github.com/rudra-builds/fraud-relay-arena)  

---

## 1. Problem Definition & The Accuracy Trap

In credit card payment systems, fraud is rare (0.17% of transactions in our dataset) but financially damaging. Relying on raw accuracy creates a severe trap: a naive model that predicts every single transaction is legitimate achieves **99.87% accuracy**, yet it catches **0% of fraud** and results in **$37,500.00 in unstopped fraud losses**.

### End-User Persona & The Exact Triage Moment
Our target user is a **Tier-1 Payment-Gateway Fraud Analyst**. At the moment an authorization request hits the payment switch, our pipeline assigns transactions to one of three automated tiers:
1. **Tier 1 (Auto-Approve, Risk < 15%):** Frictionless approval for benign transactions.
2. **Tier 2 (Analyst Desk Triage, 15% ≤ Risk < 65%):** Routed to the analyst queue with automated, plain-English reason codes (e.g., *"Anomalous V17 inversion on a $2.22 micro-charge suggests card-testing fraud"*).
3. **Tier 3 (Auto-Block, Risk ≥ 65%):** Immediate termination of high-confidence syndicate attacks.

Decisions are governed by an **asymmetric cost matrix**:
- **Cost of a Missed Fraud (False Negative):** **$500.00** (average chargeback liability + processing fines).
- **Cost of a False Alarm (False Positive):** **$10.00** (customer friction, SMS verification, and support overhead).

---

## 2. Empirical Methodology & Zero-Leakage Architecture

- **Dataset:** Evaluated on the full 284,807 transactions of the Kaggle ULB Credit Card Fraud dataset (492 frauds).
- **Chronological Split:** Transactions are ordered strictly by the `Time` feature and split **80% train (227,845 rows, 417 frauds)** and **20% test (56,962 rows, 75 frauds)**. Shuffling is strictly prohibited to prevent lookahead data leakage.
- **Architectural Guardrails:** Feature transformations hand off states via a cryptographically hashed contract (`RelayBaton`). Code is audited via automated Python `ast` syntax-tree inspection to prove zero `.fit()` calls on test data, alongside Bandit SAST scans (0 security flaws).
- **Synthetic Fallback Disclosure:** For offline clones without immediate Kaggle credentials, a 25,000-row synthetic fallback generator is provided. All benchmark numbers below were computed on the real Kaggle dataset.

---

## 3. Benchmark Results & Baseline Comparisons

Evaluated on the unseen 56,962-transaction test set:

| Pipeline | PR-AUC | Precision | Recall | Bank Cost ($) | Latency (µs/tx) | Key Takeaway |
|---|---|---|---|---|---|---|
| **Naive Baseline (Always Legitimate)** | 0.0013 | 0.0000 | 0.0000 | $37,500.00 | 0.000 | 99.87% accuracy trap; misses all 75 frauds. |
| **Baseline Code 0 (Balanced LogisticReg)** | 0.7841 | 0.0708 | **0.8933** | $12,790.00 | 0.014 | Catches fraud but floods desk with 879 false alarms. |
| **Lane 1: Velocity & HistGBDT** | 0.7964 | 0.8889 | 0.7467 | $9,570.00 | 0.388 | Slashes false alarms from 879 to 7; saves $3,220. |
| **Lane 2: Anomaly + Forest Hybrid** | **0.8079** | **0.9818** | 0.7200 | $10,510.00 | 0.442 | **Precision Winner:** Only 1 false alarm out of 56,887 swipes. |
| **Lane 3: Pruned Cost-Tuner ($T=0.35$)** | 0.7921 | 0.5882 | 0.8000 | **$7,920.00** | 0.328 | **Bank Cost Winner:** Cuts losses by **38.1% vs Code 0** and **78.9% vs Naive**. |

*Score Formula:* $(45\% \times \text{PR-AUC}) + (35\% \times \text{Cost Reduction vs Baseline}) + (20\% \times \text{Latency SLA} \le 5\mu\text{s})$. All pipelines clear the payment gateway SLA (< 0.55 µs/swipe).

---

## 4. "Own Question" Exploration: 2-Day Temporal Concept Drift

The dataset spans 48 hours (172,800 seconds). We tested whether fraud patterns drift between **Day 1 ($0 \le \text{Time} < 86,400$)** and **Day 2 ($\text{Time} \ge 86,400$)**:
- **Day 1:** 144,786 transactions, 281 frauds (0.194% fraud rate, mean amount $88.58).
- **Day 2:** 140,021 transactions, 211 frauds (0.151% fraud rate, mean amount $84.33).

**Empirical Finding:** When trained strictly on Day 1 and evaluated on Day 2:
- Linear models maintained a steady **0.7757 PR-AUC** (hyperplanes remained robust across PCA components).
- Unregularized gradient boosting trees dropped from **~0.80 to 0.4924 PR-AUC**, overfitting to Day 1's local PCA splits.
- **System Insight:** Real-world payment gateways must utilize continuous daily updating or heavily regularized shallow trees to resist day-to-day concept drift.

---

## 5. Honest Failure Analysis (Unseen Test Set Mistakes)

In accordance with Section 3 of the project guidelines, we analyzed real errors:
1. **Missed Fraud / False Negative (Test Row #1882):** A real-world card-testing micro-charge of **$2.22**. While $V_{14}$ showed a negative fraud deviation ($-4.943$), $V_{17}$ was anomalous in the positive direction ($+3.702$), routing it down majority-legitimate tree branches. Cost: **$500.00**.
2. **False Alarm / False Positive (Test Row #660):** A legitimate transaction of **$0.77** with extreme PCA variance ($V_{14} = -7.537$, $V_{12} = -2.037$) that overlapped with the multivariate fraud centroid. Cost: **$10.00** in customer friction.

---

## 6. Limitations, AI Credits & Mandatory Disclaimer

### Limitations
- Models currently evaluate transactions as independent rows rather than constructing 1-hour sequential velocity graphs per cardholder.
- Uniform cost matrix ($500/$10) models average loss rather than merchant-category-specific chargeback risk.

### AI Tools Citation
- **Google Antigravity IDE:** Utilized as an AI pair programming environment for AST leakage-visitor implementation, Streamlit pit-wall dashboard layout, and documentation structuring under Section 2 (Rule 4) of the Hackathon guidelines.

> [!IMPORTANT]
> **Mandatory Disclaimer:** This project is a prototype and is not financial advice.
