#!/usr/bin/env python3
"""
app.py — Streamlit Pit-Wall Dashboard for RelayGuard
Launch: streamlit run app.py
"""

import importlib
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
import plotly.express as px
import streamlit as st

# Direct import from .agents/skills/fraud-relay-auditor/scripts/ (single source of truth)
_scripts_dir = Path(__file__).resolve().parent / ".agents" / "skills" / "fraud-relay-auditor" / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from scrutineer import run_pipeline

st.set_page_config(page_title="🏁 RelayGuard", page_icon="🏁", layout="wide")

@st.cache_data
def load():
    p = Path("relay_telemetry.json")
    return json.loads(p.read_text()) if p.exists() else None

data = load()
if data is None:
    st.title("🏁 RelayGuard — Trackside Pit Wall")
    st.warning("**relay_telemetry.json not found.** Run `python3 .agents/skills/fraud-relay-auditor/scripts/scrutineer.py` or click below.")
    if st.button("🚀 Run Relay Benchmark with Default Dataset", type="primary"):
        with st.spinner("Executing 3-Lane Sprint Relay..."):
            run_pipeline()
            st.cache_data.clear()
            st.rerun()
    st.stop()

meta = data.get("metadata", {})
train_rows = meta.get("train_rows", meta.get("dataset_rows_train", "?"))
test_rows = meta.get("test_rows", meta.get("dataset_rows_test", "?"))
fraud_tr = meta.get("fraud_pct_train", meta.get("fraud_rate_train_pct", "?"))
fraud_te = meta.get("fraud_pct_test", meta.get("fraud_rate_test_pct", "?"))

train_disp = f"{train_rows:,}" if isinstance(train_rows, (int, float)) else str(train_rows)
test_disp = f"{test_rows:,}" if isinstance(test_rows, (int, float)) else str(test_rows)

st.sidebar.title("🏁 RelayGuard")
st.sidebar.warning("⚠️ This project is a prototype and is not financial advice.")
st.sidebar.markdown(f"**Track:** {meta.get('track','Track 4.2.2')}  \n"
                     f"**Train:** {train_disp}  \n"
                     f"**Test:** {test_disp}  \n"
                     f"**Fraud (train):** {fraud_tr}%  \n"
                     f"**Fraud (test):** {fraud_te}%")

# ── Custom Dataset Uploader ──────────────────────────────────────────
st.sidebar.divider()
st.sidebar.subheader("📤 Custom Dataset Benchmark")
uploaded_file = st.sidebar.file_uploader("Upload Custom CSV", type=["csv"])
custom_target = st.sidebar.text_input("Target Column", value=meta.get("target_col", "Class"))
custom_time = st.sidebar.text_input("Time Column (optional)", value=meta.get("time_col", "Time"))

if uploaded_file is not None:
    if st.sidebar.button("🚀 Run Relay Benchmark", type="primary", width="stretch"):
        with st.spinner("Executing 3-Lane Sprint Relay on uploaded dataset..."):
            save_path = Path("data") / "uploaded_custom.csv"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(uploaded_file.getvalue())
            try:
                run_pipeline(
                    csv_path=str(save_path),
                    target_col=custom_target.strip(),
                    time_col=custom_time.strip(),
                )
                st.cache_data.clear()
                st.sidebar.success("✅ Benchmark completed successfully!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Benchmark error: {e}")

st.title("🏁 RelayGuard — Trackside Pit Wall")
naive_r = [data["naive_baseline"]] if "naive_baseline" in data else []
all_r = naive_r + [data["baseline"]] + data["lanes"]

tab1, tab2, tab3, tab4 = st.tabs(["🏆 Leaderboard", "🔍 Fraud Inspector",
                                   "💀 Failure Analysis", "🛡️ Security Audit"])

# ── TAB 1: LEADERBOARD ──────────────────────────────────────────────
with tab1:
    rows = []
    for r in all_r:
        rows.append({
            "Lane": r["lane"], "Leg 1": r.get("leg1_agent", r.get("leg1_agent_name", "Leg 1")),
            "Leg 2": r.get("leg2_agent", r.get("leg2_agent_name", "Leg 2")),
            "PR-AUC": r["pr_auc"], "Precision": r["precision"], "Recall": r["recall"],
            "Bank Cost ($)": r.get("bank_cost", r.get("bank_cost_usd", 0.0)), "µs/tx": r["us_per_tx"],
            "Threshold": r["threshold"], "Relay Score": r.get("relay_score", "—"),
            "TP": r["tp"], "FP": r["fp"], "FN": r["fn"], "TN": r["tn"],
        })
    df = pd.DataFrame(rows)
    st.dataframe(df.style
                 .highlight_max(subset=["PR-AUC","Precision","Recall","Relay Score"],
                                color="#2e7d32", props="color:white;font-weight:bold")
                 .highlight_min(subset=["Bank Cost ($)","µs/tx"],
                                color="#1565c0", props="color:white;font-weight:bold"),
                 width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(df, x="Lane", y="PR-AUC", color="Lane", text="PR-AUC",
                     title="PR-AUC (↑ better)")
        fig.update_traces(textposition="outside", texttemplate="%{text:.4f}")
        fig.update_layout(showlegend=False, yaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.bar(df, x="Lane", y="Bank Cost ($)", color="Lane", text="Bank Cost ($)",
                     title="Bank Cost (↓ better)")
        fig.update_traces(textposition="outside", texttemplate="$%{text:,.0f}")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

    st.subheader("⏱️ Split Times (ms)")
    trows = []
    for r in all_r:
        trows.append({"Lane": r["lane"], "Phase": "Leg 1", "ms": r.get("leg1_ms", r.get("leg1_time_ms", 0.0))})
        trows.append({"Lane": r["lane"], "Phase": "Leg 2", "ms": r.get("leg2_ms", 0.0)})
        trows.append({"Lane": r["lane"], "Phase": "Inference", "ms": r.get("inference_ms", r.get("inference_ms_total", 0.0))})
    st.plotly_chart(px.bar(pd.DataFrame(trows), x="Lane", y="ms", color="Phase",
                           barmode="stack", title="Relay Leg Timings"),
                    width="stretch")

    st.subheader("🔢 Confusion Matrices")
    cols = st.columns(len(all_r))
    for i, r in enumerate(all_r):
        with cols[i]:
            cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
            fig = px.imshow(cm, x=["Legit","Fraud"], y=["Legit","Fraud"],
                            text_auto=True, color_continuous_scale="Blues",
                            title=r["lane"][:22])
            fig.update_layout(height=280, margin=dict(l=5,r=5,t=35,b=5))
            st.plotly_chart(fig, width="stretch")


# ── TAB 2: LIVE FRAUD INSPECTOR ─────────────────────────────────────
with tab2:
    st.subheader("🔍 Live Transaction Fraud Inspector & Analyst Workflow")
    st.markdown(
        "Simulate incoming transactions flowing through RelayGuard's **3-Tier Decision Pipeline** "
        "and Lane 3's cost-calibrated threshold ($T = 0.35$)."
    )

    preset = st.selectbox(
        "⚡ Choose a Real Test Scenario (or select Custom Sliders)",
        [
            "Test Row #1882: Escaped Micro-Charge ($2.22 fraud, inverted V17 anomaly)",
            "Test Row #660: False Alarm Friction ($0.77 legitimate, extreme V14 variance)",
            "Real Caught Fraud: High-Value Syndicate ($1,480.00)",
            "Benign Everyday Purchase: Coffee ($4.75)",
            "Custom Sliders",
        ],
    )
    if preset.startswith("Test Row #1882"):
        d_amt, d_v14, d_v17, d_v12, d_v10, d_v4 = 2.22, -4.94, 3.70, -0.74, -0.63, 0.45
    elif preset.startswith("Test Row #660"):
        d_amt, d_v14, d_v17, d_v12, d_v10, d_v4 = 0.77, -7.54, 5.61, -2.04, -1.82, 3.59
    elif preset.startswith("Real Caught Fraud"):
        d_amt, d_v14, d_v17, d_v12, d_v10, d_v4 = 1480.00, -6.80, -4.50, -3.20, -2.90, 4.20
    elif preset.startswith("Benign Everyday"):
        d_amt, d_v14, d_v17, d_v12, d_v10, d_v4 = 4.75, 0.20, -0.10, 0.10, 0.05, -0.20
    else:
        d_amt, d_v14, d_v17, d_v12, d_v10, d_v4 = 122.50, 0.00, 0.00, 0.00, 0.00, 0.00

    ca, cb = st.columns(2)
    with ca:
        amt = st.slider("💰 Amount ($)", 0.0, 5000.0, float(d_amt), 0.5)
        v14 = st.slider("V14 (strongest fraud signal)", -10.0, 10.0, float(d_v14), 0.1)
        v17 = st.slider("V17", -10.0, 10.0, float(d_v17), 0.1)
    with cb:
        v12 = st.slider("V12", -10.0, 10.0, float(d_v12), 0.1)
        v10 = st.slider("V10", -10.0, 10.0, float(d_v10), 0.1)
        v4  = st.slider("V4", -10.0, 10.0, float(d_v4), 0.1)

    score = -1.8 * v14 - 1.5 * v17 - 1.2 * v12 + 0.002 * amt
    reasons = []
    if v14 < -2: reasons.append(f"🔴 **V14 = {v14:.1f}** strongly negative (fraud deviation)")
    if v17 < -1.5: reasons.append(f"🔴 **V17 = {v17:.1f}** anomalous negative deviation")
    if v17 > 2.5: reasons.append(f"⚠️ **V17 = {v17:.1f}** anomalous positive inversion (card-testing mask)")
    if v12 < -2: reasons.append(f"🟡 **V12 = {v12:.1f}** unusual transaction deviation")
    if amt > 1000: reasons.append(f"🟡 **Amount = ${amt:.2f}** high-value transaction risk")
    if amt < 3.0: reasons.append(f"ℹ️ **Amount = ${amt:.2f}** micro-charge (card-testing velocity pattern)")
    if not reasons: reasons.append("🟢 All features within normal bounds")

    l3 = data["lanes"][2] if len(data["lanes"]) >= 3 else data["baseline"]
    thr = l3.get("threshold", 0.35)
    prob = 1 / (1 + math.exp(-score + 3))

    st.divider()
    if prob >= 0.65:
        st.error(f"## 🛑 TIER 3: HARD BLOCK (Score: {prob:.1%} ≥ 65% Auto-Block Ceiling)")
        st.caption("Action: Transaction blocked automatically at payment switch. Fraud risk exceeds SLA threshold.")
    elif prob >= thr:
        st.warning(f"## ⚠️ TIER 2: ANALYST DESK TRIAGE (Score: {prob:.1%} ≥ {thr:.0%} Cost-Tuned Threshold)")
        st.caption("Action: Flagged for human review. Plain-English reason codes generated below for desk analyst.")
    else:
        st.success(f"## ✅ TIER 1: AUTO-APPROVED (Score: {prob:.1%} < {thr:.0%} Threshold)")
        st.caption("Action: Frictionless transaction approved without user disruption.")

    st.markdown("### 📋 Plain-English Reason Codes (Analyst Queue)")
    for r in reasons: st.markdown(f"- {r}")
    st.caption(f"Cost Matrix: Missed fraud = **\\$500** · False alarm card block = **\\$10**")


# ── TAB 3: FAILURE ANALYSIS ─────────────────────────────────────────
with tab3:
    st.subheader("💀 Honest Failure Analysis")
    st.markdown("Real mistakes on the unseen chronological test set.")
    sel = st.selectbox("Lane", [r["lane"] for r in all_r])
    sr = next(r for r in all_r if r["lane"] == sel)
    fails = sr.get("failures", {})
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🔴 False Negatives (Missed Frauds)")
        fn_l = fails.get("false_negatives", [])
        if fn_l:
            st.dataframe(pd.DataFrame(fn_l).rename(
                columns={"row":"Row","prob":"Probability","amount":"Amount ($)"}),
                width="stretch", hide_index=True)
            st.metric("💸 Cost (sample)", f"${len(fn_l)*500:,}",
                      delta=f"-{len(fn_l)} escaped", delta_color="inverse")
        else:
            st.success("All frauds caught!")
    with c2:
        st.markdown("### 🟡 False Positives (False Alarms)")
        fp_l = fails.get("false_positives", [])
        if fp_l:
            st.dataframe(pd.DataFrame(fp_l).rename(
                columns={"row":"Row","prob":"Probability","amount":"Amount ($)"}),
                width="stretch", hide_index=True)
            st.metric("😤 Friction (sample)", f"${len(fp_l)*10:,}",
                      delta=f"+{len(fp_l)} blocked", delta_color="inverse")
        else:
            st.success("Zero false alarms!")

    if "threshold_sweep" in sr:
        st.divider()
        sw = pd.DataFrame(sr["threshold_sweep"])
        fig = px.line(sw, x="threshold", y="cost_usd", markers=True,
                      title="Bank Cost vs. Decision Threshold",
                      labels={"threshold":"Threshold","cost_usd":"Cost ($)"})
        fig.add_vline(x=sr["threshold"], line_dash="dash", line_color="red",
                      annotation_text=f"Optimal T={sr['threshold']:.2f}")
        st.plotly_chart(fig, width="stretch")


# ── TAB 4: SECURITY AUDIT ───────────────────────────────────────────
with tab4:
    st.subheader("🛡️ Track Official Security Audit")
    for nm, a in data.get("security_audit", {}).items():
        ok = a["status"].startswith("✅")
        with st.expander(f"{'🟢' if ok else '🔴'} **{nm}** — {a['status']}", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Functions", a.get("functions_verified", a.get("modular_functions", 0)))
            banned_l = a.get("banned_calls", a.get("banned_dynamic_exec", []))
            c2.metric("Banned calls", len(banned_l))
            c3.metric("Bandit issues", a.get("bandit_issues", a.get("bandit_security_issues", 0)))
            leak = a.get("leakage", a.get("leakage_violations", []))
            if leak: st.error("Leakage: " + ", ".join(leak))
            b_detail = a.get("bandit_detail", a.get("bandit_details", []))
            if b_detail: st.info("Bandit: " + " · ".join(b_detail))

    st.divider()
    st.markdown(r"**Composite Score** $= 45\% \times \text{PR-AUC}"
                r" + 35\% \times \text{Cost Reduction vs Baseline}"
                r" + 20\% \times \text{Latency SLA } (\le 5.0\,\mu\text{s/tx})$")
    sc_rows = [{"Lane": r["lane"], "Relay Score": r["relay_score"]}
               for r in all_r if "relay_score" in r]
    if sc_rows:
        fig = px.bar(pd.DataFrame(sc_rows), x="Lane", y="Relay Score",
                     color="Lane", text="Relay Score", title="Composite Relay Scores")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

st.sidebar.divider()
st.sidebar.caption("Track 4.2.2 · Veritas AI 2026 · Blind Evaluation")
st.sidebar.caption("Prototype only · Not financial advice")
