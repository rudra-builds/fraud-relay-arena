#!/usr/bin/env python3
"""
app.py — Streamlit Pit-Wall Dashboard for RelayGuard
Launch: streamlit run app.py
"""

import json, math
from pathlib import Path
import numpy as np, pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="🏁 RelayGuard", page_icon="🏁", layout="wide")

@st.cache_data
def load():
    p = Path("relay_telemetry.json")
    return json.loads(p.read_text()) if p.exists() else None

data = load()
if data is None:
    st.error("**relay_telemetry.json not found.** Run `python scrutineer.py` first.")
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
st.sidebar.markdown(f"**Team:** Rudra Sharma  \n"
                     f"**Track:** {meta.get('track','?')}  \n"
                     f"**Train:** {train_disp}  \n"
                     f"**Test:** {test_disp}  \n"
                     f"**Fraud (train):** {fraud_tr}%  \n"
                     f"**Fraud (test):** {fraud_te}%")

st.title("🏁 RelayGuard — Trackside Pit Wall")
all_r = [data["baseline"]] + data["lanes"]

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
                 use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(df, x="Lane", y="PR-AUC", color="Lane", text="PR-AUC",
                     title="PR-AUC (↑ better)")
        fig.update_traces(textposition="outside", texttemplate="%{text:.4f}")
        fig.update_layout(showlegend=False, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(df, x="Lane", y="Bank Cost ($)", color="Lane", text="Bank Cost ($)",
                     title="Bank Cost (↓ better)")
        fig.update_traces(textposition="outside", texttemplate="$%{text:,.0f}")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("⏱️ Split Times (ms)")
    trows = []
    for r in all_r:
        trows.append({"Lane": r["lane"], "Phase": "Leg 1", "ms": r.get("leg1_ms", r.get("leg1_time_ms", 0.0))})
        trows.append({"Lane": r["lane"], "Phase": "Leg 2", "ms": r.get("leg2_ms", 0.0)})
        trows.append({"Lane": r["lane"], "Phase": "Inference", "ms": r.get("inference_ms", r.get("inference_ms_total", 0.0))})
    st.plotly_chart(px.bar(pd.DataFrame(trows), x="Lane", y="ms", color="Phase",
                           barmode="stack", title="Relay Leg Timings"),
                    use_container_width=True)

    st.subheader("🔢 Confusion Matrices")
    cols = st.columns(len(all_r))
    for i, r in enumerate(all_r):
        with cols[i]:
            cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
            fig = px.imshow(cm, x=["Legit","Fraud"], y=["Legit","Fraud"],
                            text_auto=True, color_continuous_scale="Blues",
                            title=r["lane"][:22])
            fig.update_layout(height=280, margin=dict(l=5,r=5,t=35,b=5))
            st.plotly_chart(fig, use_container_width=True)


# ── TAB 2: LIVE FRAUD INSPECTOR ─────────────────────────────────────
with tab2:
    st.subheader("🔍 Live Transaction Fraud Inspector")
    st.markdown("Adjust sliders to simulate a transaction (Lane 3 explainability).")
    ca, cb = st.columns(2)
    with ca:
        amt = st.slider("💰 Amount ($)", 0.0, 5000.0, 122.5, 0.5)
        v14 = st.slider("V14 (strongest fraud signal)", -10.0, 10.0, 0.0, 0.1)
        v17 = st.slider("V17", -10.0, 10.0, 0.0, 0.1)
    with cb:
        v12 = st.slider("V12", -10.0, 10.0, 0.0, 0.1)
        v10 = st.slider("V10", -10.0, 10.0, 0.0, 0.1)
        v4  = st.slider("V4", -10.0, 10.0, 0.0, 0.1)

    score = -1.8 * v14 - 1.5 * v17 - 1.2 * v12 + 0.002 * amt
    reasons = []
    if v14 < -2: reasons.append(f"🔴 **V14 = {v14:.1f}** strongly negative (fraud pattern)")
    if v17 < -1.5: reasons.append(f"🔴 **V17 = {v17:.1f}** below normal")
    if v12 < -2: reasons.append(f"🟡 **V12 = {v12:.1f}** unusual")
    if amt > 1000: reasons.append(f"🟡 **Amount = ${amt:.2f}** high-value")
    if not reasons: reasons.append("🟢 All features within normal range")

    l3 = data["lanes"][2] if len(data["lanes"]) >= 3 else data["baseline"]
    thr = l3.get("threshold", 0.5)
    prob = 1 / (1 + math.exp(-score + 3))

    st.divider()
    if prob >= thr:
        st.error(f"## 🚨 FRAUD FLAGGED  (Score: {prob:.2%} ≥ Threshold: {thr:.0%})")
    else:
        st.success(f"## ✅ APPROVED  (Score: {prob:.2%} < Threshold: {thr:.0%})")

    st.markdown("### 📋 Reason Codes")
    for r in reasons: st.markdown(f"- {r}")
    st.caption(f"Cost if missed fraud: **\\$500** · Cost if false block: **\\$10**")


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
                use_container_width=True, hide_index=True)
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
                use_container_width=True, hide_index=True)
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
        st.plotly_chart(fig, use_container_width=True)


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
    st.markdown(r"**Relay Score** $= 100 \times \text{PR-AUC}"
                r" - 0.05 \times \text{Latency}_\text{ms}"
                r" - 10 \times \text{Security Warnings}$")
    sc_rows = [{"Lane": r["lane"], "Relay Score": r["relay_score"]}
               for r in all_r if "relay_score" in r]
    if sc_rows:
        fig = px.bar(pd.DataFrame(sc_rows), x="Lane", y="Relay Score",
                     color="Lane", text="Relay Score", title="Composite Relay Scores")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

st.sidebar.divider()
st.sidebar.caption("Team: Rudra Sharma · Veritas AI 2026 · Track 4.2.2")
st.sidebar.caption("Prototype only · Not financial advice")
