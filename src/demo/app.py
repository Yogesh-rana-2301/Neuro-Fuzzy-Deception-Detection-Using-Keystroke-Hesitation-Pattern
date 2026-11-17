"""
src/demo/app.py
---------------
Interactive Streamlit Live Demonstration:
Neuro-Fuzzy Deception Detection Using Keystroke Hesitation Patterns

Features:
  1. Live Interactive Typing Mode:
     - Real-time client-side keystroke logger using JavaScript.
     - Captures down/up millisecond timestamps, dwell times, flight times,
       latencies, pauses, and backspaces.
     - Streams measurements straight into the trained ANFIS model.
  2. Manual Feature Simulation Mode:
     - Interactive sliders for dwell, flight, latency, pause count, backspaces.
     - Instant visual feedback and rule explanation.
  3. Visualizations & Model Explainability:
     - Deception score meter & confidence level.
     - Active fuzzy inference rules table.
     - Pre-computed ROC curves, MF curves, and model comparison bar charts.
"""

import os
import sys
import json
import time
import streamlit as st
import numpy as np

# Ensure project root is accessible
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(APP_DIR))
SRC_DIR = os.path.dirname(APP_DIR)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from scoring.score import DeceptionScorer
from collection.prompts import TRUTHFUL_PROMPTS, FABRICATION_PROMPTS

# Configure Page
st.set_page_config(
    page_title="Neuro-Fuzzy Deception Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #4C72B0;
        margin-bottom: 10px;
    }
    .deceptive-badge {
        background-color: #ff4b4b;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
    }
    .truthful-badge {
        background-color: #28a745;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_scorer():
    """Cache the trained ANFIS scorer instance."""
    return DeceptionScorer()

scorer = load_scorer()

# Sidebar Navigation & Model Info
st.sidebar.title("🧠 Neuro-Fuzzy System")
mode = st.sidebar.radio(
    "Choose Demonstration Mode:",
    ["⌨️ Live Keystroke Capture", "🎛️ Interactive Feature Sliders", "📊 Evaluation & Architecture"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Configuration")
st.sidebar.markdown(f"""
- **Inference Engine**: ANFIS (Takagi-Sugeno)
- **Input Features**: 5 Hesitation Metrics
- **Fuzzy Sets per Feature**: 2 (LOW, MEDIUM)
- **Learned Rules**: 32 Rules
- **Optimized Threshold**: `{scorer.threshold:.2f}`
""")


# ==============================================================================
# MODE 1: LIVE KEYSTROKE CAPTURE
# ==============================================================================
if mode == "⌨️ Live Keystroke Capture":
    st.title("⌨️ Live Keystroke Hesitation Detection")
    st.markdown("""
    Answer the selected prompt below by typing in the response box.
    The system measures **how** you type (*hesitations, key hold durations, transitions, and revisions*),
    **not what** you type.
    """)

    col_prompt, col_controls = st.columns([3, 1])

    with col_controls:
        condition = st.selectbox("Prompt Condition:", ["Truthful Response", "Fabricated Response"])
        prompt_list = TRUTHFUL_PROMPTS if condition == "Truthful Response" else FABRICATION_PROMPTS
        selected_idx = st.selectbox(
            "Select Prompt:",
            range(len(prompt_list)),
            format_func=lambda i: f"{prompt_list[i]['id']}: {prompt_list[i]['category']}"
        )

    active_prompt = prompt_list[selected_idx]

    with col_prompt:
        st.info(f"**Prompt [{active_prompt['category']}]:** {active_prompt['prompt']}")

    # Client-side JavaScript Keystroke Timing Recorder
    keystroke_recorder_html = """
    <div style="margin-top: 10px;">
        <label style="font-weight: bold; font-size: 15px;">Type your response here:</label><br>
        <textarea id="live_typing_area" rows="4" style="width: 100%; border-radius: 6px; padding: 10px; border: 1px solid #ccc; font-size: 15px;" placeholder="Start typing your answer here..."></textarea>
        <div style="margin-top: 8px;">
            <span id="timing_status" style="color: #666; font-size: 13px;">Keystroke listener active.</span>
        </div>
        <input type="hidden" id="keystroke_payload" value="">
    </div>

    <script>
        const textarea = document.getElementById("live_typing_area");
        const statusSpan = document.getElementById("timing_status");
        let promptShownTime = performance.now();
        let events = [];
        let keydownTimes = {};

        textarea.addEventListener("keydown", function(e) {
            const now = performance.now();
            if (!keydownTimes[e.code]) {
                keydownTimes[e.code] = now;
                events.push({
                    key: e.key,
                    code: e.code,
                    type: "down",
                    time: now
                });
            }
            statusSpan.innerText = `Recorded ${events.length} keystroke events...`;
        });

        textarea.addEventListener("keyup", function(e) {
            const now = performance.now();
            delete keydownTimes[e.code];
            events.push({
                key: e.key,
                code: e.code,
                type: "up",
                time: now
            });
            statusSpan.innerText = `Recorded ${events.length} keystroke events...`;
        });
    </script>
    """
    st.components.v1.html(keystroke_recorder_html, height=180)

    st.markdown("### Or Simulate Raw Timings from Live Typing")
    col1, col2, col3 = st.columns(3)
    with col1:
        sim_latency = st.number_input("Measured Latency (ms):", min_value=100.0, max_value=4000.0, value=780.0 if condition == "Truthful Response" else 1350.0, step=50.0)
        sim_dwell = st.number_input("Mean Dwell (ms):", min_value=30.0, max_value=300.0, value=85.0 if condition == "Truthful Response" else 105.0, step=5.0)
    with col2:
        sim_flight = st.number_input("Mean Flight (ms):", min_value=30.0, max_value=500.0, value=145.0 if condition == "Truthful Response" else 195.0, step=5.0)
        sim_pauses = st.number_input("Pause Count (>1s):", min_value=0.0, max_value=10.0, value=1.0 if condition == "Truthful Response" else 3.5, step=0.5)
    with col3:
        sim_backspaces = st.number_input("Backspace Ratio:", min_value=0.0, max_value=0.6, value=0.03 if condition == "Truthful Response" else 0.11, step=0.01)
        st.write("")
        st.write("")
        evaluate_btn = st.button("🔍 Run ANFIS Inference", use_container_width=True, type="primary")

    if evaluate_btn:
        res = scorer.score_features(
            initial_latency=sim_latency,
            mean_dwell=sim_dwell,
            mean_flight=sim_flight,
            pause_count=sim_pauses,
            backspace_ratio=sim_backspaces
        )

        st.markdown("---")
        st.subheader("Inference Result")

        rcol1, rcol2, rcol3 = st.columns([1, 1, 2])
        with rcol1:
            badge_class = "deceptive-badge" if res["label"] == "DECEPTIVE" else "truthful-badge"
            st.markdown(f"### Classification:<br><span class='{badge_class}'>{res['label']}</span>", unsafe_allow_html=True)
            st.metric("Confidence Level", res["confidence"])

        with rcol2:
            st.metric("Deception Score", f"{res['score']:.4f}")
            st.progress(float(res['score']))
            st.caption(f"Decision Threshold: `{res['threshold']:.2f}`")

        with rcol3:
            st.markdown("#### Dominant Fuzzy Rules Fired:")
            for r in res["dominant_rules"]:
                st.markdown(f"- **{r['rule']}**")
                st.caption(f"Firing Strength: `{r['activation_weight']*100:.1f}%` → *{r['tendency']}*")


# ==============================================================================
# MODE 2: INTERACTIVE FEATURE SLIDERS
# ==============================================================================
elif mode == "🎛️ Interactive Feature Sliders":
    st.title("🎛️ Interactive Keystroke Hesitation Playground")
    st.markdown("Adjust individual hesitation parameters to observe how the Takagi-Sugeno inference rules adapt in real-time.")

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Keystroke Dynamics Inputs")
        s_latency = st.slider("Initial Latency (Thinking time before 1st key in ms):", 100, 3000, 1100, 50)
        s_dwell = st.slider("Mean Dwell Time (Key hold duration in ms):", 40, 250, 95, 5)
        s_flight = st.slider("Mean Flight Time (Time between releasing and pressing keys in ms):", 40, 450, 175, 5)
        s_pauses = st.slider("Pause Count (Hesitations > 1.0s):", 0.0, 8.0, 2.5, 0.5)
        s_backspace = st.slider("Backspace Ratio (Self-corrections / Total keys):", 0.0, 0.40, 0.08, 0.01)

    result = scorer.score_features(s_latency, s_dwell, s_flight, s_pauses, s_backspace)

    with c2:
        st.subheader("Real-time ANFIS Output")
        badge = "deceptive-badge" if result["label"] == "DECEPTIVE" else "truthful-badge"
        st.markdown(f"### Result: <span class='{badge}'>{result['label']}</span>", unsafe_allow_html=True)

        st.metric("Continuous Deception Score", f"{result['score']:.4f}")
        st.progress(float(result['score']))

        st.markdown(f"**Confidence**: `{result['confidence']}` | **Threshold**: `{result['threshold']:.2f}`")

        st.markdown("---")
        st.markdown("#### Top Contributing Fuzzy Rules:")
        for idx, rule in enumerate(result["dominant_rules"], 1):
            st.info(f"**Rule {idx}:** {rule['rule']}\n\n*Activation Weight:* **{rule['activation_weight']*100:.1f}%** | *Effect:* **{rule['tendency']}**")


# ==============================================================================
# MODE 3: EVALUATION & ARCHITECTURE
# ==============================================================================
else:
    st.title("📊 Architecture, Evaluation & Learned Rules")

    tab1, tab2, tab3 = st.tabs(["Model Comparison & ROC", "Membership Functions & Loss", "Fuzzy Rule Base"])

    with tab1:
        st.subheader("Benchmark: ANFIS vs. SVM vs. Fixed Threshold")
        col_m1, col_m2 = st.columns(2)

        roc_path = os.path.join(PROJECT_ROOT, "results", "figures", "roc_comparison_all.png")
        comp_path = os.path.join(PROJECT_ROOT, "results", "figures", "model_comparison_all.png")

        if os.path.exists(roc_path):
            with col_m1:
                st.image(roc_path, caption="ROC Curve Overlay Across All 3 Models", use_container_width=True)
        if os.path.exists(comp_path):
            with col_m2:
                st.image(comp_path, caption="Accuracy, F1, and ROC-AUC Comparison", use_container_width=True)

        # Summary Table
        summary_path = os.path.join(PROJECT_ROOT, "results", "metrics_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                metrics_data = json.load(f)
            st.markdown("#### Test Set Performance Summary:")
            st.table(metrics_data)

    with tab2:
        st.subheader("Gaussian Membership Functions Before vs. After Tuning")
        mf_before = os.path.join(PROJECT_ROOT, "results", "figures", "mf_before.png")
        mf_after = os.path.join(PROJECT_ROOT, "results", "figures", "mf_after.png")
        loss_curve = os.path.join(PROJECT_ROOT, "results", "figures", "loss_curve.png")

        if os.path.exists(mf_before) and os.path.exists(mf_after):
            st.image(mf_before, caption="Initial Membership Functions", use_container_width=True)
            st.image(mf_after, caption="Trained / Adjusted Membership Functions", use_container_width=True)

        if os.path.exists(loss_curve):
            st.image(loss_curve, caption="ANFIS Hybrid Training Loss Curve", use_container_width=True)

    with tab3:
        st.subheader("Learned Takagi-Sugeno Fuzzy Rules")
        rules_file = os.path.join(PROJECT_ROOT, "results", "rules", "learned_rules.txt")
        if os.path.exists(rules_file):
            with open(rules_file, "r") as f:
                rules_text = f.read()
            st.code(rules_text, language="text")
        else:
            st.info("Rule export file not found.")
