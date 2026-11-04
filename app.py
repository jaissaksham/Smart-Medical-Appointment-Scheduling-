"""
Smart Medical Appointment Scheduling Dashboard.
A minimal, clean, native Python ML application built with Streamlit and Scikit-Learn.
Predicts patient no-show risks and optimizes clinic scheduling to reduce doctor downtime.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_raw_data, get_merged_dataset
from src.features import (
    extract_features,
    create_single_patient_features,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES
)
from src.train import (
    load_champion_model,
    train_and_benchmark,
    predict_patient_propensity,
    MODEL_DIR
)
from src.optimizer import AppointmentScheduler
from src.utils import sample_patient_cohort

# Page configuration
st.set_page_config(
    page_title="Smart Clinic AI — Appointment Scheduling System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def get_cached_raw_data():
    return load_raw_data()


@st.cache_data
def get_cached_merged_data():
    return get_merged_dataset(valid_only=True)


@st.cache_resource
def get_cached_champion_model():
    return load_champion_model()


def load_benchmark_cache():
    results_path = os.path.join(MODEL_DIR, "benchmark_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            return json.load(f)
    return None


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/hospital.png", width=60)
st.sidebar.title("Smart Clinic AI")
st.sidebar.caption("AI-Powered Medical Appointment Scheduling System")
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard Overview",
        "🤖 Model Comparison",
        "🩺 Patient Attendance Predictor",
        "📅 Smart Appointment Scheduler",
        "🗂️ Hospital Records Explorer"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Project Purpose:**\n\n"
    "Predicts patient attendance probability using Machine Learning, "
    "allowing clinics to optimize doctor schedules and prevent empty time slots."
)

# Load base dataset
with st.spinner("Loading clinic data..."):
    app_df, patients_df, slots_df = get_cached_raw_data()
    merged_valid = get_cached_merged_data()
    model = get_cached_champion_model()
    benchmark_data = load_benchmark_cache()


# ==============================================================================
# TAB 1: DASHBOARD OVERVIEW
# ==============================================================================
if nav_choice == "📊 Dashboard Overview":
    st.title("Smart Medical Appointment Scheduling")
    st.caption("An AI-powered system to predict patient attendance and maximize clinic efficiency")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Appointments", f"{len(app_df):,}", "Historical Records")
    with col2:
        st.metric("Registered Patients", f"{len(patients_df):,}", "Unique Patients")
    with col3:
        attended_count = (app_df['status'] == 'attended').sum()
        att_rate = attended_count / len(app_df) * 100
        st.metric("Overall Attendance Rate", f"{att_rate:.1f}%", f"{attended_count:,} attended")
    with col4:
        multi_slot_count = (app_df['slot_id'].value_counts() > 1).sum()
        st.metric("Overbooked Slots in DB", f"{multi_slot_count:,}", "Multi-patient slots")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("💡 About the Project")
        st.markdown("""
        Missed medical appointments (no-shows) are a major challenge for healthcare clinics:
        - **The Problem:** When patients don't show up, doctors sit idle, consultation rooms are wasted, and other patients face longer waitlists.
        - **Why Blind Overbooking Fails:** Many clinics double-book appointments blindly. If both patients arrive at once, it leads to severe delays and overcrowded waiting rooms.
        - **Our AI Solution:** 
          1. **Machine Learning Model:** Analyzes booking lead time, age, past attendance history, and appointment time to calculate the exact probability of a patient attending.
          2. **Smart Scheduling:** If a patient has a high chance of missing their visit, the clinic can safely schedule another patient in that slot without overcrowding.
        """)

        st.success("🎯 **Key Result:** By pairing low-risk with moderate-risk patients, the system maximizes doctor room utilization while keeping patient wait time and collisions at zero.")

    with col_right:
        st.subheader("🔄 How It Works")
        st.markdown("""
        ```mermaid
        flowchart TD
            A[Patient Requests Appointment] --> B[Extract Features & Visit History]
            B --> C[Machine Learning Model]
            C --> D[Predicted Attendance Probability]
            D --> E[Smart Scheduler]
            E --> F[Optimal Weekly Clinic Timetable]
        ```
        """, unsafe_allow_html=False)


# ==============================================================================
# TAB 2: MODEL COMPARISON
# ==============================================================================
elif nav_choice == "🤖 Model Comparison":
    st.title("Machine Learning Model Comparison")
    st.caption("Evaluation of 5 machine learning models trained on clinic appointment data")

    if benchmark_data is None:
        st.warning("Benchmark results not found. Click below to train models.")
        if st.button("🚀 Train Models Now"):
            with st.spinner("Training scikit-learn models..."):
                benchmark_data, _ = train_and_benchmark(sample_size=20000)
            st.rerun()
    else:
        df_bench = []
        for name, m in benchmark_data.items():
            df_bench.append({
                "Algorithm": name,
                "Accuracy": f"{m['accuracy'] * 100:.2f}%",
                "Precision": f"{m['precision'] * 100:.2f}%",
                "Recall": f"{m['recall'] * 100:.2f}%",
                "F1-Score": f"{m['f1_score'] * 100:.2f}%",
                "ROC-AUC": f"{m['auc']:.3f}",
                "Training Time (s)": f"{m['train_time_sec']:.2f}s"
            })
        bench_df = pd.DataFrame(df_bench)

        st.subheader("📋 Performance Comparison")
        st.dataframe(bench_df.set_index("Algorithm"), width="stretch")

        col_roc, col_bar = st.columns([1, 1])

        with col_roc:
            st.subheader("📈 ROC Curves")
            fig_roc = go.Figure()
            for name, m in benchmark_data.items():
                if "roc_data" in m:
                    fig_roc.add_trace(go.Scatter(
                        x=m["roc_data"]["fpr"],
                        y=m["roc_data"]["tpr"],
                        mode='lines',
                        name=f"{name} (AUC={m['auc']:.2f})"
                    ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name="Random Guess"
            ))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                margin=dict(l=20, r=20, t=30, b=20),
                height=350,
                legend=dict(font=dict(size=9))
            )
            st.plotly_chart(fig_roc, width="stretch")

        with col_bar:
            st.subheader("📊 F1-Score & Recall")
            metrics_chart_df = pd.DataFrame([
                {"Model": k, "Metric": "Recall", "Score": v["recall"] * 100} for k, v in benchmark_data.items()
            ] + [
                {"Model": k, "Metric": "F1-Score", "Score": v["f1_score"] * 100} for k, v in benchmark_data.items()
            ])
            fig_bar = px.bar(
                metrics_chart_df,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                color_discrete_sequence=["#2563EB", "#10B981"]
            )
            fig_bar.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                height=350,
                yaxis_title="Score (%)",
                xaxis_tickangle=-20
            )
            st.plotly_chart(fig_bar, width="stretch")

        # Confusion Matrix Inspection
        st.subheader("🎯 Confusion Matrix Inspector")
        model_names = list(benchmark_data.keys())
        selected_model = st.selectbox("Select model to inspect:", model_names, index=0)
        
        cm = np.array(benchmark_data[selected_model]["confusion_matrix"])
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Status", y="Actual Status", color="Count"),
            x=["Missed / Cancelled (0)", "Attended (1)"],
            y=["Missed / Cancelled (0)", "Attended (1)"],
            text_auto=True,
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(height=300, margin=dict(l=40, r=40, t=20, b=20))
        st.plotly_chart(fig_cm, width="stretch")


# ==============================================================================
# TAB 3: PATIENT ATTENDANCE PREDICTOR
# ==============================================================================
elif nav_choice == "🩺 Patient Attendance Predictor":
    st.title("Patient Attendance Predictor")
    st.caption("Enter patient and booking details to predict attendance probability in real time")

    col_in, col_out = st.columns([3, 2])

    with col_in:
        st.subheader("📝 Patient Intake Details")
        with st.form("patient_form"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("Patient Age", min_value=15, max_value=95, value=48)
                sex = st.selectbox("Sex", ["Female", "Male"])
                insurance = st.selectbox(
                    "Insurance Provider",
                    [
                        "Vitalynx Orbit",
                        "Carebubble Spectrum",
                        "CurativeWhale",
                        "HealthZenotron",
                        "Mediflora Nexus",
                        "BioCrest Harmony",
                        "MediNimbus",
                        "QuantumPetal"
                    ]
                )
                is_priority = st.checkbox("Priority / Senior Care (Needs double slot)", value=(age >= 75))

            with c2:
                sched_interval = st.slider(
                    "Booking Lead Time (Days booked in advance)",
                    min_value=1,
                    max_value=30,
                    value=5,
                    help="Patients booking far in advance often have higher cancellation rates."
                )
                day_of_week = st.selectbox("Appointment Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
                hour = st.slider("Appointment Hour", min_value=8, max_value=16, value=10)
                time_slot_num = (hour - 8) * 4

            st.markdown("##### 📜 Patient Attendance History")
            h1, h2, h3 = st.columns(3)
            with h1:
                is_first = st.checkbox("First-Time Patient", value=False)
            with h2:
                prior_appts = 0 if is_first else st.number_input("Prior Appointments", min_value=0, max_value=15, value=3)
            with h3:
                cum_att = 0 if is_first else st.number_input("Prior Attended Visits", min_value=0, max_value=int(prior_appts), value=min(2, int(prior_appts)))

            submit_btn = st.form_submit_button("🔮 Predict Attendance", width="stretch")

    with col_out:
        st.subheader("🎯 Prediction Result")
        
        patient_dict = {
            "scheduling_interval": sched_interval,
            "age": age,
            "sex": sex,
            "insurance": insurance,
            "prior_appointments": prior_appts,
            "cum_attended": cum_att,
            "cum_missed": max(0, prior_appts - cum_att),
            "is_priority": is_priority,
            "day_of_week": day_of_week,
            "hour": hour,
            "time_slot_num": time_slot_num,
            "month": 6
        }

        feat_single = create_single_patient_features(patient_dict)
        prb = predict_patient_propensity(feat_single, model=model)

        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prb * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Attendance Probability", 'font': {'size': 18}},
            number={'suffix': "%", 'font': {'size': 32}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2563EB"},
                'steps': [
                    {'range': [0, 50], 'color': "#FEE2E2"},
                    {'range': [50, 75], 'color': "#FEF3C7"},
                    {'range': [75, 100], 'color': "#DCFCE7"}
                ]
            }
        ))
        fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, width="stretch")

        max_companion_prb = max(0.0, round(1.20 - prb, 2))

        if prb >= 0.70:
            st.success(
                f"**High Attendance Likelihood ({prb*100:.1f}%)**\n\n"
                f"Patient is highly likely to attend. If overbooked, companion patient must have probability ≤ **{max_companion_prb*100:.0f}%**."
            )
        elif prb >= 0.50:
            st.warning(
                f"**Moderate Attendance Likelihood ({prb*100:.1f}%)**\n\n"
                f"Eligible for pairing with a companion patient having probability ≤ **{max_companion_prb*100:.0f}%**."
            )
        else:
            st.error(
                f"**Elevated No-Show Risk ({prb*100:.1f}%)**\n\n"
                f"Good candidate for overbooking! Can be scheduled alongside another patient to ensure the room is not empty."
            )

        slots_needed = 2 if is_priority else 1
        st.write(f"⏱️ **Appointment Duration:** {slots_needed * 15} minutes ({slots_needed} slot{'s' if slots_needed > 1 else ''})")


# ==============================================================================
# TAB 4: SMART APPOINTMENT SCHEDULER
# ==============================================================================
elif nav_choice == "📅 Smart Appointment Scheduler":
    st.title("Smart Appointment Scheduler")
    st.caption("Generate and compare weekly clinic schedules (160 slots)")

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        cohort_size = st.select_slider(
            "Patient Batch Size",
            options=[150, 200, 250, 300, 350, 400],
            value=250
        )
    with ctrl2:
        alpha = st.slider(
            "Overbooking Delay Penalty",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Higher values reduce aggressive overbooking."
        )
    with ctrl3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_sim = st.button("⚡ Generate Weekly Schedule", width="stretch")

    # Run optimization
    cohort = sample_patient_cohort(merged_valid, cohort_size=cohort_size, model_pipeline=model, features_extractor=extract_features)
    scheduler = AppointmentScheduler(
        num_days=5,
        slots_per_day=32,
        k_max_per_slot=2,
        alpha_penalty=alpha,
        max_prob_sum=1.20
    )

    smart_grid, smart_metrics = scheduler.optimize_smart_schedule(cohort)
    base_grid, base_metrics = scheduler.optimize_hospital_baseline(cohort)

    st.markdown("---")
    st.subheader("⚖️ Scheduling Comparison")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Scheduled Patients", f"{smart_metrics['num_scheduled_patients']}", f"{smart_metrics['num_scheduled_patients'] - base_metrics['num_scheduled_patients']:+d} vs Baseline")
    with m2:
        st.metric("Occupancy Rate", f"{smart_metrics['occupancy_rate_pct']:.1f}%", f"{smart_metrics['occupancy_rate_pct'] - base_metrics['occupancy_rate_pct']:+.1f}%")
    with m3:
        st.metric("Overbooked Slots", f"{smart_metrics['overbooked_slots']} / 160", "Double-booked slots")
    with m4:
        st.metric("High Collision Risk Slots", f"{smart_metrics['high_collision_risk_slots']}", f"{smart_metrics['high_collision_risk_slots'] - base_metrics['high_collision_risk_slots']:+d} (Zero collisions!)")

    # Comparison summary table
    comp_df = pd.DataFrame([
        {
            "Metric": "Scheduled Patients",
            "Smart AI Scheduling": str(smart_metrics['num_scheduled_patients']),
            "Blind Double-Booking": str(base_metrics['num_scheduled_patients'])
        },
        {
            "Metric": "Total Slots Assigned",
            "Smart AI Scheduling": str(smart_metrics['total_scheduled_slots']),
            "Blind Double-Booking": str(base_metrics['total_scheduled_slots'])
        },
        {
            "Metric": "Expected Clinic Utility Score",
            "Smart AI Scheduling": f"{smart_metrics['utility_Z']:.2f}",
            "Blind Double-Booking": f"{base_metrics['utility_Z']:.2f}"
        },
        {
            "Metric": "High Collision Slots (Both arrive at once)",
            "Smart AI Scheduling": f"✅ {smart_metrics['high_collision_risk_slots']} slots (Zero!)",
            "Blind Double-Booking": f"⚠️ {base_metrics['high_collision_risk_slots']} slots (High Delays!)"
        },
        {
            "Metric": "Average Attendance Density per Slot",
            "Smart AI Scheduling": f"{smart_metrics['avg_slot_propensity']:.2f}",
            "Blind Double-Booking": f"{base_metrics['avg_slot_propensity']:.2f}"
        }
    ])
    st.table(comp_df.set_index("Metric"))

    # Weekly Heatmap
    st.subheader("🗓️ Weekly Schedule Heatmap (Monday – Friday, 8:00 AM – 4:00 PM)")
    
    view_mode = st.radio("Display Timetable For:", ["Smart AI Scheduling", "Blind Double-Booking"], horizontal=True)
    active_grid = smart_grid if "Smart" in view_mode else base_grid

    pivot_df = active_grid.pivot(index="time_str", columns="day_name", values="prob_sum")
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    pivot_df = pivot_df[days_order]

    fig_heat = px.imshow(
        pivot_df,
        labels=dict(x="Day", y="Time Slot", color="Patient Density"),
        color_continuous_scale="Viridis",
        title=f"Clinic Slot Attendance Density ({view_mode})"
    )
    fig_heat.update_layout(height=600, margin=dict(l=40, r=40, t=30, b=20))
    st.plotly_chart(fig_heat, width="stretch")

    with st.expander("🔍 Inspect Individual Day Slots"):
        selected_day = st.selectbox("Choose Day:", days_order)
        day_slots = active_grid[active_grid["day_name"] == selected_day][["time_str", "assigned_patients", "prob_sum", "is_overbooked"]]
        day_slots["patients_count"] = day_slots["assigned_patients"].apply(len)
        day_slots["prob_sum"] = day_slots["prob_sum"].round(2)
        st.dataframe(day_slots, width="stretch")


# ==============================================================================
# TAB 5: HOSPITAL RECORDS EXPLORER
# ==============================================================================
elif nav_choice == "🗂️ Hospital Records Explorer":
    st.title("Hospital Records Explorer")
    st.caption("Browse and search raw records from the clinic database")

    table_choice = st.radio("Select Table:", ["Appointments (111k rows)", "Patients (36k rows)", "Slots (104k rows)"], horizontal=True)

    if "Appointments" in table_choice:
        c1, c2, c3 = st.columns(3)
        with c1:
            status_filter = st.multiselect("Status:", options=app_df["status"].unique(), default=["attended", "did not attend", "cancelled"])
        with c2:
            sex_filter = st.multiselect("Sex:", options=app_df["sex"].unique(), default=list(app_df["sex"].unique()))
        with c3:
            max_rows = st.slider("Rows:", 50, 500, 150)

        filtered = app_df[app_df["status"].isin(status_filter) & app_df["sex"].isin(sex_filter)].head(max_rows)
        st.dataframe(filtered, width="stretch")
        st.download_button(
            "📥 Download Filtered Appointments CSV",
            data=filtered.to_csv(index=False),
            file_name="appointments_export.csv",
            mime="text/csv"
        )

    elif "Patients" in table_choice:
        ins_filter = st.multiselect("Insurance:", options=patients_df["insurance"].unique(), default=list(patients_df["insurance"].unique()[:4]))
        p_filtered = patients_df[patients_df["insurance"].isin(ins_filter)].head(150)
        st.dataframe(p_filtered, width="stretch")

    else:
        avail_filter = st.selectbox("Slot Availability:", ["All", "Available Only (True)", "Booked Only (False)"])
        if avail_filter == "Available Only (True)":
            s_filtered = slots_df[slots_df["is_available"] == True].head(150)
        elif avail_filter == "Booked Only (False)":
            s_filtered = slots_df[slots_df["is_available"] == False].head(150)
        else:
            s_filtered = slots_df.head(150)
        st.dataframe(s_filtered, width="stretch")
