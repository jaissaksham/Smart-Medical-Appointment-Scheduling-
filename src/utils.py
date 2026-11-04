"""
Utility functions for Smart Medical Scheduling:
Cohort sampling, sensitivity experiments, statistical hypothesis tests (Welch's ANOVA / t-test),
and visualization helpers replicating the research paper's figures and tables.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Tuple
from src.optimizer import AppointmentScheduler


def sample_patient_cohort(
    merged_df: pd.DataFrame,
    cohort_size: int = 300,
    model_pipeline = None,
    features_extractor = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Samples a cohort of patients from the database and attaches ML-predicted propensity Prbp.
    """
    sample = merged_df.sample(n=cohort_size, random_state=random_state).copy()
    
    # If features extractor and model are provided, predict propensity
    if features_extractor is not None and model_pipeline is not None:
        feat_df = features_extractor(sample, is_training=False)
        probs = model_pipeline.predict_proba(feat_df)[:, 1]
        sample["propensity"] = np.clip(probs, 0.15, 0.95)
    else:
        # Fallback based on scheduling interval and age with realistic noise
        base_rate = 0.77
        interval_effect = -(sample["scheduling_interval"] - 7.0) * 0.015
        age_effect = (sample["age"] - 50.0) * 0.002
        noise = np.random.normal(0, 0.10, size=len(sample))
        sample["propensity"] = np.clip(base_rate + interval_effect + age_effect + noise, 0.15, 0.95)

    if "is_priority" not in sample.columns:
        sample["is_priority"] = (sample["age"] >= 70).astype(int)
    if "is_first_time" not in sample.columns:
        sample["is_first_time"] = np.random.choice([0, 1], size=len(sample), p=[0.7, 0.3])

    sample["slots_needed"] = np.where(sample["is_priority"] == 1, 2, 1)
    return sample


def run_alpha_sensitivity_analysis(
    cohort_df: pd.DataFrame,
    alpha_range: List[float] = None
) -> pd.DataFrame:
    """
    Varies alpha from 0.1 to 0.9 in increments of 0.1, calculating the occupancy rate.
    """
    if alpha_range is None:
        alpha_range = [round(a, 1) for a in np.arange(0.1, 1.0, 0.1)]

    records = []
    for alpha in alpha_range:
        scheduler = AppointmentScheduler(alpha_penalty=alpha, k_max_per_slot=2, max_prob_sum=1.20)
        _, metrics = scheduler.optimize_smart_schedule(cohort_df)
        records.append({
            "alpha": alpha,
            "occupancy_rate": metrics["occupancy_rate"],
            "occupancy_rate_pct": metrics["occupancy_rate_pct"],
            "utility_Z": metrics["utility_Z"],
            "overbooked_slots": metrics["overbooked_slots"],
            "scheduled_patients": metrics["num_scheduled_patients"]
        })
    return pd.DataFrame(records)


def run_cohort_size_benchmark(
    merged_df: pd.DataFrame,
    sizes: List[int] = [200, 250, 300, 350, 400, 450, 500, 550, 600],
    alpha: float = 0.5,
    model_pipeline = None,
    features_extractor = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Runs smart scheduling vs baseline across cohort sizes (200 to 600),
    replicating Tables 3-8 and statistical tests in Table 9 from the paper.
    """
    smart_records = []
    base_records = []

    for size in sizes:
        cohort = sample_patient_cohort(merged_df, cohort_size=size, model_pipeline=model_pipeline, features_extractor=features_extractor, random_state=size)
        scheduler = AppointmentScheduler(alpha_penalty=alpha, k_max_per_slot=2, max_prob_sum=1.20)
        
        _, m_smart = scheduler.optimize_smart_schedule(cohort)
        _, m_base = scheduler.optimize_hospital_baseline(cohort)
        
        smart_records.append({
            "cohort_size": size,
            **m_smart
        })
        base_records.append({
            "cohort_size": size,
            **m_base
        })

    smart_df = pd.DataFrame(smart_records)
    base_df = pd.DataFrame(base_records)

    # Statistical significance test (Welch's t-test / ANOVA)
    t_stat, p_val = stats.ttest_ind(smart_df["utility_Z"], base_df["utility_Z"], equal_var=False)
    
    shapiro_smart = stats.shapiro(smart_df["occupancy_rate"])
    shapiro_base = stats.shapiro(base_df["occupancy_rate"])
    bartlett_test = stats.bartlett(smart_df["occupancy_rate"], base_df["occupancy_rate"])

    stats_summary = {
        "t_statistic": round(float(t_stat), 4),
        "welch_p_value": float(p_val),
        "shapiro_smart_p": round(float(shapiro_smart.pvalue), 4),
        "shapiro_base_p": round(float(shapiro_base.pvalue), 4),
        "bartlett_p": float(bartlett_test.pvalue),
        "smart_mean_occupancy": round(float(smart_df["occupancy_rate"].mean()), 4),
        "base_mean_occupancy": round(float(base_df["occupancy_rate"].mean()), 4)
    }

    combined = pd.DataFrame({
        "Cohort Size": sizes,
        "Smart Utility Z": smart_df["utility_Z"],
        "Hospital Utility Z": base_df["utility_Z"],
        "Smart Overbooked Slots": smart_df["overbooked_slots"],
        "Hospital Overbooked Slots": base_df["overbooked_slots"],
        "Smart Collision Slots": smart_df["high_collision_risk_slots"],
        "Hospital Collision Slots": base_df["high_collision_risk_slots"],
        "Smart Occupancy Rate (%)": smart_df["occupancy_rate_pct"],
        "Hospital Occupancy Rate (%)": base_df["occupancy_rate_pct"]
    })

    return combined, stats_summary

