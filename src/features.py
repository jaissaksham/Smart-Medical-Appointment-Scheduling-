"""
Feature Engineering Module for Medical Attendance Propensity.
Extracts patient history, temporal features, and priority markers based on the research paper.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any


NUMERIC_FEATURES = [
    "scheduling_interval",
    "age",
    "prior_appointments",
    "cum_attended",
    "cum_missed",
    "prior_attendance_rate",
    "is_first_time",
    "is_priority",
    "hour",
    "month",
    "time_slot_num"
]

CATEGORICAL_FEATURES = [
    "sex",
    "insurance",
    "day_of_week"
]


def extract_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Computes chronological patient attendance history and feature representations.
    
    Ensures zero temporal data leakage by using only preceding appointments
    for history calculations.
    """
    data = df.copy()

    # Ensure datetime format and sort chronologically per patient
    if not pd.api.types.is_datetime64_any_dtype(data["appointment_date"]):
        data["appointment_date"] = pd.to_datetime(data["appointment_date"])

    data = data.sort_values(["patient_id", "appointment_date"]).reset_index(drop=True)

    # 1. Historical patient attendance metrics
    if "target" in data.columns and is_training:
        data["_target_shift"] = data.groupby("patient_id")["target"].shift(1).fillna(0)
        data["cum_attended"] = data.groupby("patient_id")["_target_shift"].cumsum()
        data["prior_appointments"] = data.groupby("patient_id").cumcount()
        data["cum_missed"] = data["prior_appointments"] - data["cum_attended"]
        data["is_first_time"] = (data["prior_appointments"] == 0).astype(int)
        
        # Smoothed prior attendance rate (Laplace smoothing or global mean fallback)
        global_att_rate = data["target"].mean() if len(data) > 0 else 0.77
        data["prior_attendance_rate"] = np.where(
            data["prior_appointments"] > 0,
            (data["cum_attended"] + 1.0) / (data["prior_appointments"] + 1.0 / global_att_rate),
            global_att_rate
        )
        data.drop(columns=["_target_shift"], inplace=True)
    else:
        # Defaults if inference without explicit prior columns
        if "prior_appointments" not in data.columns:
            data["prior_appointments"] = 0
        if "cum_attended" not in data.columns:
            data["cum_attended"] = 0
        if "cum_missed" not in data.columns:
            data["cum_missed"] = 0
        if "is_first_time" not in data.columns:
            data["is_first_time"] = (data["prior_appointments"] == 0).astype(int)
        if "prior_attendance_rate" not in data.columns:
            data["prior_attendance_rate"] = 0.77

    # 2. Priority classification (Paper: fp = 1 for priority care, needing 2 slots)
    if "is_priority" not in data.columns:
        data["is_priority"] = (data["age"] >= 70).astype(int)

    # 3. Temporal & calendar features
    data["day_of_week"] = data["appointment_date"].dt.day_name()
    data["month"] = data["appointment_date"].dt.month

    # Time parsing
    if "appointment_time" in data.columns:
        times = pd.to_datetime(data["appointment_time"], format="%H:%M:%S", errors="coerce").dt
        data["hour"] = times.hour.fillna(9).astype(int)
        data["minute"] = times.minute.fillna(0).astype(int)
        data["time_slot_num"] = ((data["hour"] - 8) * 4 + (data["minute"] // 15)).clip(0, 31)
    else:
        data["hour"] = 9
        data["time_slot_num"] = 4

    # Ensure insurance and sex exist
    if "insurance" not in data.columns:
        data["insurance"] = "Standard Clinic"
    data["insurance"] = data["insurance"].fillna("Unknown")

    if "sex" not in data.columns:
        data["sex"] = "Female"
    data["sex"] = data["sex"].fillna("Female")

    return data


def create_single_patient_features(patient_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Constructs a single-row feature DataFrame from user-entered dictionary in Streamlit.
    """
    row = {
        "patient_id": patient_dict.get("patient_id", 99999),
        "scheduling_interval": int(patient_dict.get("scheduling_interval", 5)),
        "age": int(patient_dict.get("age", 45)),
        "sex": patient_dict.get("sex", "Female"),
        "insurance": patient_dict.get("insurance", "Vitalynx Orbit"),
        "prior_appointments": int(patient_dict.get("prior_appointments", 2)),
        "cum_attended": int(patient_dict.get("cum_attended", 2)),
        "cum_missed": int(patient_dict.get("cum_missed", 0)),
        "is_first_time": 1 if int(patient_dict.get("prior_appointments", 2)) == 0 else 0,
        "is_priority": 1 if patient_dict.get("is_priority", False) or int(patient_dict.get("age", 45)) >= 70 else 0,
        "hour": int(patient_dict.get("hour", 10)),
        "month": int(patient_dict.get("month", 5)),
        "day_of_week": patient_dict.get("day_of_week", "Monday"),
        "time_slot_num": int(patient_dict.get("time_slot_num", 8)),
    }

    if row["prior_appointments"] > 0:
        row["prior_attendance_rate"] = row["cum_attended"] / row["prior_appointments"]
    else:
        row["prior_attendance_rate"] = 0.77

    return pd.DataFrame([row])

