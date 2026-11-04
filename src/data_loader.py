"""
Data Loader Module for Smart Medical Appointment Scheduling.
Handles loading, merging, and cleaning appointment, patient, and slot data.
"""

import os
import pandas as pd
from typing import Tuple, Optional


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_raw_data(data_dir: str = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads appointments, patients, and slots CSVs."""
    app_path = os.path.join(data_dir, "appointments.csv")
    patients_path = os.path.join(data_dir, "patients.csv")
    slots_path = os.path.join(data_dir, "slots.csv")

    if not os.path.exists(app_path):
        raise FileNotFoundError(f"Missing appointments file at: {app_path}")
    if not os.path.exists(patients_path):
        raise FileNotFoundError(f"Missing patients file at: {patients_path}")
    if not os.path.exists(slots_path):
        raise FileNotFoundError(f"Missing slots file at: {slots_path}")

    app_df = pd.read_csv(app_path)
    patients_df = pd.read_csv(patients_path)
    slots_df = pd.read_csv(slots_path)

    return app_df, patients_df, slots_df


def get_merged_dataset(data_dir: str = DATA_DIR, valid_only: bool = True) -> pd.DataFrame:
    """
    Merges appointments with patient demographics and computes primary clean columns.
    
    Parameters:
        data_dir: Path to directory containing CSV files.
        valid_only: If True, filters to records with resolved outcomes (attended, did not attend, cancelled).
    """
    app_df, patients_df, _ = load_raw_data(data_dir)

    # Merge on patient_id
    merged = app_df.merge(
        patients_df[["patient_id", "name", "insurance", "dob"]],
        on="patient_id",
        how="left"
    )

    merged["appointment_date"] = pd.to_datetime(merged["appointment_date"])
    merged["scheduling_date"] = pd.to_datetime(merged["scheduling_date"])

    if valid_only:
        valid_statuses = ["attended", "did not attend", "cancelled"]
        merged = merged[merged["status"].isin(valid_statuses)].copy()

    # Target variable: 1 if attended, 0 if missed/cancelled (non-attendance)
    # This directly represents Attendance Propensity Prb_p as defined in Equation (2) of the paper
    merged["target"] = (merged["status"] == "attended").astype(int)

    return merged

