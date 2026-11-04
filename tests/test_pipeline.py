"""
Unit and Integration Tests for the MLOPs Medical Appointment Scheduling Pipeline.
Tests data loading, feature extraction, ML inference, and optimization constraints.
Uses Python's standard unittest framework for 100% native execution.
"""

import unittest
import numpy as np
import pandas as pd

from src.features import (
    extract_features,
    create_single_patient_features,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES
)
from src.optimizer import AppointmentScheduler
from src.train import load_champion_model, predict_patient_propensity


class TestMedicalSchedulingPipeline(unittest.TestCase):

    def test_feature_creation_single_patient(self):
        patient_dict = {
            "scheduling_interval": 5,
            "age": 45,
            "sex": "Female",
            "insurance": "Vitalynx Orbit",
            "prior_appointments": 3,
            "cum_attended": 2,
            "cum_missed": 1,
            "is_priority": False,
            "day_of_week": "Monday",
            "hour": 10,
            "time_slot_num": 8,
            "month": 4
        }
        df = create_single_patient_features(patient_dict)
        self.assertEqual(len(df), 1)
        self.assertIn("prior_attendance_rate", df.columns)
        self.assertAlmostEqual(df["prior_attendance_rate"].iloc[0], 2 / 3, places=2)
        self.assertEqual(df["is_first_time"].iloc[0], 0)

    def test_model_inference_propensity(self):
        model = load_champion_model()
        patient_dict = {
            "scheduling_interval": 2,
            "age": 35,
            "sex": "Male",
            "insurance": "Carebubble Spectrum",
            "prior_appointments": 0,
            "cum_attended": 0,
            "cum_missed": 0,
            "is_priority": False,
            "day_of_week": "Tuesday",
            "hour": 11,
            "time_slot_num": 12,
            "month": 5
        }
        df = create_single_patient_features(patient_dict)
        prb = predict_patient_propensity(df, model=model)
        self.assertTrue(0.0 <= prb <= 1.0)

    def test_smart_optimizer_safety_constraint(self):
        """Verifies that the optimizer respects the max_prob_sum constraint (1.20)."""
        np.random.seed(42)
        n = 100
        cohort = pd.DataFrame({
            "patient_id": np.arange(1, n + 1),
            "name": [f"Patient {i}" for i in range(1, n + 1)],
            "propensity": np.random.uniform(0.3, 0.9, size=n),
            "is_priority": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
            "is_first_time": np.random.choice([0, 1], size=n, p=[0.7, 0.3])
        })
        cohort["slots_needed"] = np.where(cohort["is_priority"] == 1, 2, 1)

        scheduler = AppointmentScheduler(num_days=5, slots_per_day=32, max_prob_sum=1.20)
        slots_grid, metrics = scheduler.optimize_smart_schedule(cohort)

        self.assertGreater(metrics["num_scheduled_patients"], 0)
        # Overbooked slots must have at most 2 patients
        for _, row in slots_grid.iterrows():
            self.assertLessEqual(len(row["assigned_patients"]), 2)
            # Sum of probabilities must not exceed maxPrb (1.20)
            self.assertLessEqual(row["prob_sum"], 1.2001)

    def test_overbooking_penalty_impact(self):
        """Higher alpha penalty should yield fewer or equal overbooked slots."""
        np.random.seed(42)
        n = 120
        cohort = pd.DataFrame({
            "patient_id": np.arange(1, n + 1),
            "name": [f"Patient {i}" for i in range(1, n + 1)],
            "propensity": np.random.uniform(0.3, 0.9, size=n),
            "is_priority": np.zeros(n, dtype=int),
            "is_first_time": np.zeros(n, dtype=int),
            "slots_needed": np.ones(n, dtype=int)
        })

        scheduler_low = AppointmentScheduler(alpha_penalty=0.1, max_prob_sum=1.20)
        _, m_low = scheduler_low.optimize_smart_schedule(cohort)

        scheduler_high = AppointmentScheduler(alpha_penalty=0.9, max_prob_sum=1.20)
        _, m_high = scheduler_high.optimize_smart_schedule(cohort)

        self.assertGreaterEqual(m_low["overbooked_slots"], m_high["overbooked_slots"])


if __name__ == "__main__":
    unittest.main()
