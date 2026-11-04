"""
Mathematical Optimization and Smart Overbooking Scheduling Module.
Implements the integer scheduling model for healthcare resource utilization.

Equations:
- Utility: Max Z = sum(yp * Prbp * SPp) - sum(alpha_k * Bdbs)
- Overbooking safety limit: sum(Prbp in slot) <= maxPrb (default 1.20)
- Max patients per slot: k = 2
- Priority requirement: SPp = 2 for fp=1; SPp = 1 for others
- Occupancy rate: sum(yp * Prbp * SPp) / sum(yp * SPp)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional


class AppointmentScheduler:
    """
    Solves smart appointment scheduling with overbooking based on ML attendance propensities.
    """

    def __init__(
        self,
        num_days: int = 5,
        slots_per_day: int = 32,
        k_max_per_slot: int = 2,
        alpha_penalty: float = 0.5,
        max_prob_sum: float = 1.20,
        first_time_quota: float = 0.30
    ):
        """
        Parameters:
            num_days: Days in planning horizon (default 5 for Mon-Fri).
            slots_per_day: Slots per day (default 32 for 8:00 to 16:00 at 15 min slots).
            k_max_per_slot: Maximum patients assigned to a single slot (k = 2).
            alpha_penalty: Penalty alpha_k for overbooking a slot (paper uses 0.5).
            max_prob_sum: Maximum sum of attendance probabilities for overbooked slot (maxPrb = 1.20).
            first_time_quota: Target fraction of slots reserved for first-time visitors (qd).
        """
        self.num_days = num_days
        self.slots_per_day = slots_per_day
        self.total_slots = num_days * slots_per_day
        self.k_max = k_max_per_slot
        self.alpha_penalty = alpha_penalty
        self.max_prob_sum = max_prob_sum
        self.first_time_quota = first_time_quota

    def generate_slot_grid(self) -> pd.DataFrame:
        """Creates the weekly slot grid (day, slot_num, time_str)."""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][:self.num_days]
        grid = []
        for d_idx, day_name in enumerate(days):
            for s_idx in range(self.slots_per_day):
                total_minutes = 8 * 60 + s_idx * 15
                h = total_minutes // 60
                m = total_minutes % 60
                time_str = f"{h:02d}:{m:02d}"
                grid.append({
                    "day_idx": d_idx,
                    "day_name": day_name,
                    "slot_idx": s_idx,
                    "global_slot_id": d_idx * self.slots_per_day + s_idx,
                    "time_str": time_str
                })
        return pd.DataFrame(grid)

    def optimize_smart_schedule(
        self,
        patients_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes smart scheduling with overbooking constrained by ML propensities (maxPrb <= 1.20).
        
        Priority order:
        1. Priority patients (fp = 1, require 2 slots)
        2. First-time patients (Gp = 1, require 1 slot)
        3. Regular patients (require 1 slot)
        4. Overbooking second patients up to k=2, only if sum(Prb) <= maxPrb.
        """
        slots_grid = self.generate_slot_grid()
        slots_grid["assigned_patients"] = [[] for _ in range(len(slots_grid))]
        slots_grid["prob_sum"] = 0.0
        slots_grid["is_overbooked"] = False

        patients = patients_df.copy()
        if "propensity" not in patients.columns:
            raise ValueError("patients_df must contain 'propensity' column (Prbp).")

        # Ensure parameters
        if "is_priority" not in patients.columns:
            patients["is_priority"] = 0
        if "is_first_time" not in patients.columns:
            patients["is_first_time"] = 0
        if "slots_needed" not in patients.columns:
            patients["slots_needed"] = np.where(patients["is_priority"] == 1, 2, 1)

        scheduled_patients = []
        occupied_slots = set()

        # Step 1: Assign Priority Patients (fp = 1, need 2 consecutive slots, no overbooking primary)
        priority_pts = patients[patients["is_priority"] == 1].sort_values("propensity", ascending=False)
        for _, pt in priority_pts.iterrows():
            # Find 2 consecutive available slots in same day
            assigned = False
            for d in range(self.num_days):
                day_start = d * self.slots_per_day
                for s in range(self.slots_per_day - 1):
                    s1 = day_start + s
                    s2 = day_start + s + 1
                    if s1 not in occupied_slots and s2 not in occupied_slots:
                        occupied_slots.add(s1)
                        occupied_slots.add(s2)
                        slots_grid.at[s1, "assigned_patients"].append(pt["patient_id"])
                        slots_grid.at[s1, "prob_sum"] += pt["propensity"]
                        slots_grid.at[s2, "assigned_patients"].append(pt["patient_id"])
                        slots_grid.at[s2, "prob_sum"] += pt["propensity"]
                        scheduled_patients.append({
                            **pt.to_dict(),
                            "assigned_slots": [s1, s2],
                            "day_name": slots_grid.at[s1, "day_name"],
                            "time_str": slots_grid.at[s1, "time_str"],
                            "is_overbook": False
                        })
                        assigned = True
                        break
                if assigned:
                    break

        # Step 2: Assign First-time Patients (Gp = 1) and Regulars to remaining single slots
        other_pts = patients[
            (patients["is_priority"] == 0) & 
            (~patients["patient_id"].isin([p["patient_id"] for p in scheduled_patients]))
        ].sort_values(["is_first_time", "propensity"], ascending=[False, False])

        remaining_pts = []
        for _, pt in other_pts.iterrows():
            # Find an empty slot
            empty_slots = [s for s in range(self.total_slots) if s not in occupied_slots]
            if empty_slots:
                s = empty_slots[0]
                occupied_slots.add(s)
                slots_grid.at[s, "assigned_patients"].append(pt["patient_id"])
                slots_grid.at[s, "prob_sum"] += pt["propensity"]
                scheduled_patients.append({
                    **pt.to_dict(),
                    "assigned_slots": [s],
                    "day_name": slots_grid.at[s, "day_name"],
                    "time_str": slots_grid.at[s, "time_str"],
                    "is_overbook": False
                })
            else:
                remaining_pts.append(pt)

        # Step 3: Smart Overbooking using ML Propensity Safety Cap
        # Pair patients so that total probability in slot <= max_prob_sum (1.20)
        # To maximize center utility, pair low propensity with high propensity
        rem_df = pd.DataFrame(remaining_pts)
        if not rem_df.empty:
            rem_df = rem_df.sort_values("propensity", ascending=True) # Start with lower propensity for safe pairing
            for _, pt in rem_df.iterrows():
                # Eligible slots for overbooking: currently has exactly 1 patient, and (current_prob + pt_prob) <= max_prob_sum
                best_slot = None
                best_fit = -1.0
                
                for s in range(self.total_slots):
                    assigned_list = slots_grid.at[s, "assigned_patients"]
                    current_prob = slots_grid.at[s, "prob_sum"]
                    
                    if len(assigned_list) == 1 and (current_prob + pt["propensity"]) <= self.max_prob_sum:
                        # Find slot that gets closest to max_prob_sum without exceeding it
                        fit_score = current_prob + pt["propensity"]
                        if fit_score > best_fit:
                            best_fit = fit_score
                            best_slot = s

                if best_slot is not None:
                    slots_grid.at[best_slot, "assigned_patients"].append(pt["patient_id"])
                    slots_grid.at[best_slot, "prob_sum"] += pt["propensity"]
                    slots_grid.at[best_slot, "is_overbooked"] = True
                    scheduled_patients.append({
                        **pt.to_dict(),
                        "assigned_slots": [best_slot],
                        "day_name": slots_grid.at[best_slot, "day_name"],
                        "time_str": slots_grid.at[best_slot, "time_str"],
                        "is_overbook": True
                    })

        sched_df = pd.DataFrame(scheduled_patients)
        metrics = self._calculate_metrics(sched_df, slots_grid)
        metrics["strategy"] = "Smart ML Overbooking Engine"

        return slots_grid, metrics

    def optimize_hospital_baseline(
        self,
        patients_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Replicates current hospital scheduling practice from the paper:
        Assigns patients randomly or first-come first-served, and overbooks slots blindly
        without checking the attendance propensity limit (maxPrb).
        """
        slots_grid = self.generate_slot_grid()
        slots_grid["assigned_patients"] = [[] for _ in range(len(slots_grid))]
        slots_grid["prob_sum"] = 0.0
        slots_grid["is_overbooked"] = False

        patients = patients_df.copy().sample(frac=1.0, random_state=42).reset_index(drop=True)
        if "slots_needed" not in patients.columns:
            patients["slots_needed"] = np.where(patients["is_priority"] == 1, 2, 1)

        scheduled_patients = []
        occupied_slots = set()

        # Step 1: Assign priority and regular patients randomly to fill slots
        unassigned = []
        for _, pt in patients.iterrows():
            slots_needed = pt["slots_needed"]
            assigned = False
            if slots_needed == 2:
                for d in range(self.num_days):
                    day_start = d * self.slots_per_day
                    for s in range(self.slots_per_day - 1):
                        s1 = day_start + s
                        s2 = day_start + s + 1
                        if s1 not in occupied_slots and s2 not in occupied_slots:
                            occupied_slots.add(s1)
                            occupied_slots.add(s2)
                            slots_grid.at[s1, "assigned_patients"].append(pt["patient_id"])
                            slots_grid.at[s1, "prob_sum"] += pt["propensity"]
                            slots_grid.at[s2, "assigned_patients"].append(pt["patient_id"])
                            slots_grid.at[s2, "prob_sum"] += pt["propensity"]
                            scheduled_patients.append({
                                **pt.to_dict(),
                                "assigned_slots": [s1, s2],
                                "day_name": slots_grid.at[s1, "day_name"],
                                "time_str": slots_grid.at[s1, "time_str"],
                                "is_overbook": False
                            })
                            assigned = True
                            break
                    if assigned:
                        break
            else:
                empty = [s for s in range(self.total_slots) if s not in occupied_slots]
                if empty:
                    s = empty[0]
                    occupied_slots.add(s)
                    slots_grid.at[s, "assigned_patients"].append(pt["patient_id"])
                    slots_grid.at[s, "prob_sum"] += pt["propensity"]
                    scheduled_patients.append({
                        **pt.to_dict(),
                        "assigned_slots": [s],
                        "day_name": slots_grid.at[s, "day_name"],
                        "time_str": slots_grid.at[s, "time_str"],
                        "is_overbook": False
                    })
                    assigned = True

            if not assigned:
                unassigned.append(pt)

        # Step 2: Blind overbooking (hospital practice: overbook slots up to k=2 without propensity cap)
        for pt in unassigned:
            eligible = [s for s in range(self.total_slots) if len(slots_grid.at[s, "assigned_patients"]) == 1]
            if eligible:
                s = eligible[0]
                slots_grid.at[s, "assigned_patients"].append(pt["patient_id"])
                slots_grid.at[s, "prob_sum"] += pt["propensity"]
                slots_grid.at[s, "is_overbooked"] = True
                scheduled_patients.append({
                    **pt.to_dict(),
                    "assigned_slots": [s],
                    "day_name": slots_grid.at[s, "day_name"],
                    "time_str": slots_grid.at[s, "time_str"],
                    "is_overbook": True
                })

        sched_df = pd.DataFrame(scheduled_patients)
        metrics = self._calculate_metrics(sched_df, slots_grid)
        metrics["strategy"] = "Hospital Baseline (Blind Overbooking)"

        return slots_grid, metrics

    def _calculate_metrics(self, sched_df: pd.DataFrame, slots_grid: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates utility Z, occupancy rate, overbooking penalty, and collisions.
        """
        if sched_df.empty:
            return {
                "num_scheduled_patients": 0,
                "total_scheduled_slots": 0,
                "expected_attendance_slots": 0.0,
                "occupancy_rate": 0.0,
                "overbooked_slots": 0,
                "overbooking_penalty_total": 0.0,
                "utility_Z": 0.0,
                "high_collision_risk_slots": 0,
                "avg_slot_propensity": 0.0
            }

        num_pts = len(sched_df)
        total_sched_slots = sched_df["slots_needed"].sum()
        expected_att = (sched_df["propensity"] * sched_df["slots_needed"]).sum()
        
        # Occupancy rate = sum(yp * Prbp * SPp) / sum(yp * SPp)
        occupancy_rate = expected_att / total_sched_slots if total_sched_slots > 0 else 0.0

        overbooked_slots_cnt = int(slots_grid["is_overbooked"].sum())
        penalty_total = overbooked_slots_cnt * self.alpha_penalty
        
        # Center Utility Z = Expected Revenue - Overbooking Penalty
        utility_z = expected_att - penalty_total

        # High collision risk: slots where sum of probabilities > 1.25 (causes extreme waiting/spillover)
        high_collisions = int((slots_grid["prob_sum"] > 1.25).sum())

        return {
            "num_scheduled_patients": int(num_pts),
            "total_scheduled_slots": int(total_sched_slots),
            "expected_attendance_slots": round(float(expected_att), 2),
            "occupancy_rate": round(float(occupancy_rate), 4),
            "occupancy_rate_pct": round(float(occupancy_rate * 100), 2),
            "overbooked_slots": overbooked_slots_cnt,
            "overbooking_penalty_total": round(float(penalty_total), 2),
            "utility_Z": round(float(utility_z), 2),
            "high_collision_risk_slots": high_collisions,
            "avg_slot_propensity": round(float(slots_grid["prob_sum"].mean()), 3)
        }

