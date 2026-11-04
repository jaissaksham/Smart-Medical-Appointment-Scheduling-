# Smart Medical Appointment Scheduling with Machine Learning & Overbooking

An end-to-end, native Python healthcare operations platform designed to eliminate no-show downtime, prevent clinic waiting room congestion, and optimize facility scheduling via probabilistic overbooking.

---

## 🌟 Key Innovations Implemented

1. **Patient Attendance Propensity Estimation ($Prb_p$):**
   - Trained using **Scikit-Learn** across 5 core classification algorithms:
     - **Support Vector Machine (SVM)** (*Highest Recall & AUC*)
     - **Decision Tree (DT)**
     - **Neural Network / Multi-Layer Perceptron (NN)**
     - **Logistic Regression (LR)**
     - **Naive Bayes (NB)**
     - **Random Forest (RF)**
   - Continuous probability calibration with zero data leakage over chronological patient histories.

2. **Integer Linear Optimization & Smart Overbooking:**
   - **Center Expected Utility Objective:**
     $$\max Z = \sum_{p \in P} y_p \cdot Prb_p \cdot SP_p - \sum_{(d,b,s)} \alpha_k \cdot B_{dbs}$$
   - **The $\le 120\%$ Propensity Safety Cap:**
     $$\sum_{p \in \text{slot}} Prb_p \le 1.20$$
     Prevents overbooking collisions by pairing high-attendance likelihood patients with low-attendance likelihood patients, ensuring physician idle time is minimized while keeping unexpected collisions near zero.
   - **Patient Prioritization:**
     - Priority patients ($f_p=1$): Reserved 2 slots ($SP_p = 2$).
     - First-time patients ($G_p=1$): Reserved designated quotas ($q_d$).
     - Regular patients: 1 slot ($SP_p = 1$).

3. **Operational Benchmarking & Statistical Validation:**
   - Multi-model evaluation benchmark across accuracy, precision, recall, F1, and ROC-AUC.
   - Alpha penalty sensitivity curve ($\alpha_k = 0.1 \dots 0.9$).
   - Multi-cohort scheduling simulations (200 to 600 patients).
   - Statistical hypothesis validation (Welch's ANOVA, Shapiro-Wilk normality, Bartlett's variance tests) confirming statistically superior utility over blind overbooking.

4. **Native Python Streamlit UI:**
   - Interactive multi-tab clinical dashboard.
   - Real-time patient propensity gauge and smart companion matching.
   - Weekly color-coded 160-slot clinic schedule heatmap (Monday–Friday, 8:00 AM – 4:00 PM).
   - Database explorer with filtering and CSV downloads.

---

## 📁 Repository Structure

```
.
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation
├── data/
│   ├── appointments.csv        # 111,488 historical clinical appointments
│   ├── patients.csv            # 36,697 registered patient profiles
│   └── slots.csv               # 104,360 clinic slots
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion and cohort merging
│   ├── features.py             # Feature engineering & patient history
│   ├── train.py                # Scikit-learn training & benchmarking
│   ├── optimizer.py            # Integer scheduling & overbooking engine
│   └── utils.py                # Sensitivity analysis & Welch's ANOVA
└── models/
    ├── best_svm_model.joblib   # Serialized champion calibrated SVM pipeline
    ├── benchmark_results.json  # Model comparison metrics & ROC curves
    └── test_sample.joblib      # Validation test set
```

---

## 🚀 Quickstart Guide

### 1. Activate Environment & Install Dependencies

```bash
# Using the preconfigured virtual environment:
source .venv/bin/activate

# Or install in a fresh virtualenv:
pip install -r requirements.txt
```

### 2. Train and Benchmark the Models

```bash
PYTHONPATH=. python src/train.py
```

### 3. Launch the Streamlit Frontend

```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Features & Navigation

- **🏥 System Overview & Operations:** System architecture, problem formulation, and live dataset KPIs.
- **🤖 ML Model Hub & Benchmark:** Interactive comparison of all 5 algorithms, ROC curves, confusion matrices, and custom retraining controls.
- **🩺 Real-time Patient Propensity:** Real-time form for doctors and intake staff to compute attendance probability and safe overbooking companion thresholds.
- **📅 Smart Overbooking Optimizer:** Solves weekly schedules across 160 slots, comparing Smart ML Overbooking with the blind hospital baseline.
- **📈 Sensitivity & What-If Analysis:** Interactive alpha sensitivity curves and Welch's ANOVA tests.
- **🗂️ Hospital Database Explorer:** Filter, inspect, and export patient and appointment records.
