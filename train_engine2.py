import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import logging

# FIX H-2: Import the hash recorder so the integrity sidecar stays in
# sync with the model file after every training run. Without this call,
# model_loader.py finds no sidecar on first boot, writes one from whatever
# .pkl is on disk, and silently nullifies the tamper-detection mechanism.
from model_loader import record_hash_after_training

# Configure professional execution tracking logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RxGuardStateSpaceTrainer")


def run_elite_simulation_pipeline():
    logger.info("📡 Launching Multi-Compartment Pharmacokinetic State-Space Network...")

    np.random.seed(1337)
    records = 7500  # High sample density to map a comprehensive, smooth risk landscape

    # ==========================================
    # 1. CONTINUOUS BIOMARKER INGESTION SPACE
    # ==========================================
    egfr        = np.random.uniform(35, 125, records)          # Glomerular Filtration Rate (mL/min)
    albumin     = np.random.uniform(2.8, 5.4, records)         # Hepatic Transport Protein Buffer (g/dL)
    haemoglobin = np.random.uniform(10.0, 18.0, records)       # Oxygen Carrier Mass (g/dL)

    # Categorical Drug Deployment Distribution Vectors
    has_antibiotic   = np.random.choice([0, 1], size=records, p=[0.75, 0.25])
    has_adaptogen    = np.random.choice([0, 1], size=records, p=[0.65, 0.35])
    has_antidiabetic = np.random.choice([0, 1], size=records, p=[0.80, 0.20])
    has_botanical    = np.random.choice([0, 1], size=records, p=[0.70, 0.30])

    # ==========================================
    # 2. STATE-SPACE PK/PD MODELLING LAYER
    # ==========================================
    # Continuous elimination rate constant driven directly by fractional renal clearance
    ke_renal = 0.0075 * egfr

    # Ingested Active Pharmaceutical Ingredient (API) starting mass proxies
    central_dose_load = (has_antibiotic * 50.0) + (has_antidiabetic * 40.0)

    # Non-linear Hepatic Competitive Saturation Matrix via Michaelis-Menten Extensions
    # Herbs act as competitive inhibitors, expanding the apparent Michaelis Constant (Km)
    herbal_inhibitor_concentration = (has_adaptogen * 2.2) + (has_botanical * 1.9)
    Ki = 1.35  # Inhibitor dissociation affinity constant
    competitive_inhibition_scalar = 1.0 + (herbal_inhibitor_concentration / Ki)

    # Protein-bound deconvolution (Low albumin increases the highly reactive free unbound fraction)
    free_fraction_scalar = 4.2 / albumin

    # Solve for Steady-State Peripheral Tissue Accumulation Concentration (Cp)
    # This represents the continuous biological drug/herb stress loading surface
    systemic_toxic_concentration = (
        (central_dose_load / (1.0 + ke_renal))
        * competitive_inhibition_scalar
        * free_fraction_scalar
    )

    # ==========================================
    # 3. AUTONOMIC DEVICE TELEMETRY STREAM COUPLING
    # ==========================================
    hrv_base  = np.random.normal(52, 4.5, records)
    temp_base = np.random.normal(36.55, 0.07, records)

    # Apply continuous biological decay curves to the wearable signal variables
    # Parasympathetic withdrawal forces an exponential HRV crash; inflammation drives thermal drift
    hrv       = hrv_base  * np.exp(-0.024 * systemic_toxic_concentration)
    skin_temp = temp_base + (0.052 * np.log1p(systemic_toxic_concentration))

    # Continuous Sleep Architecture Analytics (Cole-Kripke Alignment Modeling)
    # Overnight toxic overload induces autonomic micro-arousals, disrupting deep N3 sleep
    # FIX #2 — clip total_sleep to physiological floor (>=0.5h) to prevent negative hours
    total_sleep_raw = (
        np.random.uniform(6.8, 8.6, records)
        - (0.16 * systemic_toxic_concentration)
    )
    total_sleep = np.clip(total_sleep_raw, 0.5, 12.0)

    deep_sleep_probability = 1.0 / (1.0 + np.exp(-0.12 * (hrv - 36.5)))
    deep_sleep = (total_sleep * 0.26) * deep_sleep_probability

    # Inject continuous hardware noise distributions matching realistic Bluetooth LE sensor variants
    hrv         += np.random.normal(0, 1.1,   records)
    skin_temp   += np.random.normal(0, 0.035, records)
    total_sleep += np.random.normal(0, 0.08,  records)
    deep_sleep  += np.random.normal(0, 0.04,  records)

    # FIX #3 — enforce deep_sleep <= total_sleep after noise injection (physiological constraint)
    deep_sleep = np.clip(np.minimum(deep_sleep, total_sleep * 0.5), 0.0, None)

    # Correlated secondary vitals buffers
    # FIX #4 — clip SpO2 to physiological floor (>=88%) and ceiling (<=100%)
    spo2_raw = (
        np.random.uniform(96.5, 99.5, records)
        - (systemic_toxic_concentration * 0.04)
    )
    spo2 = np.clip(spo2_raw, 88.0, 100.0)

    glucose = (
        np.random.normal(102, 7.5, records)
        + (has_antidiabetic * -15)
        + (has_botanical    * -5)
    )

    # ==========================================
    # 4. TRI-TIER TRIAGE ANCHOR CALCULATION
    # ==========================================
    # System evaluates training labels as a continuous gradient, maximising real-world precision
    critical_toxic_threshold = 14.5
    y = (systemic_toxic_concentration > critical_toxic_threshold).astype(int)

    # FIX #5 — Diagnose class balance and compute corrective weight for XGBoost
    positive_rate = y.mean()
    negative_rate = 1.0 - positive_rate
    logger.info(f"📊 ADR Prevalence Rate  : {positive_rate:.2%}  (ADR=1)")
    logger.info(f"📊 Non-ADR Prevalence   : {negative_rate:.2%}  (ADR=0)")

    if positive_rate < 0.05 or positive_rate > 0.95:
        logger.warning(
            "⚠️  Extreme class imbalance detected. "
            "Consider adjusting critical_toxic_threshold for better label balance."
        )

    # Inverse-frequency weight — prevents XGBoost collapsing to an all-zero predictor
    scale_pos_weight = negative_rate / positive_rate

    # Assemble Unified Data Matrix Feature Space
    X = pd.DataFrame({
        'egfr'            : egfr,
        'albumin'         : albumin,
        'haemoglobin'     : haemoglobin,
        'hrv'             : hrv,
        'skin_temp'       : skin_temp,
        'spo2'            : spo2,
        'glucose'         : glucose,
        'total_sleep'     : total_sleep,
        'deep_sleep'      : deep_sleep,
        'has_antibiotic'  : has_antibiotic,
        'has_adaptogen'   : has_adaptogen,
        'has_antidiabetic': has_antidiabetic,
        'has_botanical'   : has_botanical,
    })

    # ==========================================
    # 5. XGBOOST COMPILATION & OPTIMIZATION
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=1337
    )

    logger.info("⚡ Mapping physiological decision surface via Gradient-Boosted Tree Matrix...")

    # FIX #1 — use 'seed' (not 'random_state') for correct XGBoost reproducibility
    # FIX #5 — scale_pos_weight corrects for label imbalance
    # FIX #6 — eval_metric changed to 'aucpr', far more informative than logloss for imbalanced clinical labels
    #           use_label_encoder=False suppresses deprecation warning on XGBoost < 2.0
    model = XGBClassifier(
        n_estimators      = 250,
        max_depth         = 5,
        learning_rate     = 0.035,
        subsample         = 0.85,
        colsample_bytree  = 0.85,
        scale_pos_weight  = scale_pos_weight,   # FIX #5
        eval_metric       = 'aucpr',             # FIX #6
        use_label_encoder = False,               # FIX #1
        seed              = 1337,                # FIX #1
    )

    model.fit(X_train, y_train)

    # ==========================================
    # 6. HOLDOUT EVALUATION SUITE (FIX #6)
    # ==========================================
    # FIX #6 — Replace misleading accuracy-only metric with full clinical evaluation suite:
    #           AUC-ROC, precision, recall, F1-score per class.
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    holdout_accuracy = model.score(X_test, y_test) * 100
    auc_roc          = roc_auc_score(y_test, y_prob)

    logger.info(f"🎯 Holdout Accuracy      : {holdout_accuracy:.2f}%")
    logger.info(f"📈 AUC-ROC               : {auc_roc:.4f}  (target >= 0.90 for clinical use)")
    logger.info(
        "📋 Full Classification Report:\n"
        + classification_report(y_test, y_pred, target_names=["No ADR", "ADR"])
    )

    if auc_roc < 0.85:
        logger.warning(
            "⚠️  AUC-ROC below clinical threshold. "
            "Review threshold, feature engineering, or increase records."
        )

    # Serialize model weights to production binary artifact
    model_output_path = Path("symbiote_classifier.pkl")
    joblib.dump(model, model_output_path)

    # FIX H-2: Write the SHA-256 integrity sidecar immediately after the model
    # is saved, so model_loader.py can verify the file on the next server boot.
    record_hash_after_training(model_output_path)
    logger.info("📦 Success! Model and integrity hash saved as 'symbiote_classifier.pkl' + '.sha256'.")


if __name__ == "__main__":
    run_elite_simulation_pipeline()
