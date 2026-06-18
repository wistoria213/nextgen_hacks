"""
inference.py
=============
Pure business logic: validated HealthProfile → AnalysisResponse.

Kept separate from main.py so the API layer stays thin (auth, parsing,
error formatting) and this module remains testable in isolation — you
can unit-test assess_risk() without spinning up a server.

Fixes applied vs. original
---------------------------
  C-1  — FEATURE_ORDER corrected to 13 features matching train_engine.py.
          The original had 9 features with different names; XGBoost raised
          a ValueError on every prediction call.
  C-1  — Feature vector now built by iterating FEATURE_ORDER, so the
          column count and column order can never silently drift again.
  C-2  — DRUG_CLASS_MAP added, covering all 4 trained binary flags
          (has_antibiotic, has_adaptogen, has_antidiabetic, has_botanical)
          with real-world drug/supplement name tokens. The original only
          handled 2 of 4 classes, feeding permanent zeros for the rest.
  M-1  — FEATURE_ORDER is now actively used (was dead code in original).
  M-2  — Wearable data read from the validated HealthProfile instead
          of 2 hardcoded scalar values that bore no relation to the
          continuous distribution the model was trained on.
"""

from typing import Dict, List, Optional

import numpy as np

from schemas import AnalysisResponse, HealthProfile, MedicationRow, OrganMetrics

# ---------------------------------------------------------------------------
# Feature contract — must match train_engine.py EXACTLY (order + count).
# If train_engine.py is ever modified, update this tuple to match.
# ---------------------------------------------------------------------------
FEATURE_ORDER = (
    "egfr",
    "albumin",
    "haemoglobin",
    "hrv",
    "skin_temp",
    "spo2",
    "glucose",
    "total_sleep",
    "deep_sleep",
    "has_antibiotic",
    "has_adaptogen",
    "has_antidiabetic",
    "has_botanical",
)

# ---------------------------------------------------------------------------
# Drug class token map (FIX C-2)
# Maps each training flag to the medicine / supplement name substrings
# that should activate it. Substring matching is intentionally broad
# to catch brand names and compound names (e.g., "amoxicillin-clavulanate").
# ---------------------------------------------------------------------------
DRUG_CLASS_MAP: Dict[str, List[str]] = {
    "has_antibiotic": [
        "amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline",
        "metronidazole", "clarithromycin", "levofloxacin", "trimethoprim",
        "nitrofurantoin", "cephalexin", "clindamycin", "erythromycin",
        "tetracycline", "ampicillin", "penicillin", "sulfamethoxazole",
    ],
    "has_adaptogen": [
        "ashwagandha", "rhodiola", "ginseng", "eleuthero", "schisandra",
        "maca", "holy basil", "tulsi", "astragalus", "reishi", "cordyceps",
    ],
    "has_antidiabetic": [
        "metformin", "glipizide", "glimepiride", "sitagliptin", "empagliflozin",
        "dapagliflozin", "insulin", "pioglitazone", "acarbose", "glyburide",
        "saxagliptin", "canagliflozin", "linagliptin", "repaglinide",
    ],
    "has_botanical": [
        "turmeric", "curcumin", "garlic", "st. john", "ginkgo", "echinacea",
        "bitter melon", "berberine", "milk thistle", "valerian", "saw palmetto",
        "licorice", "liquorice", "feverfew", "kava", "cat's claw",
    ],
}

# Human-readable interaction notes per drug class (used in med_table output)
_INTERACTION_NOTES: Dict[str, str] = {
    "has_antibiotic":   "Possible CYP3A4 / P-gp pathway competition",
    "has_adaptogen":    "Hepatic metabolic saturation — Michaelis-Menten inhibition",
    "has_antidiabetic": "Additive glycaemic lowering — monitor blood glucose closely",
    "has_botanical":    "CYP enzyme induction/inhibition — altered drug serum levels",
}

# Which drug class carries the highest single-drug ADR weight (used for status colouring)
_CLASS_STATUS: Dict[str, str] = {
    "has_antibiotic":   "red",
    "has_adaptogen":    "amber",
    "has_antidiabetic": "amber",
    "has_botanical":    "amber",
}

ADR_PROBABILITY_THRESHOLD = 0.45


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _map_to_drug_classes(meds_lower: List[str]) -> Dict[str, int]:
    """
    Map a list of lowercase medicine names to 4 binary drug-class flags.
    Returns a dict with the same keys as DRUG_CLASS_MAP, each set to 0 or 1.
    """
    flags: Dict[str, int] = {cls: 0 for cls in DRUG_CLASS_MAP}
    for cls, tokens in DRUG_CLASS_MAP.items():
        if any(token in med for med in meds_lower for token in tokens):
            flags[cls] = 1
    return flags


def _classify_medicine(med_lower: str) -> Optional[str]:
    """
    Return the first drug class whose token list matches this medicine name,
    or None if it doesn't match any known class.
    """
    for cls, tokens in DRUG_CLASS_MAP.items():
        if any(token in med_lower for token in tokens):
            return cls
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assess_risk(profile: HealthProfile, classifier: Optional[object]) -> AnalysisResponse:
    """
    Convert a validated HealthProfile into a structured AnalysisResponse.

    Args:
        profile:    Validated inbound payload from the frontend.
        classifier: Loaded XGBClassifier, or None for rule-based fallback.

    Returns:
        AnalysisResponse with triage tier, organ metrics, alerts, and
        a per-medication interaction table.
    """
    meds_lower = [m.lower() for m in profile.medicines]
    drug_classes = _map_to_drug_classes(meds_lower)

    # FIX M-2: Read real wearable values from the validated profile.
    # The original simulated only 2 hardcoded scalars (31.0 or 45.0 ms for HRV,
    # fixed skin_temp), which bears no resemblance to the continuous distribution
    # the model was trained on and ignores total_sleep / deep_sleep entirely.
    wearable = {
        "hrv":         profile.hrv,
        "skin_temp":   profile.skin_temp,
        "spo2":        profile.spo2,
        "glucose":     profile.glucose,
        "total_sleep": profile.total_sleep,
        "deep_sleep":  profile.deep_sleep,
    }

    # FIX C-1 + M-1: Build the feature vector by iterating FEATURE_ORDER.
    # This guarantees the column order and count always match training exactly.
    # Adding or removing a feature in FEATURE_ORDER is the single source of truth.
    feature_row: Dict[str, float] = {
        "egfr":            profile.egfr,
        "albumin":         profile.albumin,
        "haemoglobin":     profile.haemoglobin,
        "hrv":             wearable["hrv"],
        "skin_temp":       wearable["skin_temp"],
        "spo2":            wearable["spo2"],
        "glucose":         wearable["glucose"],
        "total_sleep":     wearable["total_sleep"],
        "deep_sleep":      wearable["deep_sleep"],
        **{k: float(v) for k, v in drug_classes.items()},
    }
    feature_vector = np.array(
        [[feature_row[f] for f in FEATURE_ORDER]], dtype=np.float64
    )

    # Rule-based fallback mirrors the training label logic so the demo
    # produces a coherent result even without a loaded model.
    rule_based_flag = (
        bool(drug_classes["has_antibiotic"])
        and bool(drug_classes["has_adaptogen"])
        and profile.egfr < 70
    )

    if classifier is not None:
        predicted_class = int(classifier.predict(feature_vector)[0])
        probability = float(classifier.predict_proba(feature_vector)[0][1])
    else:
        predicted_class = int(rule_based_flag)
        probability = 0.84 if rule_based_flag else 0.05

    is_high_risk = predicted_class == 1 or probability > ADR_PROBABILITY_THRESHOLD

    # ------------------------------------------------------------------
    # Build clinical narrative
    # ------------------------------------------------------------------
    alerts: List[str] = []
    symptoms: List[str] = []

    if is_high_risk:
        alerts.append(
            f"Potential adverse drug reaction pattern detected "
            f"(model confidence: {probability * 100:.1f}%)."
        )

        if profile.egfr < 70:
            symptoms.append(
                f"Renal system: reduced filtration clearance (eGFR {profile.egfr:.0f} mL/min). "
                "Drug serum concentration may be trending toward accumulation."
            )

        if drug_classes["has_antibiotic"] and drug_classes["has_adaptogen"]:
            symptoms.append(
                "Hepatic system: possible CYP3A4 metabolic pathway competition between "
                "the prescription antibiotic and the herbal adaptogen."
            )

        if drug_classes["has_antidiabetic"] and drug_classes["has_botanical"]:
            symptoms.append(
                "Glycaemic axis: concurrent antidiabetic and botanical compound may produce "
                "additive blood-glucose lowering. Monitor glucose closely."
            )

        if wearable["hrv"] < 35:
            symptoms.append(
                f"Cardiovascular system: autonomic tone suppression — HRV {wearable['hrv']:.1f} ms "
                "RMSSD consistent with parasympathetic withdrawal."
            )

        if wearable["skin_temp"] > 37.2:
            symptoms.append(
                f"Inflammatory axis: peripheral temperature {wearable['skin_temp']:.1f}°C consistent "
                "with an early systemic hypersensitivity response."
            )

        if wearable["deep_sleep"] < 0.8:
            symptoms.append(
                "Sleep architecture: disrupted deep (N3) sleep stage consistent with "
                "overnight autonomic micro-arousals driven by drug-herb metabolic burden."
            )

    # ------------------------------------------------------------------
    # Per-medication interaction table (FIX C-2)
    # Now driven by DRUG_CLASS_MAP, not hardcoded medicine names.
    # ------------------------------------------------------------------
    med_table: List[MedicationRow] = []
    for med in profile.medicines:
        matched_class = _classify_medicine(med.lower())

        if matched_class is not None and is_high_risk:
            row_status = _CLASS_STATUS[matched_class]
            interaction_note = _INTERACTION_NOTES[matched_class]
        else:
            row_status = "green"
            interaction_note = "Clear baseline"

        med_table.append(MedicationRow(
            name=med,
            dose="Standard dose",
            interaction=interaction_note,
            status=row_status,
        ))

    # ------------------------------------------------------------------
    # Organ metrics for the dashboard
    # ------------------------------------------------------------------
    metrics = OrganMetrics(
        hrv=wearable["hrv"],
        hrv_status="red" if is_high_risk and wearable["hrv"] < 35 else "green",
        temp=wearable["skin_temp"],
        temp_status="amber" if is_high_risk and wearable["skin_temp"] > 37.2 else "green",
        spo2=wearable["spo2"],
        glucose=wearable["glucose"],
        total_sleep=wearable["total_sleep"],
        deep_sleep=wearable["deep_sleep"],
    )

    return AnalysisResponse(
        status="success",
        metrics=metrics,
        alerts=alerts,
        symptoms=symptoms,
        med_table=med_table,
        model_confidence=round(probability, 4),
    )
