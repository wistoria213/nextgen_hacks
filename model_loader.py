"""
model_loader.py
================
Loads the trained XGBoost classifier with a SHA-256 integrity check.

Why this matters
----------------
Python's pickle format (which joblib uses under the hood) can execute
arbitrary code when deserialised. A .pkl file is only as trustworthy
as its provenance — if an attacker could swap the file on disk,
loading it naively would run their code inside this server's process.

Mitigation implemented here
----------------------------
- After train_engine.py produces a new model, it calls
  record_hash_after_training() which writes a .sha256 sidecar file.
- On every subsequent startup, load_classifier() recomputes the hash
  of the model file on disk and compares it to the sidecar using
  hmac.compare_digest() — a constant-time function that does not
  short-circuit, closing the timing side-channel attack path.
- A mismatch (tampered, corrupted, or replaced file) causes the
  loader to return None rather than deserialise untrusted bytes.
  The API then starts in a degraded mode and returns HTTP 503.

Fixes applied vs. original
---------------------------
  H-1  — Plain == replaced with hmac.compare_digest() (constant-time).
  H-2  — record_hash_after_training() is now called by train_engine.py
          via an explicit import, not left as a dead utility.
  M-4  — sys.stderr.write() calls replaced with structured logger.
"""

import hashlib
import hmac
import sys
from pathlib import Path
from typing import Optional

import joblib

from config import settings
from logging_setup import logger


def _sha256_of(path: Path) -> str:
    """Compute the hex-encoded SHA-256 digest of a file in streaming chunks."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_classifier() -> Optional[object]:
    """
    Load the trained model, verifying integrity against the stored
    SHA-256 sidecar hash produced by record_hash_after_training().

    Returns None (rather than raising) if the model file is absent
    or the integrity check fails, so the API can start in a degraded
    mode and tell callers explicitly that inference is unavailable —
    instead of crashing the whole server on boot.
    """
    model_path = settings.model_path
    hash_path = model_path.with_suffix(model_path.suffix + ".sha256")

    if not model_path.exists():
        logger.warning(
            "Model file not found at '%s'. "
            "Run 'python train_engine.py' first. "
            "The API will start in degraded mode — /api/analyze-profile returns 503.",
            model_path,
        )
        return None

    current_hash = _sha256_of(model_path)

    if hash_path.exists():
        recorded_hash = hash_path.read_text().strip()

        # FIX H-1: Use constant-time comparison.
        # Plain string == short-circuits on the first differing byte; an
        # attacker who can measure response latency could exploit that to
        # reconstruct the expected hash one character at a time.
        if not hmac.compare_digest(current_hash, recorded_hash):
            logger.critical(
                "Integrity check FAILED for '%s'. "
                "The SHA-256 hash on disk does not match the recorded sidecar. "
                "Possible causes: model was retrained without updating the hash "
                "(re-run train_engine.py), file corruption, or tampering. "
                "Refusing to load an unverified model file.",
                model_path.name,
            )
            return None

        logger.info("Integrity check passed for '%s'.", model_path.name)

    else:
        # First boot with no sidecar — record the current hash and proceed.
        # On subsequent boots, the hash is verified rather than just accepted.
        hash_path.write_text(current_hash)
        logger.info(
            "No integrity sidecar found. Recorded initial hash for '%s'.",
            model_path.name,
        )

    try:
        model = joblib.load(model_path)
        logger.info("Model '%s' loaded successfully.", model_path.name)
    except Exception as exc:
        logger.critical(
            "Failed to deserialise model file '%s': %s",
            model_path.name,
            exc,
        )
        return None

    return model


def record_hash_after_training(model_path: Path) -> None:
    """
    Write (or overwrite) the SHA-256 sidecar file for a freshly trained model.

    Call this from train_engine.py immediately after joblib.dump() so that
    the integrity sidecar stays in sync with the model file on disk.
    Subsequent server startups will verify the new model against this hash.

    FIX H-2: This function must be called by train_engine.py. Without it,
    every retrain cycle produces a model with no matching sidecar, causing
    load_classifier() to accept the new file unconditionally on first boot
    and silently nullify the tamper-detection mechanism.
    """
    hash_path = model_path.with_suffix(model_path.suffix + ".sha256")
    hash_path.write_text(_sha256_of(model_path))
    logger.info(
        "Integrity hash recorded at '%s'.",
        hash_path.name,
    )
