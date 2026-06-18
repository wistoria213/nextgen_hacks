"""
logging_setup.py
================
Centralised logger configuration for the RxGuard backend.

Import `logger` from this module in every other file instead of
calling logging.getLogger() directly. This guarantees all components
share a consistent format, timestamp style, and log level — which
matters when logs are ingested by cloud collectors (Datadog, GCP
Cloud Logging, etc.) that parse structured fields.
"""

import logging
import os

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

logger = logging.getLogger("RxGuard")
