"""
config.py
=========
Centralised, validated configuration for the RxGuard backend.

Design principle: fail loudly at startup rather than silently at
runtime. If a critical secret or setting is missing or weak, the
server refuses to start instead of running in an insecure state.

Cloud deployment notes:
  - PORT is read from the environment (injected by Render/Railway/Fly).
  - Set IN_CLOUD=true in your platform dashboard to prevent load_dotenv()
    from overwriting platform env vars with a stale local .env file.
  - REDIS_URL activates a Redis-backed rate limiter; omit it for
    single-instance deploys (in-memory limiter is used instead).
  - Never commit a .env file; set secrets in the platform dashboard.

Fixes applied vs. original:
  H-4  — HTTP origins rejected in production (HTTPS-only enforcement).
  M-5  — generate_sample_key() converted to @staticmethod.
  M-6  — Upper bound added to RATE_LIMIT_PER_MINUTE (max 10,000).
  misc — load_dotenv() guarded against cloud environments.
"""

import os
import secrets as _secrets
import sys
from pathlib import Path

# Load .env only in local dev. Set IN_CLOUD=true in your cloud platform
# dashboard to prevent a stale local .env from overwriting platform vars.
if not os.getenv("IN_CLOUD"):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv is optional; not needed in cloud deployments.

BASE_DIR = Path(__file__).resolve().parent


def _get_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        raise RuntimeError(
            f"Environment variable {name} must be an integer, got: {val!r}"
        )


def _require(name: str) -> str:
    """Fetch a required environment variable or refuse to start."""
    val = os.getenv(name)
    if not val or not val.strip():
        sys.stderr.write(
            f"\n[FATAL] Missing required environment variable: {name}\n"
            f"        Set it in your cloud platform's environment variables.\n"
            f"        Refusing to start with an undefined secret.\n\n"
        )
        raise SystemExit(1)
    return val.strip()


class Settings:
    """Immutable application settings, validated once at import time."""

    def __init__(self) -> None:
        self.debug_mode: bool = _get_bool("DEBUG_MODE", default=False)
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

        # --- Cloud server port ---
        # Most cloud platforms inject $PORT. Default to 8000 for local dev.
        self.port: int = _get_int("PORT", 8000)

        # --- CORS allow-list ---
        raw_origins = os.getenv("ALLOWED_ORIGINS", "")
        self.allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

        if not self.allowed_origins:
            sys.stderr.write(
                "\n[FATAL] ALLOWED_ORIGINS is empty.\n"
                "        Set it to your frontend's HTTPS URL, e.g.:\n"
                "        ALLOWED_ORIGINS=https://your-app.vercel.app\n\n"
            )
            raise SystemExit(1)

        if "*" in self.allowed_origins:
            sys.stderr.write(
                "\n[FATAL] ALLOWED_ORIGINS contains '*'.\n"
                "        Wildcard CORS is not permitted — list explicit origins.\n\n"
            )
            raise SystemExit(1)

        # FIX H-4: Reject plain HTTP origins in production environments.
        # Man-in-the-middle attackers on the same network can exploit HTTP
        # CORS origins to bypass same-origin protections.
        if not self.debug_mode:
            http_origins = [o for o in self.allowed_origins if o.startswith("http://")]
            if http_origins:
                sys.stderr.write(
                    f"\n[FATAL] Non-HTTPS origins detected in ALLOWED_ORIGINS:\n"
                    f"        {http_origins}\n"
                    f"        Production deployments must use HTTPS-only origins.\n"
                    f"        Use https:// prefixes or enable DEBUG_MODE for local dev.\n\n"
                )
                raise SystemExit(1)

        # --- API key (frontend → backend shared secret) ---
        self.api_key: str = _require("RXGUARD_API_KEY")

        if len(self.api_key) < 32:
            sys.stderr.write(
                "\n[FATAL] RXGUARD_API_KEY is too short (need >= 32 characters).\n"
                "        Generate one with:\n"
                '        python -c "import secrets; print(secrets.token_hex(32))"\n\n'
            )
            raise SystemExit(1)

        if self.api_key == "replace_this_with_a_long_random_value":
            sys.stderr.write(
                "\n[FATAL] RXGUARD_API_KEY is still set to the placeholder value.\n"
                "        Generate a real secret and set it in your cloud env vars.\n\n"
            )
            raise SystemExit(1)

        # --- Model artifact ---
        self.model_path: Path = BASE_DIR / os.getenv(
            "MODEL_PATH", "symbiote_classifier.pkl"
        )

        # --- Rate limiting ---
        self.rate_limit_per_minute: int = _get_int("RATE_LIMIT_PER_MINUTE", 30)

        # FIX M-6: Enforce both a lower AND upper bound.
        # An absurdly high value (e.g. 9999999) would silently disable rate limiting.
        if not (1 <= self.rate_limit_per_minute <= 10_000):
            raise RuntimeError(
                "RATE_LIMIT_PER_MINUTE must be between 1 and 10,000. "
                f"Got: {self.rate_limit_per_minute}"
            )

        # --- Optional Redis URL for multi-instance rate limiting ---
        # If set, uses a Redis-backed sliding window (required for 2+ workers).
        # If absent, falls back to in-memory limiter (fine for single instance).
        self.redis_url: str = os.getenv("REDIS_URL", "")

    # FIX M-5: Converted from instance method to @staticmethod.
    # The original version took `self` but never used it, making it
    # impossible to call without instantiating Settings first.
    @staticmethod
    def generate_sample_key() -> str:
        """Return a cryptographically secure 64-character hex API key."""
        return _secrets.token_hex(32)


settings = Settings()
