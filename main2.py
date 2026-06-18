"""
main.py
========
RxGuard cloud inference server.

Local dev:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Cloud (auto-detects $PORT injected by Render/Railway/Fly/Heroku):
    gunicorn main:app -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT --workers 1

Before deploying:
    1. Set all env vars in your platform dashboard (see .env.example).
    2. Set Build Command to: pip install -r requirements.txt && python train_engine.py
    3. Set Start Command to: gunicorn main:app -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT --workers 1
    4. Set ALLOWED_ORIGINS to your frontend's HTTPS URL (e.g. https://yourapp.vercel.app).
    5. Never commit .env or *.pkl to version control.

Security posture:
  - CORS:    Explicit allow-list loaded from config.py; never wildcard.
  - Auth:    Constant-time API key check via hmac.compare_digest (security.py).
  - Rate:    Per-IP sliding window; Redis-backed if REDIS_URL is set (security.py).
  - Input:   All fields range/pattern checked before reaching the model (schemas.py).
  - Model:   SHA-256 hash verified on every startup (model_loader.py).
  - Errors:  Full detail logged server-side; only generic message sent to client.
  - Headers: HSTS, CSP, nosniff, no-referrer, no-store on every response.
  - PII:     No personally identifying fields accepted by the inference endpoint.

Fixes applied vs. original:
  C-3  — logging_setup.py and security.py now exist (they were missing).
  C-4  — Classifier loaded inside lifespan → stored in app.state.
          Original loaded it at module scope outside lifespan, which is
          not thread-safe, cannot be refreshed without a full restart, and
          runs before any logging is configured.
  H-3  — Content-Security-Policy header added to the security middleware.
  H-5  — 503 path raises HTTPException instead of returning JSONResponse
          inside a response_model=AnalysisResponse endpoint (which bypasses
          schema validation and breaks the OpenAPI contract).
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from logging_setup import logger
from model_loader import load_classifier
from schemas import AnalysisResponse, ErrorResponse, HealthProfile
from security import enforce_rate_limit, get_rate_limiter_backend, verify_api_key
import inference


# ---------------------------------------------------------------------------
# Lifespan — model loading on startup (FIX C-4)
# ---------------------------------------------------------------------------
# The original code called load_classifier() at module scope, before
# lifespan ran and before logging was configured. Storing the classifier
# in app.state means:
#   - Loading happens inside the async lifecycle, after logging is ready.
#   - Every request handler accesses it through the request object, making
#     the dependency explicit and testable (you can override app.state in
#     tests without patching globals).
#   - A future hot-reload endpoint can replace app.state.classifier without
#     restarting the process.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.classifier = load_classifier()
    rl_backend = get_rate_limiter_backend()

    if app.state.classifier is None:
        logger.warning(
            "Server starting WITHOUT a verified model loaded. "
            "/api/analyze-profile will return 503 until train_engine.py is run."
        )
    else:
        logger.info(
            "RxGuard server started. Model: loaded. Rate-limiter: %s. Port: %s.",
            rl_backend,
            settings.port,
        )

    yield

    # Teardown (placeholder — extend when connections / pools need cleanup)
    app.state.classifier = None
    logger.info("RxGuard server shutting down.")


app = FastAPI(
    title="RxGuard Inference API",
    description="Cloud clinical decision-support inference service. Not a medical device.",
    version="2.0.0",
    # Docs endpoints only exposed in debug mode — no public API surface in prod.
    docs_url="/docs" if settings.debug_mode else None,
    redoc_url="/redoc" if settings.debug_mode else None,
    openapi_url="/openapi.json" if settings.debug_mode else None,
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — explicit allow-list loaded from validated settings.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "X-API-Key"],
)


# ---------------------------------------------------------------------------
# Security response headers on every response (FIX H-3: added CSP).
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    # HSTS — instructs browsers to only connect via HTTPS for 1 year.
    # Cloud platforms terminate TLS before reaching this app, so this is safe.
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    # FIX H-3: Content-Security-Policy — the most effective header against XSS.
    # Tighten src directives further once you know the full asset origin list.
    response.headers["Content-Security-Policy"] = (
        "default-src 'none'; "
        "script-src 'self'; "
        "connect-src 'self'; "
        "img-src 'self'; "
        "style-src 'self'; "
        "frame-ancestors 'none';"
    )
    return response


# ---------------------------------------------------------------------------
# Global exception handler — never leak stack traces to the client.
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,  # attach full traceback to the server log
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            message="An internal error occurred. Please try again."
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Health check — unauthenticated, reveals only liveness + model status.
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health_check(request: Request):
    # FIX C-4: Read model status from app.state, not a module-level global.
    return {
        "status": "ok",
        "model_loaded": request.app.state.classifier is not None,
        "rate_limiter": get_rate_limiter_backend(),
    }


# ---------------------------------------------------------------------------
# Core inference endpoint — authenticated and rate-limited.
# ---------------------------------------------------------------------------
@app.post(
    "/api/analyze-profile",
    response_model=AnalysisResponse,
    dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)],
)
async def analyze_profile(request: Request, profile: HealthProfile) -> AnalysisResponse:
    # FIX C-4: Read the classifier from app.state instead of a module global.
    classifier = request.app.state.classifier

    # FIX H-5: Raise HTTPException instead of returning a bare JSONResponse.
    # Returning JSONResponse inside a response_model=AnalysisResponse endpoint
    # silently bypasses Pydantic serialisation and breaks the OpenAPI contract —
    # the 503 branch never appears in the generated schema docs.
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model is not available. Run train_engine.py and restart.",
        )

    logger.info(
        "Analysis request received (medicines=%d, egfr=%.1f, hrv=%.1f).",
        len(profile.medicines),
        profile.egfr,
        profile.hrv,
    )

    result = inference.assess_risk(profile, classifier)

    logger.info(
        "Analysis complete (high_risk=%s, confidence=%.4f).",
        bool(result.alerts),
        result.model_confidence or 0.0,
    )

    return result
