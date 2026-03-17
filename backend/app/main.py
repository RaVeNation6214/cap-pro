import time
import logging
import logging.config

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .api.routes import router

# ── Minimal structured logging ─────────────────────────────────────────────
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(asctime)s [%(levelname).1s] %(name)s › %(message)s",
            "datefmt": "%H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        }
    },
    "root": {"level": "WARNING", "handlers": ["console"]},
    "loggers": {
        "app":              {"level": "INFO",    "propagate": False, "handlers": ["console"]},
        "uvicorn.error":    {"level": "WARNING", "propagate": False, "handlers": ["console"]},
        "uvicorn.access":   {"level": "WARNING", "propagate": False, "handlers": ["console"]},
    },
})

log = logging.getLogger("app")

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SmartAudit API",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request timing middleware (minimal) ────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    if request.url.path.startswith("/api"):
        status = response.status_code
        symbol = "✓" if status < 400 else "✗"
        log.info(f"{symbol} {request.method} {request.url.path} → {status} ({ms:.0f}ms)")
    return response

app.include_router(router, prefix=settings.API_V1_PREFIX)

@app.on_event("startup")
async def on_startup():
    log.info(f"SmartAudit API v{settings.VERSION} — demo_mode={settings.DEMO_MODE}")
    log.info(f"CORS: {len(settings.CORS_ORIGINS)} origins  |  docs → /docs")

@app.get("/")
async def root():
    return {"name": "SmartAudit API", "version": settings.VERSION, "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8002, reload=False)
