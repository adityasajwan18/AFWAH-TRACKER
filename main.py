# ============================================================
# main.py — FastAPI Application Entry Point
# Run with: uvicorn main:app --reload --port 8000
# ============================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.core.config import settings

# ── App Initialization ────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Real-time Misinformation Detection, Tracking & Prediction Platform",
    docs_url="/docs",        # Swagger UI at /docs
    redoc_url="/redoc",      # ReDoc UI at /redoc
)

# ── CORS Middleware ───────────────────────────────────────────
# Allows our frontend (React or plain HTML) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Route Registration ────────────────────────────────────────
from backend.api.routes import posts, analyze
app.include_router(posts.router,   prefix="/api", tags=["Posts"])
app.include_router(analyze.router, prefix="/api", tags=["Analysis"])
from backend.api.routes import graph
app.include_router(graph.router,   prefix="/api", tags=["Graph"])
from backend.api.routes import image
app.include_router(image.router,   prefix="/api", tags=["Image Analysis"])

# ── Health Check ──────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint — confirms the API is alive."""
    return {
        "status": "🟢 Online",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check for monitoring."""
    from backend.ml.classifier import get_model_status
    return JSONResponse(
        content={
            "status": "healthy",
            "services": {
                "api": "up",
                "mongodb": "pending_connection",   # Connected in Phase 3
                "neo4j": "pending_connection",     # Connected in Phase 3
                "ml_model": get_model_status(),
            },
        }
    )


# ── Dev Entry Point ───────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info",
    )
