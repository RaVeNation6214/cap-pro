from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .api.routes import router

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
    ## Smart Contract Vulnerability Detection API

    AI-powered vulnerability detection for Ethereum smart contracts using
    a Hierarchical Transformer architecture.

    ### Features
    - **Multi-class Detection**: Identifies 4 vulnerability types
      - Arithmetic (Integer Overflow/Underflow)
      - Access Control (tx.origin Authentication)
      - Unchecked Calls (Missing Return Value Checks)
      - Reentrancy (External Call Before State Update)

    - **Explainability**: Attention weights show which code regions are risky

    - **Line-level Analysis**: Per-line risk scores and vulnerability markers

    ### Usage
    1. POST `/api/analyze` with your Solidity code
    2. Get vulnerability predictions with probabilities
    3. Review attention heatmap for explainability
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router, prefix=settings.API_V1_PREFIX)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "documentation": "/docs",
        "api_prefix": settings.API_V1_PREFIX,
        "demo_mode": settings.DEMO_MODE
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
