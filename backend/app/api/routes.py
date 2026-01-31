import logging
from fastapi import APIRouter, HTTPException
from typing import List

from .schemas import (
    AnalyzeRequest,
    AnalysisResult,
    SampleContract,
    SampleContractsResponse,
    HealthResponse
)
from ..core.config import settings
from ..services.demo_mode import DemoModeAnalyzer

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize demo analyzer
demo_analyzer = DemoModeAnalyzer()


@router.post("/analyze", response_model=AnalysisResult)
async def analyze_contract(request: AnalyzeRequest) -> AnalysisResult:
    """
    Analyze a smart contract for vulnerabilities.

    Returns:
        - Overall risk score and level
        - Per-vulnerability predictions with probabilities
        - Line-by-line risk analysis
        - Attention weights for explainability
        - Summary and recommendations
    """
    try:
        # Validate input
        if len(request.code.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Contract code is too short. Please provide valid Solidity code."
            )

        logger.info("Analysis request received. code_chars=%s", len(request.code))
        logger.info("Demo mode=%s", settings.DEMO_MODE)

        # Use demo mode or real model based on settings
        if settings.DEMO_MODE:
            result = demo_analyzer.analyze(request.code)
        else:
            # TODO: Implement real model inference
            # For now, fall back to demo mode
            result = demo_analyzer.analyze(request.code)

        logger.info(
            "Analysis completed. risk_level=%s score=%s vulnerabilities=%s",
            result.risk_level,
            result.overall_risk_score,
            [v.type for v in result.vulnerabilities],
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.get("/sample-contracts", response_model=SampleContractsResponse)
async def get_sample_contracts() -> SampleContractsResponse:
    """
    Get sample smart contracts for demo purposes.

    Returns a list of contracts with known vulnerabilities for testing.
    """
    contracts = demo_analyzer.get_sample_contracts()
    return SampleContractsResponse(contracts=contracts)


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns the service status and configuration.
    """
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        demo_mode=settings.DEMO_MODE,
        model_loaded=not settings.DEMO_MODE  # False in demo mode
    )


@router.get("/vulnerability-classes")
async def get_vulnerability_classes() -> dict:
    """
    Get information about the vulnerability classes.
    """
    return {
        "classes": [
            {
                "name": "Arithmetic",
                "description": "Integer overflow/underflow vulnerabilities",
                "severity": "High",
                "examples": ["Overflow in token transfer", "Underflow in balance subtraction"]
            },
            {
                "name": "Access Control",
                "description": "Authentication and authorization vulnerabilities",
                "severity": "Critical",
                "examples": ["tx.origin authentication", "Missing access modifiers"]
            },
            {
                "name": "Unchecked Calls",
                "description": "External calls without proper return value checks",
                "severity": "High",
                "examples": ["Unchecked .call()", "Unchecked .send()"]
            },
            {
                "name": "Reentrancy",
                "description": "Recursive call vulnerabilities",
                "severity": "Critical",
                "examples": ["External call before state update", "Cross-function reentrancy"]
            }
        ]
    }
