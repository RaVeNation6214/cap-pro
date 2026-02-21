import logging
from fastapi import APIRouter, HTTPException

from .schemas import (
    AnalyzeRequest,
    AnalysisResult,
    SampleContractsResponse,
    HealthResponse,
    AIHelpRequest,
    AIHelpResponse,
)
from ..core.config import settings
from ..services.demo_mode import DemoModeAnalyzer
from ..services.gemini_service import GeminiService

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services (lazy - avoids slow startup)
_demo_analyzer: DemoModeAnalyzer = None
_gemini_service: GeminiService = None
_gnn_service = None


def get_demo_analyzer() -> DemoModeAnalyzer:
    global _demo_analyzer
    if _demo_analyzer is None:
        _demo_analyzer = DemoModeAnalyzer()
    return _demo_analyzer


def get_gnn_service():
    """Load GNN inference service (lazy, only when DEMO_MODE=False)."""
    global _gnn_service
    if _gnn_service is None:
        from ..services.gnn_inference import GNNInferenceService
        _gnn_service = GNNInferenceService(settings.MODEL_PATH)
    return _gnn_service


def get_gemini_service() -> GeminiService:
    global _gemini_service
    if _gemini_service is None:
        api_key = settings.GEMINI_API_KEY or None
        _gemini_service = GeminiService(api_key=api_key)
    return _gemini_service


@router.post("/analyze", response_model=AnalysisResult)
async def analyze_contract(request: AnalyzeRequest) -> AnalysisResult:
    """
    Analyze a smart contract for vulnerabilities.

    Uses the Hybrid GNN + GraphCodeBERT model when DEMO_MODE=False,
    or pattern-based demo mode otherwise.

    Returns:
        - Overall risk score and level
        - Per-vulnerability predictions with probabilities
        - Line-by-line risk analysis
        - Attention weights for explainability
        - Summary and recommendations
    """
    try:
        if len(request.code.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Contract code is too short. Please provide valid Solidity code."
            )

<<<<<<< HEAD
        if not settings.DEMO_MODE:
            # Use trained GNN model
            gnn = get_gnn_service()
            if gnn.is_loaded():
                # Get model probabilities, then enrich with demo analyzer formatting
                probs = gnn.predict(request.code)
                demo = get_demo_analyzer()
                result = demo.analyze_with_probs(request.code, probs)
                return result

        # Demo mode (pattern-based)
        analyzer = get_demo_analyzer()
        result = analyzer.analyze(request.code)
=======
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

>>>>>>> 7b835fe18c96efb700ebae38d468b68d763db934
        return result

    except HTTPException:
        raise
    except Exception as e:
<<<<<<< HEAD
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/ai-help", response_model=AIHelpResponse)
async def ai_help(request: AIHelpRequest) -> AIHelpResponse:
    """
    Get AI-powered explanation and fix suggestions for a detected vulnerability.

    Uses Google Gemini 1.5 Flash to generate personalized explanations.
    Falls back to static suggestions if Gemini is unavailable.

    Body:
        code: Solidity contract source code
        issue: Vulnerability type (reentrancy, access_control, arithmetic, unchecked_calls)
    """
    try:
        if len(request.code.strip()) < 5:
            raise HTTPException(
                status_code=400,
                detail="Contract code is required."
            )

        valid_issues = {"reentrancy", "access_control", "arithmetic", "unchecked_calls"}
        issue = request.issue.lower().replace(' ', '_').replace('-', '_')
        if issue not in valid_issues:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid issue type. Must be one of: {', '.join(valid_issues)}"
            )

        gemini = get_gemini_service()
        result = gemini.get_help(request.code, issue)

        return AIHelpResponse(
            response=result["response"],
            status=result["status"],
            model=result["model"],
=======
        logger.exception("Analysis failed")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
>>>>>>> 7b835fe18c96efb700ebae38d468b68d763db934
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI help request failed: {str(e)}")


@router.get("/sample-contracts", response_model=SampleContractsResponse)
async def get_sample_contracts() -> SampleContractsResponse:
    """
    Get sample smart contracts for demo purposes.
    Includes both synthetic and real contracts from the newALLBUGS dataset.
    """
    analyzer = get_demo_analyzer()
    contracts = analyzer.get_sample_contracts()
    return SampleContractsResponse(contracts=contracts)


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint with component status."""
    # Check which components are available
    try:
        from ..ml.embedding import TRANSFORMERS_AVAILABLE
        gcb_loaded = TRANSFORMERS_AVAILABLE
    except Exception:
        gcb_loaded = False

    try:
        from ..ml.gnn_model import TORCH_GEOMETRIC_AVAILABLE
        gnn_available = TORCH_GEOMETRIC_AVAILABLE
    except Exception:
        gnn_available = False

    try:
        from ..static_analysis.slither_features import SLITHER_AVAILABLE
        slither_avail = SLITHER_AVAILABLE
    except Exception:
        slither_avail = False

    # Check if trained model file exists
    import os
    model_loaded = os.path.exists(settings.MODEL_PATH) and not settings.DEMO_MODE

    gemini = get_gemini_service()

    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        demo_mode=settings.DEMO_MODE,
        model_loaded=model_loaded,
        graphcodebert_loaded=gcb_loaded,
        gnn_available=gnn_available,
        slither_available=slither_avail,
        gemini_available=gemini.is_available(),
    )


@router.get("/vulnerability-classes")
async def get_vulnerability_classes() -> dict:
    """Get vulnerability class information with detection thresholds."""
    from ..utils.suggestions import SUGGESTIONS, THRESHOLDS

    return {
        "classes": [
            {
                "id": "arithmetic",
                "name": "Arithmetic",
                "description": "Integer overflow/underflow vulnerabilities (SWC-101)",
                "severity": SUGGESTIONS["arithmetic"]["severity"],
                "threshold": THRESHOLDS["arithmetic"],
                "examples": ["Overflow in token transfer", "Underflow in balance subtraction"],
                "swc": "SWC-101",
            },
            {
                "id": "access_control",
                "name": "Access Control",
                "description": "Authentication and authorization vulnerabilities (SWC-115)",
                "severity": SUGGESTIONS["access_control"]["severity"],
                "threshold": THRESHOLDS["access_control"],
                "examples": ["tx.origin authentication bypass", "Missing access modifiers"],
                "swc": "SWC-115",
            },
            {
                "id": "unchecked_calls",
                "name": "Unchecked Calls",
                "description": "External calls without return value checks (SWC-104)",
                "severity": SUGGESTIONS["unchecked_calls"]["severity"],
                "threshold": THRESHOLDS["unchecked_calls"],
                "examples": ["Unchecked .send()", "Unchecked .call() return value"],
                "swc": "SWC-104",
            },
            {
                "id": "reentrancy",
                "name": "Reentrancy",
                "description": "Recursive call vulnerabilities (SWC-107)",
                "severity": SUGGESTIONS["reentrancy"]["severity"],
                "threshold": THRESHOLDS["reentrancy"],
                "examples": ["External call before state update", "Cross-function reentrancy"],
                "swc": "SWC-107",
            },
        ]
    }


@router.get("/model-info")
async def model_info() -> dict:
    """Get information about the ML model architecture."""
    return {
        "architecture": "Hybrid GNN + GraphCodeBERT",
        "components": {
            "graph_builder": {
                "name": "CFGBuilder",
                "description": "Builds Control Flow Graph from Solidity AST",
                "node_feature_dim": 12,
            },
            "embedder": {
                "name": "GraphCodeBERT",
                "model": "microsoft/graphcodebert-base",
                "embedding_dim": 768,
                "description": "Produces semantic embeddings for each function node",
            },
            "gnn": {
                "name": "HybridGNN (GAT)",
                "layers": ["GATConv(768→256, heads=4)", "GATConv(1024→256)", "GlobalMeanPool"],
                "description": "Propagates vulnerability signals across function call graph",
            },
            "classifier": {
                "name": "MLP",
                "layers": ["Linear(268→128)", "LayerNorm", "ReLU", "Dropout", "Linear(128→4)"],
                "output": "4 vulnerability probabilities",
            },
        },
        "dataset": {
            "primary": "SmartBugs (labeled vulnerability categories)",
            "secondary": "newALLBUGS (190 labeled contracts, 682 total)",
            "vulnerability_ids": {
                "31,32,36,89": "Arithmetic",
                "39": "Access Control",
                "40,42,43,82": "Unchecked Calls",
                "41": "Reentrancy",
            },
        },
        "training": {
            "loss": "BCEWithLogitsLoss with class weights",
            "optimizer": "AdamW (lr=1e-4)",
            "strategy": "GraphCodeBERT baseline → Add GNN → Fine-tune end-to-end",
        },
    }
