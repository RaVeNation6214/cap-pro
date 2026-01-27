from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum


class VulnerabilityType(str, Enum):
    """Vulnerability type enumeration."""
    ARITHMETIC = "Arithmetic"
    ACCESS_CONTROL = "Access Control"
    UNCHECKED_CALLS = "Unchecked Calls"
    REENTRANCY = "Reentrancy"


class AnalyzeRequest(BaseModel):
    """Request schema for contract analysis."""
    code: str = Field(..., min_length=1, description="Solidity contract source code")

    class Config:
        json_schema_extra = {
            "example": {
                "code": "pragma solidity ^0.8.0;\n\ncontract Vulnerable {\n    mapping(address => uint) balances;\n    \n    function withdraw() public {\n        uint amount = balances[msg.sender];\n        (bool success, ) = msg.sender.call{value: amount}(\"\");\n        require(success);\n        balances[msg.sender] = 0;\n    }\n}"
            }
        }


class LineRisk(BaseModel):
    """Risk information for a single line of code."""
    line_number: int = Field(..., ge=1, description="Line number (1-indexed)")
    content: str = Field(..., description="Line content")
    risk_score: float = Field(..., ge=0, le=1, description="Risk score for this line")
    is_vulnerable: bool = Field(default=False, description="Whether this line is flagged as vulnerable")


class VulnerabilityPrediction(BaseModel):
    """Prediction for a single vulnerability type."""
    type: VulnerabilityType
    probability: float = Field(..., ge=0, le=1, description="Probability of vulnerability presence")
    confidence: str = Field(..., description="Confidence level: Low, Medium, High")
    description: str = Field(..., description="Description of the vulnerability")
    affected_lines: List[int] = Field(default_factory=list, description="Lines affected by this vulnerability")


class AnalysisResult(BaseModel):
    """Complete analysis result."""
    overall_risk_score: float = Field(..., ge=0, le=1, description="Overall contract risk score")
    risk_level: str = Field(..., description="Risk level: Safe, Low, Medium, High, Critical")
    vulnerabilities: List[VulnerabilityPrediction] = Field(..., description="Detected vulnerabilities")
    line_risks: List[LineRisk] = Field(..., description="Per-line risk analysis")
    attention_weights: List[float] = Field(..., description="Attention weights for each code window")
    summary: str = Field(..., description="Summary of the analysis")
    recommendations: List[str] = Field(default_factory=list, description="Security recommendations")


class SampleContract(BaseModel):
    """Sample contract for demo purposes."""
    id: str
    name: str
    description: str
    code: str
    expected_vulnerabilities: List[str]


class SampleContractsResponse(BaseModel):
    """Response containing sample contracts."""
    contracts: List[SampleContract]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    demo_mode: bool
    model_loaded: bool
