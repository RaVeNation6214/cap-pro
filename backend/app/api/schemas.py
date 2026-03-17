from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum


class VulnerabilityType(str, Enum):
    REENTRANCY = "Reentrancy"
    ARITHMETIC = "Arithmetic"
    ACCESS_CONTROL = "Access Control"
    UNCHECKED_CALLS = "Unchecked Calls"
    TIMESTAMP = "Timestamp"


class AnalyzeRequest(BaseModel):
    code: str = Field(..., min_length=1)

    class Config:
        json_schema_extra = {
            "example": {"code": "pragma solidity ^0.8.0;\ncontract Vulnerable {}"}
        }


class LineRisk(BaseModel):
    line_number: int = Field(..., ge=1)
    content: str
    risk_score: float = Field(..., ge=0, le=1)
    is_vulnerable: bool = False


class VulnerabilityPrediction(BaseModel):
    type: VulnerabilityType
    probability: float = Field(..., ge=0, le=1)
    confidence: str
    description: str
    affected_lines: List[int] = Field(default_factory=list)


class CFGNode(BaseModel):
    id: str
    label: str
    node_type: str
    risk_score: float = 0.0
    attention_weight: float = 0.0
    lines: List[int] = Field(default_factory=list)
    vuln_types: List[str] = Field(default_factory=list)


class CFGEdge(BaseModel):
    source: str
    target: str
    edge_type: str = "control"


class FunctionMetric(BaseModel):
    name: str
    risk_score: float
    top_vuln: str
    gat_score: float
    loc: int
    vulnerability_count: int


class SlitherFinding(BaseModel):
    check: str
    impact: str
    confidence: str
    description: str
    lines: List[int] = Field(default_factory=list)


class SlitherComparison(BaseModel):
    slither_available: bool
    slither_findings: List[SlitherFinding] = Field(default_factory=list)
    model_findings: List[str] = Field(default_factory=list)
    agreement: List[str] = Field(default_factory=list)
    model_only: List[str] = Field(default_factory=list)
    slither_only: List[str] = Field(default_factory=list)
    winner: str = "model"


class AnalysisResult(BaseModel):
    overall_risk_score: float = Field(..., ge=0, le=1)
    risk_level: str
    vulnerabilities: List[VulnerabilityPrediction]
    line_risks: List[LineRisk]
    attention_weights: List[float]
    summary: str
    recommendations: List[str] = Field(default_factory=list)
    cfg_nodes: List[CFGNode] = Field(default_factory=list)
    cfg_edges: List[CFGEdge] = Field(default_factory=list)
    radar_data: List[Dict[str, Any]] = Field(default_factory=list)
    function_metrics: List[FunctionMetric] = Field(default_factory=list)
    slither_comparison: Optional[SlitherComparison] = None
    contract_address: Optional[str] = None
    chain: Optional[str] = None


class SampleContract(BaseModel):
    id: str
    name: str
    description: str
    code: str
    expected_vulnerabilities: List[str]


class SampleContractsResponse(BaseModel):
    contracts: List[SampleContract]


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    version: str
    demo_mode: bool
    model_loaded: bool
    graphcodebert_loaded: bool = False
    gnn_available: bool = False
    slither_available: bool = False
    gemini_available: bool = False


class AIHelpRequest(BaseModel):
    code: str
    issue: str


class AIHelpResponse(BaseModel):
    response: str
    status: str
    model: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    contract_context: Optional[str] = None
    analysis_context: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    model: str
    status: str


class SlitherRequest(BaseModel):
    code: str


class SlitherResponse(BaseModel):
    available: bool
    findings: List[SlitherFinding] = Field(default_factory=list)
    raw_output: str = ""
    error: Optional[str] = None


class MythrilFinding(BaseModel):
    swc: str
    title: str
    severity: str
    confidence: str
    description: str
    lines: List[int] = Field(default_factory=list)


class MythrilRequest(BaseModel):
    code: str


class MythrilResponse(BaseModel):
    available: bool
    findings: List[MythrilFinding] = Field(default_factory=list)
    error: Optional[str] = None
