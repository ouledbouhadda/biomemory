from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
class ExperimentResponse(BaseModel):
    experiment_id: str
    text: str
    sequence: Optional[str] = None
    conditions: Optional[Dict[str, Any]] = None
    success: Optional[bool] = None
    notes: Optional[str] = None
    embedding_metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    message: Optional[str] = None
    class Config:
        json_schema_extra = {
            "example": {
                "experiment_id": "exp_12345",
                "text": "Expression of mutant TP53...",
                "sequence": "ATGGAGGAGCCG...",
                "conditions": {"organism": "human", "temperature": 37.0},
                "success": True,
                "created_at": "2024-01-15T10:30:00Z",
                "message": "Experiment uploaded successfully"
            }
        }
class ExperimentListResponse(BaseModel):
    experiments: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int
    class Config:
        json_schema_extra = {
            "example": {
                "experiments": [],
                "total": 10,
                "limit": 20,
                "offset": 0
            }
        }
class SearchResultResponse(BaseModel):
    experiment: ExperimentResponse
    score: float = Field(..., description="Original similarity score")
    reranked_score: float = Field(..., description="Dynamic reranked score")
    evidence: Dict[str, Any] = Field(..., description="Supporting evidence and sources")
    class Config:
        json_schema_extra = {
            "example": {
                "experiment": {
                    "id": "exp_12345",
                    "text": "Expression of...",
                    "success": True
                },
                "score": 0.85,
                "reranked_score": 0.91,
                "evidence": {
                    "source_type": "publication",
                    "publication": "Nature 2023"
                }
            }
        }
class ReproducibilityRiskResponse(BaseModel):
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk score (0=low, 1=high)")
    risk_level: str = Field(..., description="LOW, MEDIUM, or HIGH")
    variance: float = Field(..., description="Outcome variance in similar experiments")
    failure_count: int = Field(..., description="Number of failed similar experiments")
    success_count: int = Field(..., description="Number of successful similar experiments")
    common_failure_patterns: Optional[Dict[str, Any]] = Field(None, description="Identified failure patterns")
    class Config:
        json_schema_extra = {
            "example": {
                "risk_score": 0.35,
                "risk_level": "MEDIUM",
                "variance": 0.42,
                "failure_count": 3,
                "success_count": 7,
                "common_failure_patterns": {
                    "common_organisms": {"mouse": 2, "human": 1}
                }
            }
        }
class DesignVariantResponse(BaseModel):
    variant_id: str
    text: str
    sequence: Optional[str] = None
    conditions: Optional[Dict[str, Any]] = None
    modifications: List[str] = Field(default_factory=list, description="List of modifications")
    justification: str = Field(..., description="Scientific justification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    supporting_evidence: List[str] = Field(default_factory=list, description="Supporting experiment IDs")
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    class Config:
        json_schema_extra = {
            "example": {
                "variant_id": "var_001",
                "text": "Optimized protein expression protocol",
                "sequence": "ATGGCTAGC",
                "conditions": {"temperature": 30.0},
                "modifications": ["Reduce temperature to 30°C", "Increase induction time"],
                "justification": "Similar experiments succeeded at lower temperature",
                "confidence": 0.85,
                "supporting_evidence": ["exp_123", "exp_456"],
                "risk_factors": ["May reduce expression rate"]
            }
        }
class DesignResponse(BaseModel):
    variants: List[DesignVariantResponse]
    reproducibility_risk: float
    base_experiment: Dict[str, Any]
    generation_metadata: Dict[str, Any]
    class Config:
        json_schema_extra = {
            "example": {
                "variants": [],
                "reproducibility_risk": 0.25,
                "base_experiment": {},
                "generation_metadata": {"method": "gemini", "model": "gemini-pro"}
            }
        }
class SearchResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Search results")
    total_results: int = Field(..., description="Total number of results")
    reproducibility_risk: float = Field(..., ge=0.0, le=1.0, description="Reproducibility risk score")
    search_metadata: Dict[str, Any] = Field(default_factory=dict, description="Search metadata")
    traceability_stats: Optional[Dict[str, Any]] = Field(None, description="Evidence traceability stats")
    class Config:
        json_schema_extra = {
            "example": {
                "results": [],
                "total_results": 15,
                "reproducibility_risk": 0.3,
                "search_metadata": {
                    "modalities_used": {"has_text": True, "has_sequence": False},
                    "search_strategy": "text_similarity"
                },
                "traceability_stats": {
                    "total_items": 15,
                    "peer_reviewed": 10
                }
            }
        }
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    mfa_required: Optional[bool] = False
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "mfa_required": False
            }
        }
class UserResponse(BaseModel):
    email: str
    full_name: Optional[str] = None
    is_active: bool = True
    mfa_enabled: bool = False
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "full_name": "Dr. Jane Smith",
                "is_active": True,
                "mfa_enabled": False
            }
        }
class MFASetupResponse(BaseModel):
    secret: str
    qr_code: str
    instructions: str
    class Config:
        json_schema_extra = {
            "example": {
                "secret": "JBSWY3DPEHPK3PXP",
                "qr_code": "data:image/png;base64,iVBORw0KG...",
                "instructions": "Scan this QR code with your authenticator app"
            }
        }
class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall system status")
    version: str
    qdrant_cloud: str = Field(..., description="Qdrant Cloud status")
    qdrant_private: str = Field(..., description="Qdrant Private status")
    timestamp: datetime
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "qdrant_cloud": "healthy",
                "qdrant_private": "healthy",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Experiment not found",
                "error_code": "EXP_NOT_FOUND",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
class MessageResponse(BaseModel):
    message: str
    data: Optional[Dict[str, Any]] = None
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Operation completed successfully",
                "data": {"processed_items": 5}
            }
        }