from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from enum import Enum
class OrganismType(str, Enum):
    HUMAN = "human"
    MOUSE = "mouse"
    ECOLI = "ecoli"
    YEAST = "yeast"
    ARABIDOPSIS = "arabidopsis"
    DROSOPHILA = "drosophila"
    OTHER = "other"
class ExperimentConditions(BaseModel):
    organism: Optional[OrganismType] = None
    temperature: Optional[float] = Field(None, ge=-273.15, le=200, description="Temperature in Celsius")
    ph: Optional[float] = Field(None, ge=0, le=14, description="pH value")
    protocol_id: Optional[str] = Field(None, description="Protocol reference ID")
    additional_params: Optional[Dict[str, Any]] = Field(None, description="Additional experimental parameters")
    class Config:
        json_schema_extra = {
            "example": {
                "organism": "human",
                "temperature": 37.0,
                "ph": 7.4,
                "protocol_id": "PROT-001"
            }
        }
class ExperimentUploadRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000, description="Experiment description or protocol")
    sequence: Optional[str] = Field(None, max_length=50000, description="Biological sequence (DNA/RNA/Protein)")
    image_base64: Optional[str] = Field(None, description="Base64 encoded experiment image")
    conditions: Optional[ExperimentConditions] = None
    success: Optional[bool] = Field(None, description="Was the experiment successful?")
    source: Optional[str] = Field(None, description="Source reference (DOI, PubMed ID, etc.)")
    notes: Optional[str] = Field(None, max_length=5000, description="Additional notes")
    @field_validator('sequence')
    @classmethod
    def validate_sequence(cls, v):
        if v is None:
            return v
        valid_chars = set('ATGCURYKMSWBDHVN-')
        if not all(c.upper() in valid_chars for c in v):
            raise ValueError("Invalid sequence characters. Use standard IUPAC codes")
        return v.upper()
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Expression of mutant TP53 in HEK293 cells using pCMV vector",
                "sequence": "ATGGAGGAGCCGCAGTCAGAT",
                "conditions": {
                    "organism": "human",
                    "temperature": 37.0,
                    "ph": 7.4
                },
                "success": True,
                "source": "DOI:10.1234/example"
            }
        }
class SearchRequest(BaseModel):
    text: Optional[str] = Field(None, max_length=5000, description="Search query text")
    sequence: Optional[str] = Field(None, max_length=50000, description="Biological sequence to search")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image to search")
    conditions: Optional[ExperimentConditions] = Field(None, description="Filter by experimental conditions")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    include_failures: bool = Field(True, description="Include failed experiments in results")
    similarity_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity score")
    class Config:
        json_schema_extra = {
            "example": {
                "text": "protein expression in mammalian cells",
                "conditions": {
                    "organism": "human",
                    "temperature": 37.0
                },
                "limit": 10,
                "include_failures": True
            }
        }
class DesignRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Base experiment description")
    sequence: Optional[str] = Field(None, description="Base sequence")
    conditions: Optional[ExperimentConditions] = None
    num_variants: int = Field(3, ge=1, le=10, description="Number of variants to generate")
    goal: Optional[str] = Field(None, description="Optimization goal")
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Protein expression optimization",
                "sequence": "ATGGCTAGC",
                "num_variants": 3,
                "goal": "increase_yield"
            }
        }
class DesignVariantRequest(BaseModel):
    experiment_id: Optional[str] = Field(None, description="Base experiment ID")
    base_experiment: Optional[ExperimentUploadRequest] = Field(None, description="Base experiment data")
    num_variants: int = Field(3, ge=1, le=10, description="Number of variants to generate")
    exploration_factor: float = Field(0.5, ge=0.0, le=1.0, description="0=conservative, 1=explorative")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Design constraints")
    class Config:
        json_schema_extra = {
            "example": {
                "experiment_id": "exp_12345",
                "num_variants": 3,
                "exploration_factor": 0.5
            }
        }
class UserRegisterRequest(BaseModel):
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$', description="User email")
    password: str = Field(..., min_length=8, description="Password (min 8 characters)")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name")
    class Config:
        json_schema_extra = {
            "example": {
                "email": "researcher@university.edu",
                "password": "SecurePass123!",
                "full_name": "Dr. Jane Smith"
            }
        }
class UserLoginRequest(BaseModel):
    email: str
    password: str
    mfa_token: Optional[str] = Field(None, description="MFA token if enabled")
    class Config:
        json_schema_extra = {
            "example": {
                "email": "researcher@university.edu",
                "password": "SecurePass123!"
            }
        }
class MFAVerifyRequest(BaseModel):
    code: str = Field(..., min_length=6, max_length=6, description="6-digit MFA code")
    class Config:
        json_schema_extra = {
            "example": {
                "code": "123456"
            }
        }