from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal
from datetime import datetime
class ExperimentConditions(BaseModel):
    ph: Optional[float] = Field(None, ge=0, le=14, description="pH value")
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    organism: Optional[str] = Field(None, description="Organism used")
    assay: Optional[str] = Field(None, description="Type of assay")
    additional: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional conditions")
class ExperimentOutcome(BaseModel):
    status: Literal["success", "partial", "fail", "unknown"] = Field(..., description="Experiment outcome")
    notes: Optional[str] = Field(None, description="Additional notes about the outcome")
    metrics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Quantitative metrics")
class ExperimentEvidence(BaseModel):
    paper_title: Optional[str] = Field(None, description="Title of the paper")
    doi: Optional[str] = Field(None, description="DOI identifier")
    arxiv_link: Optional[str] = Field(None, description="arXiv link")
    protocol_url: Optional[str] = Field(None, description="Protocol URL (e.g., protocols.io)")
    authors: Optional[list[str]] = Field(default_factory=list, description="List of authors")
    publication_date: Optional[str] = Field(None, description="Publication date")
class ScrapedExperiment(BaseModel):
    experiment_id: str = Field(..., description="Unique identifier for the experiment")
    description: str = Field(..., min_length=10, description="Experiment description")
    sequence: Optional[str] = Field(None, description="Biological sequence (DNA/RNA/Protein)")
    conditions: ExperimentConditions = Field(default_factory=ExperimentConditions)
    outcome: ExperimentOutcome
    evidence: ExperimentEvidence = Field(default_factory=ExperimentEvidence)
    source: str = Field(..., description="Source of the data (e.g., 'protocols.io')")
    scraped_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of scraping")
    raw_data: Optional[Dict[str, Any]] = Field(None, description="Original raw data from source")
    class Config:
        json_schema_extra = {
            "example": {
                "experiment_id": "protocolsio_12345",
                "description": "CRISPR-Cas9 genome editing in E. coli using pCas9 plasmid",
                "sequence": "ATGGCCAAGTTGACCAGTGCCGTTCCGGTGCTCACC",
                "conditions": {
                    "ph": 7.4,
                    "temperature": 37.0,
                    "organism": "E. coli",
                    "assay": "CRISPR editing efficiency"
                },
                "outcome": {
                    "status": "success",
                    "notes": "95% editing efficiency observed",
                    "metrics": {"efficiency": 0.95}
                },
                "evidence": {
                    "paper_title": "Efficient CRISPR editing in bacteria",
                    "doi": "10.1038/example.2024",
                    "protocol_url": "https://www.protocols.io/view/example"
                },
                "source": "protocols.io"
            }
        }