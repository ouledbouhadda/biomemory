from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional, List, Dict, Any
from backend.models.requests import DesignRequest
from backend.models.responses import DesignResponse, DesignVariantResponse
from backend.agents.orchestrator import OrchestratorAgent
from backend.security.audit_logger import get_audit_logger
from backend.security.rate_limiting import RateLimiter
from backend.api.routes.auth import get_current_user
from backend.services.embedding_service import get_embedding_service
from backend.services.qdrant_service import get_qdrant_service
import uuid
import logging

logger = logging.getLogger("biomemory.design")
router = APIRouter()
orchestrator = OrchestratorAgent()
audit_logger = get_audit_logger()
design_rate_limiter = RateLimiter(max_requests=10, window_seconds=60)


def _build_variants(
    text: str, sequence: str, conditions: Dict[str, Any],
    neighbors: List[Dict], num_variants: int
) -> List[Dict[str, Any]]:
    """Generate design variants based on experiment type detected from text."""
    t = text.lower()
    temp = conditions.get("temperature", 37.0) or 37.0
    ph = conditions.get("ph", 7.0) or 7.0
    org = conditions.get("organism", "ecoli") or "ecoli"
    ev = [n.get("id", "") for n in neighbors[:3] if n.get("id")]

    is_pcr = any(k in t for k in ["pcr", "amplif", "primer", "polymerase", "annealing"])
    is_western = any(k in t for k in ["western", "blot", "immunoblot", "sds-page"])
    is_expr = any(k in t for k in ["expression", "recombinant", "iptg", "protein production"])
    is_extract = any(k in t for k in ["extraction", "purif", "isolat", "dna extract", "rna extract"])

    if is_pcr:
        pool = [
            {
                "id": f"var_{uuid.uuid4().hex[:6]}",
                "text": "Touchdown PCR: high-fidelity polymerase, gradient annealing 72->58°C",
                "sequence": sequence, "conditions": {"organism": org, "temperature": 98.0, "ph": ph},
                "modifications": ["Polymerase = Phusion/Q5 high-fidelity", "Annealing = touchdown 72->58°C (-1°C/cycle)", "Denaturation = 98°C/30s", "Extension = 72°C 30s/kb", "Cycles = 30"],
                "justification": "Touchdown PCR reduces non-specific amplification by progressively lowering annealing temperature",
                "confidence": 0.87, "risk_factors": ["Requires high-fidelity polymerase"], "supporting_evidence": ev
            },
            {
                "id": f"var_{uuid.uuid4().hex[:6]}",
                "text": "Hot-start PCR with 3% DMSO for GC-rich templates",
                "sequence": sequence, "conditions": {"organism": org, "temperature": 98.0, "ph": 8.3},
                "modifications": ["Polymerase = Q5 Hot Start", "DMSO = 3%", "Annealing = 62°C", "Extension = 72°C 20s/kb", "MgCl2 = 2.0mM"],
                "justification": "DMSO denatures secondary structures in GC-rich regions, improving specificity and yield",
                "confidence": 0.82, "risk_factors": ["DMSO >5% inhibits polymerase", "Optimize MgCl2"], "supporting_evidence": ev
            },
            {
                "id": f"var_{uuid.uuid4().hex[:6]}",
                "text": "Gradient PCR to determine optimal annealing Tm (55-68°C)",
                "sequence": sequence, "conditions": {"organism": org, "temperature": 95.0, "ph": ph},
                "modifications": ["Annealing gradient = 55-68°C (8 columns)", "MgCl2 = 2.5mM", "Primer = 0.4µM", "Template = 10ng", "Cycles = 28"],
                "justification": "Gradient PCR empirically identifies the optimal annealing temperature for the primer pair",
                "confidence": 0.79, "risk_factors": ["Requires gradient thermocycler", "Uses more reagents"], "supporting_evidence": ev[:1]
            },
            {
                "id": f"var_{uuid.uuid4().hex[:6]}",
                "text": "Two-step PCR with combined annealing/extension at 68°C",
                "sequence": sequence, "conditions": {"organism": org, "temperature": 98.0, "ph": ph},
                "modifications": ["Denaturation = 98°C/10s", "Annealing+Extension = 68°C 30s/kb", "No separate annealing", "Cycles = 25-30"],
                "justification": "Two-step PCR works well with primers Tm >65°C, reducing non-specific binding",
                "confidence": 0.75, "risk_factors": ["Only for primers with Tm >65°C"], "supporting_evidence": ev[:1]
            },
        ]
    elif is_western:
        pool = [
            {
                "id": f"var_{uuid.uuid4().hex[:6]}",
                "text": "PVDF membrane wet transfer with ECL detection",
                "sequence": sequence, "conditions": {"organism": org, "temperature": 4.0, "ph": 7.4},
                "modifications": ["Protein = 20-40µg/lane", "Gel = 10-12% SDS-PAGE", "Transfer = wet 100V/1h PVDF", "Blocking = 5% milk/TBST 1h RT", "Primary Ab = overnight 4°C"],
                "justification": "PVDF provides better protein retention and supports multiple stripping/reprobing cycles",
                "confidence": 0.85, "risk_factors": ["PVDF requires methanol activation"], "supporting_evidence": ev
            },
            {
                "id": f"var_{uuid.uuid4().hex[:6]}",
                "text": "Semi-dry transfer with BSA blocking for phospho-proteins",
                "sequence": sequence, "conditions": {"organism": org, "temperature": 4.0, "ph": 7.4},
                "modifications": ["Transfer = semi-dry 25V/30min", "Blocking = 5% BSA/TBST (not milk)", "Primary Ab = 1:1000 in 3% BSA/TBST", "Wash = 3x10min TBST"],
                "justification": "BSA blocking prevents casein cross-reactivity that causes high background with phospho-antibodies",
                "confidence": 0.83, "risk_factors": ["BSA more expensive", "Semi-dry less efficient for high MW"], "supporting_evidence": ev
            },
            {
                "id": f"var_{uuid.uuid4().hex[:6]}",
                "text": "Fluorescent Western blot with multiplex near-IR detection",
                "sequence": sequence, "conditions": {"organism": org, "temperature": 4.0, "ph": 7.4},
                "modifications": ["Secondary Ab = IRDye 680/800 conjugated", "Detection = near-infrared fluorescence", "Blocking = Odyssey buffer", "Quantification = linear dynamic range"],
                "justification": "Fluorescent detection enables multiplex analysis with linear quantification superior to chemiluminescence",
                "confidence": 0.78, "risk_factors": ["Requires fluorescence imager"], "supporting_evidence": ev[:1]
            },
        ]
    elif is_expr:
        pool = [
            {
                "id": f"var_{uuid.uuid4().hex[:6]}",
                "text": f"Low-temperature 18°C overnight induction for soluble protein in {org}",
                "sequence": sequence, "conditions": {"organism": org, "temperature": 18.0, "ph": ph},
                "modifications": ["Induction temp = 18°C", "Duration = 16-18h overnight", "IPTG = 0.1-0.3mM (reduced)", "OD600 at induction = 0.6-0.8"],
                "justification": "Lower temperature slows folding, reducing inclusion bodies and improving solubility",
                "confidence": 0.88, "risk_factors": ["Lower yield per cell", "Longer expression time"], "supporting_evidence": ev
            },
            {
                "id": f"var_{uuid.uuid4().hex[:6]}",
                "text": "Auto-induction with ZYM-5052 media at 25°C",
                "sequence": sequence, "conditions": {"organism": org, "temperature": 25.0, "ph": ph},
                "modifications": ["Media = ZYM-5052 auto-induction", "Temperature = 25°C", "Duration = 24h", "No manual IPTG needed"],
                "justification": "Auto-induction via lactose metabolism yields higher biomass and more consistent protein expression",
                "confidence": 0.82, "risk_factors": ["Specific media preparation required"], "supporting_evidence": ev
            },
            {
                "id": f"var_{uuid.uuid4().hex[:6]}",
                "text": f"High-density TB media expression at {temp}°C with rapid induction",
                "sequence": sequence, "conditions": {"organism": org, "temperature": temp, "ph": ph},
                "modifications": ["Media = Terrific Broth (TB)", f"Temperature = {temp}°C", "IPTG = 0.5-1.0mM", "Induction OD600 = 1.0-1.5", "Duration = 3-4h"],
                "justification": "TB supports higher cell density than LB, maximizing total protein yield",
                "confidence": 0.76, "risk_factors": ["Higher inclusion body risk at 37°C"], "supporting_evidence": ev[:1]
            },
        ]
    elif is_extract:
        pool = [
            {
                "id": f"var_{uuid.uuid4().hex[:6]}",
                "text": "Column-based extraction with RNase treatment",
                "sequence": sequence, "conditions": {"organism": org, "temperature": 4.0, "ph": 8.0},
                "modifications": ["Method = silica column (Qiagen/Zymo)", "RNase A treatment = 10min RT", "Elution = 50µL warm EB buffer", "A260/280 target > 1.8"],
                "justification": "Column purification provides consistent high-purity DNA with minimal hands-on time",
                "confidence": 0.85, "risk_factors": ["Lower yield for large fragments >20kb"], "supporting_evidence": ev
            },
            {
                "id": f"var_{uuid.uuid4().hex[:6]}",
                "text": "Phenol-chloroform extraction for maximum yield",
                "sequence": sequence, "conditions": {"organism": org, "temperature": 4.0, "ph": 8.0},
                "modifications": ["Lysis = SDS/proteinase K at 55°C", "Extraction = phenol:chloroform:IAA (25:24:1)", "Precipitation = ethanol + sodium acetate", "Wash = 70% ethanol"],
                "justification": "Phenol-chloroform gives highest yield and works for all fragment sizes including high MW genomic DNA",
                "confidence": 0.80, "risk_factors": ["Phenol is toxic", "More hands-on time", "Risk of contamination"], "supporting_evidence": ev
            },
            {
                "id": f"var_{uuid.uuid4().hex[:6]}",
                "text": "Magnetic bead extraction for high-throughput processing",
                "sequence": sequence, "conditions": {"organism": org, "temperature": 25.0, "ph": 8.0},
                "modifications": ["Method = SPRI magnetic beads", "Bead ratio = 1.8x for all fragments", "Elution = 30µL nuclease-free water", "Compatible with automation"],
                "justification": "Magnetic beads allow scalable, automatable purification with size selection capability",
                "confidence": 0.77, "risk_factors": ["Bead ratio critical for size selection"], "supporting_evidence": ev[:1]
            },
        ]
    else:
        pool = [
            {
                "id": f"var_{uuid.uuid4().hex[:6]}",
                "text": f"Optimized conditions: {temp - 2}°C, pH {ph + 0.2}",
                "sequence": sequence, "conditions": {"organism": org, "temperature": temp - 2, "ph": round(ph + 0.2, 1)},
                "modifications": [f"Temperature = {temp - 2}°C (reduced for stability)", f"pH = {round(ph + 0.2, 1)} (optimized for enzyme activity)", "Incubation time +15%"],
                "justification": "Lower temperature and adjusted pH improve reaction stability and enzyme kinetics",
                "confidence": 0.75, "risk_factors": ["May require longer incubation"], "supporting_evidence": ev
            },
            {
                "id": f"var_{uuid.uuid4().hex[:6]}",
                "text": f"Alternative organism: {'mouse' if org == 'human' else 'human'} model",
                "sequence": sequence, "conditions": {"organism": "mouse" if org == "human" else "human", "temperature": temp, "ph": ph},
                "modifications": [f"Organism = {'mouse' if org == 'human' else 'human'}", "Protocol adapted for cross-species", "Species-specific controls added"],
                "justification": "Alternative model organism provides complementary data and improves cross-system reproducibility",
                "confidence": 0.68, "risk_factors": ["Species-specific differences may affect results"], "supporting_evidence": ev[:1]
            },
            {
                "id": f"var_{uuid.uuid4().hex[:6]}",
                "text": f"High-stringency protocol at {temp + 3}°C, reduced volume",
                "sequence": sequence, "conditions": {"organism": org, "temperature": temp + 3, "ph": ph},
                "modifications": [f"Temperature = {temp + 3}°C", "Volume reduced 50%", "Reagent concentration 1.5x", "Additional wash steps"],
                "justification": "Higher stringency reduces non-specific interactions, improving signal-to-noise ratio",
                "confidence": 0.72, "risk_factors": ["May denature sensitive components"], "supporting_evidence": ev[:1]
            },
        ]
    return pool[:num_variants]


@router.post("/variants", response_model=DesignResponse)
async def generate_variants(
    request: DesignRequest,
    current_user: dict = Depends(get_current_user)
):
    allowed, info = design_rate_limiter.is_allowed(current_user["email"])
    if not allowed:
        audit_logger.log_event(
            event_type="design",
            user_id=current_user["email"],
            action="design_rate_limited",
            resource="design",
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Design rate limit exceeded. Try again in {info['reset_in']} seconds"
        )
    try:
        conditions = request.conditions.model_dump() if request.conditions else {}
        embedding_service = get_embedding_service()
        qdrant = get_qdrant_service()

        # 1) Generate embedding from base experiment
        embedding = await embedding_service.generate_multimodal_embedding(
            text=request.text,
            sequence=request.sequence or "",
            conditions=conditions,
        )

        # 2) Find similar experiments for context (Retrieval)
        neighbors = await qdrant.search(
            collection_name="public_science",
            query_vector=embedding.tolist(),
            limit=20,
            with_payload=True,
        )
        logger.info("Design: found %d similar experiments", len(neighbors))

        # 3) Try Gemini generation first (Generation via RAG)
        generated = []
        method = "context_based"
        try:
            from backend.services.gemini_service import get_gemini_service
            gemini = get_gemini_service()
            summaries = []
            for n in neighbors[:5]:
                p = n.get("payload", {})
                c = p.get("conditions", {})
                summaries.append(
                    f"- {p.get('text', '')[:200]} | success={p.get('success')} | "
                    f"organism={c.get('organism', 'N/A')} temp={c.get('temperature', 'N/A')}"
                )
            prompt = (
                f"You are a biological experiment design assistant. Generate {request.num_variants} "
                f"optimized protocol variants as a JSON array.\n\n"
                f"Base experiment: {request.text}\n"
                f"Sequence: {request.sequence or 'N/A'}\nConditions: {conditions}\n"
                f"Goal: {request.goal or 'optimize_protocol'}\n\n"
                f"Similar experiments:\n" + "\n".join(summaries) + "\n\n"
                f"Each variant: {{\"id\": \"var_xxx\", \"text\": \"desc\", "
                f"\"modifications\": [\"mod1\"], \"justification\": \"why\", "
                f"\"confidence\": 0.8, \"conditions\": {{}}, \"risk_factors\": [\"risk1\"]}}\n"
                f"Return ONLY the JSON array."
            )
            import json
            resp = await gemini.generate_content(prompt)
            json_str = resp
            if "```json" in resp:
                json_str = resp.split("```json")[1].split("```")[0].strip()
            elif "```" in resp:
                json_str = resp.split("```")[1].split("```")[0].strip()
            parsed = json.loads(json_str)
            if isinstance(parsed, list) and len(parsed) > 0:
                generated = parsed
                method = "gemini"
                logger.info("Design: Gemini generated %d variants", len(generated))
        except Exception as e:
            logger.warning("Design: Gemini unavailable (%s), using fallback", e)

        # 4) Fallback: context-based variant generation
        if not generated:
            generated = _build_variants(
                request.text, request.sequence or "", conditions,
                neighbors, request.num_variants
            )
            method = "context_based"

        # 5) Format response
        formatted = []
        for v in generated:
            vid = v.get("id", f"var_{uuid.uuid4().hex[:6]}")
            ev = v.get("supporting_evidence", [])
            if not ev:
                raw = v.get("evidence", {}).get("similar_successes", [])
                ev = [str(i.get("id", i)) if isinstance(i, dict) else str(i) for i in raw]
            mods = v.get("modifications", [])
            if not mods and v.get("modified_parameters"):
                mods = [f"{k} = {val}" for k, val in v["modified_parameters"].items()]
            vtxt = v.get("text", "")
            if not vtxt and v.get("modified_parameters"):
                vtxt = ", ".join(f"{k}: {val}" for k, val in v["modified_parameters"].items())
            formatted.append(DesignVariantResponse(
                variant_id=vid,
                text=vtxt or "Design variant",
                sequence=v.get("sequence", request.sequence),
                conditions=v.get("conditions", conditions),
                modifications=mods,
                justification=v.get("justification", "Based on similar successful experiments"),
                confidence=min(1.0, max(0.0, float(v.get("confidence", 0.7)))),
                supporting_evidence=ev[:5],
                risk_factors=v.get("risk_factors", [])
            ))

        # 6) Reproducibility risk from neighbors
        ok = sum(1 for n in neighbors if n.get("payload", {}).get("success", False))
        total = max(len(neighbors), 1)
        risk = round(1.0 - (ok / total), 3)

        audit_logger.log_event(
            event_type="design",
            user_id=current_user["email"],
            action="generate_variants",
            resource="design",
            success=True,
            details={
                "num_variants_requested": request.num_variants,
                "num_variants_generated": len(formatted),
                "goal": request.goal,
                "method": method
            }
        )
        return DesignResponse(
            variants=formatted,
            reproducibility_risk=risk,
            base_experiment={
                "text": request.text,
                "sequence": request.sequence,
                "conditions": conditions
            },
            generation_metadata={
                "method": method,
                "model": "gemini-1.5-pro" if method == "gemini" else "context_based",
                "context_experiments": len(neighbors)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Design failed: %s", e, exc_info=True)
        audit_logger.log_event(
            event_type="design",
            user_id=current_user["email"],
            action="generate_variants_failed",
            resource="design",
            success=False,
            details={"error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Variant generation failed: {str(e)}"
        )
@router.post("/optimize")
async def optimize_protocol(
    experiment_id: str,
    goal: str = "increase_yield",
    current_user: dict = Depends(get_current_user)
):
    allowed, info = design_rate_limiter.is_allowed(current_user["email"])
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Design rate limit exceeded. Try again in {info['reset_in']} seconds"
        )
    try:
        from backend.services.qdrant_service import get_qdrant_service
        qdrant = get_qdrant_service()
        point = await qdrant.retrieve(
            collection_name="private_experiments",
            ids=[experiment_id]
        )
        if not point or len(point) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
        payload = point[0].get('payload', {})
        design_request = DesignRequest(
            text=payload.get('text', ''),
            sequence=payload.get('sequence'),
            conditions=payload.get('conditions'),
            num_variants=3,
            goal=goal
        )
        result = await generate_variants(design_request, current_user)
        result_dict = result.model_dump()
        result_dict['optimization_goal'] = goal
        result_dict['base_experiment_id'] = experiment_id
        audit_logger.log_event(
            event_type="design",
            user_id=current_user["email"],
            action="optimize_protocol",
            resource=f"experiment:{experiment_id}",
            success=True,
            details={"goal": goal}
        )
        return result_dict
    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_event(
            event_type="design",
            user_id=current_user["email"],
            action="optimize_protocol_failed",
            resource=f"experiment:{experiment_id}",
            success=False,
            details={"error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Protocol optimization failed: {str(e)}"
        )
@router.post("/troubleshoot")
async def troubleshoot_failure(
    failed_experiment_id: str,
    current_user: dict = Depends(get_current_user)
):
    allowed, info = design_rate_limiter.is_allowed(current_user["email"])
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Design rate limit exceeded. Try again in {info['reset_in']} seconds"
        )
    try:
        from backend.services.qdrant_service import get_qdrant_service
        qdrant = get_qdrant_service()
        point = await qdrant.retrieve(
            collection_name="private_experiments",
            ids=[failed_experiment_id]
        )
        if not point or len(point) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {failed_experiment_id} not found"
            )
        payload = point[0].get('payload', {})
        if payload.get('success', True):
            return {
                "message": "This experiment was marked as successful. Troubleshooting is for failed experiments.",
                "experiment_id": failed_experiment_id,
                "success": True
            }
        user_input = {
            "intent": "troubleshoot_failure",
            "failed_experiment": {
                "id": failed_experiment_id,
                "text": payload.get('text', ''),
                "sequence": payload.get('sequence'),
                "conditions": payload.get('conditions'),
                "notes": payload.get('notes')
            },
            "user_id": current_user["email"]
        }
        result = await orchestrator.process_request(user_input)
        troubleshooting_report = {
            "failed_experiment_id": failed_experiment_id,
            "probable_causes": result.get('probable_causes', []),
            "suggested_fixes": result.get('design_variants', []),
            "successful_references": result.get('reranked_neighbors', [])[:5],
            "key_differences": result.get('key_differences', []),
            "reproducibility_analysis": {
                "risk": result.get('reproducibility_risk', 0.0),
                "similar_failures": result.get('similar_failures', 0),
                "similar_successes": result.get('similar_successes', 0)
            }
        }
        audit_logger.log_event(
            event_type="design",
            user_id=current_user["email"],
            action="troubleshoot_failure",
            resource=f"experiment:{failed_experiment_id}",
            success=True,
            details={
                "causes_identified": len(troubleshooting_report['probable_causes']),
                "fixes_suggested": len(troubleshooting_report['suggested_fixes'])
            }
        )
        return troubleshooting_report
    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_event(
            event_type="design",
            user_id=current_user["email"],
            action="troubleshoot_failure_failed",
            resource=f"experiment:{failed_experiment_id}",
            success=False,
            details={"error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Troubleshooting failed: {str(e)}"
        )
@router.get("/templates")
async def get_design_templates(
    category: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    try:
        templates = {
            "protein_expression": {
                "name": "Protein Expression in E. coli",
                "text": "Express recombinant protein in E. coli BL21(DE3) cells",
                "conditions": {
                    "organism": "ecoli",
                    "temperature": 37.0,
                    "ph": 7.0,
                    "duration_hours": 4.0
                },
                "typical_parameters": {
                    "IPTG_concentration": "0.5mM",
                    "induction_temperature": "37°C or 18°C",
                    "growth_media": "LB or TB"
                }
            },
            "cell_culture": {
                "name": "Mammalian Cell Culture",
                "text": "Culture mammalian cells for protein production",
                "conditions": {
                    "organism": "human",
                    "temperature": 37.0,
                    "co2_percent": 5.0
                },
                "typical_parameters": {
                    "media": "DMEM + 10% FBS",
                    "passage_ratio": "1:3 to 1:6",
                    "confluence": "80-90%"
                }
            },
            "western_blot": {
                "name": "Western Blot Analysis",
                "text": "Protein detection via Western blot",
                "conditions": {
                    "temperature": 4.0,
                    "duration_hours": 16.0
                },
                "typical_parameters": {
                    "gel_percentage": "10-12%",
                    "transfer_method": "wet or semi-dry",
                    "blocking": "5% milk or BSA"
                }
            },
            "pcr": {
                "name": "PCR Amplification",
                "text": "Amplify DNA fragment using PCR",
                "conditions": {
                    "temperature": 95.0,
                    "duration_hours": 1.0
                },
                "typical_parameters": {
                    "annealing_temp": "55-65°C",
                    "extension_time": "1min/kb",
                    "cycles": "25-35"
                }
            }
        }
        if category:
            if category not in templates:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Template category '{category}' not found"
                )
            return {category: templates[category]}
        return templates
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve templates: {str(e)}"
        )