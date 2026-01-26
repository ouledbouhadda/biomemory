from typing import Dict, Any, List
from backend.services.gemini_service import get_gemini_service
import uuid
class DesignAgent:
    def __init__(self):
        self.gemini = get_gemini_service()
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        neighbors = context.get('reranked_neighbors', [])
        failures = context.get('failures', [])
        successful_neighbors = [
            n for n in neighbors
            if n['payload'].get('success', False)
        ]
        if len(successful_neighbors) < 1:
            return {
                'design_variants': self._generate_basic_variants(neighbors)
            }
        try:
            variants = await self._generate_variants_with_gemini(
                successful_neighbors,
                failures
            )
        except Exception as e:
            print(f"⚠️ Gemini generation failed: {e}")
            variants = self._generate_basic_variants(successful_neighbors)
        justified_variants = self._add_justifications(
            variants,
            successful_neighbors
        )
        return {
            'design_variants': justified_variants
        }
    async def _generate_variants_with_gemini(
        self,
        successes: List[Dict],
        failures: List[Dict]
    ) -> List[Dict]:
        success_summary = self._summarize_experiments(successes)
        failure_summary = self._summarize_experiments(failures)
        prompt = f
        response = await self.gemini.generate_content(prompt)
        variants = self._parse_gemini_variants(response)
        return variants
    def _summarize_experiments(self, experiments: List[Dict]) -> str:
        if not experiments:
            return "None"
        summaries = []
        for exp in experiments[:5]:
            payload = exp.get('payload', {})
            summary = f
            summaries.append(summary)
        return "\n".join(summaries)
    def _parse_gemini_variants(self, response: str) -> List[Dict]:
        import json
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            variants = json.loads(json_str)
            normalized = []
            for v in variants:
                normalized.append({
                    'id': v.get('id', f"var_{uuid.uuid4().hex[:6]}"),
                    'modified_parameters': v.get('modified_parameters', {}),
                    'justification': v.get('justification', 'No justification provided'),
                    'confidence': float(v.get('confidence', 0.5))
                })
            return normalized
        except Exception as e:
            print(f"⚠️ Failed to parse Gemini response: {e}")
            return []
    def _generate_basic_variants(self, experiments: List[Dict]) -> List[Dict]:
        if not experiments:
            return []
        best = experiments[0]
        best_cond = best['payload'].get('conditions', {})
        variants = []
        if best_cond.get('temperature'):
            variants.append({
                'id': f"var_{uuid.uuid4().hex[:6]}",
                'modified_parameters': {
                    'temperature': best_cond['temperature'] - 1.0
                },
                'justification': 'Slightly lower temperature may improve stability',
                'confidence': 0.7
            })
        if best_cond.get('ph'):
            variants.append({
                'id': f"var_{uuid.uuid4().hex[:6]}",
                'modified_parameters': {
                    'ph': best_cond['ph'] + 0.2
                },
                'justification': 'Slightly higher pH may optimize enzyme activity',
                'confidence': 0.65
            })
        variants.append({
            'id': f"var_{uuid.uuid4().hex[:6]}",
            'modified_parameters': {
                'organism': 'mouse' if best_cond.get('organism') == 'human' else 'human'
            },
            'justification': 'Alternative model organism may provide complementary insights',
            'confidence': 0.6
        })
        return variants
    def _add_justifications(
        self,
        variants: List[Dict],
        successes: List[Dict]
    ) -> List[Dict]:
        for variant in variants:
            variant['evidence'] = {
                'similar_successes': [
                    {
                        'id': s.get('id'),
                        'similarity': s.get('score', 0),
                        'source': s['payload'].get('source', 'unknown')
                    }
                    for s in successes[:3]
                ],
                'rationale': variant.get('justification', ''),
                'expected_outcome': self._predict_outcome(variant, successes)
            }
        return variants
    def _predict_outcome(
        self,
        variant: Dict,
        successes: List[Dict]
    ) -> str:
        confidence = variant.get('confidence', 0.5)
        if confidence > 0.8:
            return "High probability of success based on similar successful experiments"
        elif confidence > 0.6:
            return "Moderate probability of success with some uncertainty"
        else:
            return "Exploratory variant with uncertain outcome"