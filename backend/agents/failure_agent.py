from typing import Dict, Any, List
import numpy as np
class FailureAgent:
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        neighbors = context.get('reranked_neighbors', [])
        if not neighbors:
            return {
                'failures': [],
                'reproducibility_risk': 0.5,
                'risk_level': 'UNKNOWN',
                'failure_patterns': {}
            }
        failures = self._identify_failures(neighbors)
        repro_risk = self._compute_reproducibility_risk(neighbors)
        failure_patterns = self._analyze_failure_patterns(failures)
        risk_level = self._categorize_risk(repro_risk)
        recommendations = self._generate_recommendations(
            failure_patterns,
            risk_level
        )
        return {
            'failures': failures,
            'reproducibility_risk': repro_risk,
            'risk_level': risk_level,
            'failure_patterns': failure_patterns,
            'recommendations': recommendations
        }
    def _identify_failures(self, neighbors: List[Dict]) -> List[Dict]:
        return [
            n for n in neighbors
            if not n['payload'].get('success', True)
        ]
    def _compute_reproducibility_risk(self, neighbors: List[Dict]) -> float:
        if len(neighbors) < 3:
            return 0.5
        outcomes = []
        for n in neighbors:
            success = n['payload'].get('success')
            if success is not None:
                outcomes.append(1.0 if success else 0.0)
        if not outcomes:
            return 0.5
        variance = np.var(outcomes)
        failure_rate = 1.0 - np.mean(outcomes)
        risk = (variance * 0.5 + failure_rate * 0.5)
        return min(1.0, risk)
    def _analyze_failure_patterns(self, failures: List[Dict]) -> Dict[str, Any]:
        if not failures:
            return {
                'common_organisms': {},
                'temperature_range': None,
                'ph_range': None,
                'common_issues': []
            }
        organisms = {}
        temperatures = []
        phs = []
        for failure in failures:
            cond = failure['payload'].get('conditions', {})
            org = cond.get('organism', 'unknown')
            organisms[org] = organisms.get(org, 0) + 1
            if cond.get('temperature') is not None:
                temperatures.append(cond['temperature'])
            if cond.get('ph') is not None:
                phs.append(cond['ph'])
        return {
            'common_organisms': organisms,
            'temperature_range': {
                'min': min(temperatures) if temperatures else None,
                'max': max(temperatures) if temperatures else None,
                'mean': np.mean(temperatures) if temperatures else None
            },
            'ph_range': {
                'min': min(phs) if phs else None,
                'max': max(phs) if phs else None,
                'mean': np.mean(phs) if phs else None
            },
            'failure_count': len(failures),
            'variance': self._calculate_condition_variance(failures)
        }
    def _calculate_condition_variance(self, failures: List[Dict]) -> float:
        temperatures = []
        for failure in failures:
            cond = failure['payload'].get('conditions', {})
            if cond.get('temperature') is not None:
                temperatures.append(cond['temperature'])
        if len(temperatures) < 2:
            return 0.0
        return float(np.var(temperatures))
    def _categorize_risk(self, risk: float) -> str:
        if risk < 0.3:
            return "LOW"
        elif risk < 0.6:
            return "MEDIUM"
        else:
            return "HIGH"
    def _generate_recommendations(
        self,
        failure_patterns: Dict[str, Any],
        risk_level: str
    ) -> List[str]:
        recommendations = []
        if risk_level == "HIGH":
            recommendations.append(
                "⚠️ High reproducibility risk detected. "
                "Consider additional replicates and controls."
            )
        common_orgs = failure_patterns.get('common_organisms', {})
        if common_orgs:
            most_common = max(common_orgs.items(), key=lambda x: x[1])
            recommendations.append(
                f"⚠️ Multiple failures observed with {most_common[0]} "
                f"({most_common[1]} cases). Consider alternative organism."
            )
        temp_range = failure_patterns.get('temperature_range', {})
        if temp_range and temp_range.get('min') is not None:
            recommendations.append(
                f"ℹ️ Failed experiments temperature range: "
                f"{temp_range['min']:.1f}°C - {temp_range['max']:.1f}°C. "
                f"Consider adjusting temperature."
            )
        return recommendations