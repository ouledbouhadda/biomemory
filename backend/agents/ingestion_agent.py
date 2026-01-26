from typing import Dict, Any, List
import re
from bs4 import BeautifulSoup
from datetime import datetime
from backend.services.qdrant_service import get_qdrant_service
class IngestionAgent:
    def __init__(self):
        self.qdrant = get_qdrant_service()
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        user_input = context.get('user_input', {})
        raw_experiment = user_input.get('experiment', user_input.get('query', {}))
        cleaned_data = await self._clean_and_normalize(raw_experiment)
        enriched_metadata = await self._enrich_metadata(cleaned_data)
        validation_result = self._validate_experiment_data(cleaned_data)
        qdrant_payload = self._prepare_qdrant_payload(cleaned_data, enriched_metadata)
        return {
            'cleaned_experiment': cleaned_data,
            'enriched_metadata': enriched_metadata,
            'qdrant_payload': qdrant_payload,
            'validation_result': validation_result,
            'indexing_ready': validation_result['is_valid']
        }
    async def _clean_and_normalize(self, raw_experiment: Dict[str, Any]) -> Dict[str, Any]:
        cleaned_text = self._clean_text_advanced(raw_experiment.get('text', ''))
        normalized_sequence, sequence_metadata = self._normalize_sequence_advanced(
            raw_experiment.get('sequence', '')
        )
        standardized_conditions = self._standardize_conditions_advanced(raw_experiment)
        processed_image = self._process_image_data(raw_experiment.get('image_base64'))
        return {
            'text': cleaned_text,
            'sequence': normalized_sequence,
            'sequence_metadata': sequence_metadata,
            'image_base64': processed_image,
            'conditions': standardized_conditions,
            'success': raw_experiment.get('success'),
            'source': raw_experiment.get('source'),
            'notes': raw_experiment.get('notes'),
            'ingestion_timestamp': datetime.now().isoformat()
        }
    async def _enrich_metadata(self, cleaned_data: Dict[str, Any]) -> Dict[str, Any]:
        metadata = {
            'text_length': len(cleaned_data.get('text', '')),
            'has_sequence': bool(cleaned_data.get('sequence')),
            'has_image': bool(cleaned_data.get('image_base64')),
            'has_conditions': bool(cleaned_data.get('conditions')),
            'modality_count': 0,
            'keywords': [],
            'domain': 'unknown',
            'experiment_type': 'unknown',
            'confidence_score': 0.5,
            'processing_timestamp': datetime.now().isoformat()
        }
        modalities = ['text', 'sequence', 'image_base64', 'conditions']
        metadata['modality_count'] = sum(1 for mod in modalities if cleaned_data.get(mod))
        metadata['keywords'] = self._extract_keywords(cleaned_data.get('text', ''))
        metadata['domain'] = self._classify_domain(cleaned_data)
        metadata['experiment_type'] = self._classify_experiment_type(cleaned_data)
        metadata['confidence_score'] = self._calculate_confidence_score(cleaned_data)
        return metadata
    def _prepare_qdrant_payload(self, cleaned_data: Dict[str, Any],
                               enriched_metadata: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(cleaned_data)
        payload.update(enriched_metadata)
        payload['searchable_text'] = f"{payload.get('text', '')} {payload.get('notes', '')}"
        payload['full_text'] = payload['searchable_text']
        payload['indexed_at'] = datetime.now().isoformat()
        payload['data_quality_score'] = enriched_metadata.get('confidence_score', 0.5)
        payload['completeness_score'] = enriched_metadata.get('modality_count', 0) / 4.0
        return payload
    def _validate_experiment_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        errors = []
        warnings = []
        if not data.get('text') or len(data.get('text', '')) < 10:
            errors.append("Text description too short or missing")
        if data.get('sequence'):
            if not self._is_valid_sequence(data['sequence']):
                errors.append("Invalid biological sequence format")
        conditions = data.get('conditions', {})
        if not conditions.get('organism'):
            warnings.append("Organism not specified")
        if not conditions.get('temperature'):
            warnings.append("Temperature not specified")
        if data.get('image_base64'):
            if not self._is_valid_image(data['image_base64']):
                errors.append("Invalid image format")
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'quality_score': max(0, 1.0 - (len(errors) * 0.2 + len(warnings) * 0.1))
        }
    def _clean_text_advanced(self, text: str) -> str:
        if not text:
            return ""
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        text = re.sub(r'[^\w\s\.,;:()\-°%μλβαγδεζηθικλμνξοπρστυφχψω±×÷≤≥≠≈∑∏∫∂∇∈∉∋∌⊂⊃⊆⊇∪∩∧∨¬∀∃⇒⇔≡≢⊥∥∠∟⊾⊿△▽○●□■◇◆★☆♀♂♠♣♥♦]', '', text)
        abbreviations = {
            'dna': 'DNA',
            'rna': 'RNA',
            'pcr': 'PCR',
            'rt-pcr': 'RT-PCR',
            'qpcr': 'qPCR',
            'western blot': 'Western blot',
            'northern blot': 'Northern blot',
            'southern blot': 'Southern blot',
            'elisa': 'ELISA',
            'ihc': 'IHC',
            'if': 'IF',
            'icc': 'ICC'
        }
        for abbr, full in abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text, flags=re.IGNORECASE)
        return text
    def _normalize_sequence_advanced(self, sequence: str) -> tuple[str, Dict[str, Any]]:
        if not sequence:
            return "", {}
        normalized = sequence.upper().replace(' ', '').replace('\n', '').replace('\r', '')
        metadata = {
            'length': len(normalized),
            'type': self._detect_sequence_type(normalized),
            'gc_content': self._calculate_gc_content(normalized),
            'has_gaps': '-' in normalized or 'N' in normalized,
            'is_valid': self._is_valid_sequence(normalized)
        }
        return normalized, metadata
    def _standardize_conditions_advanced(self, data: Dict) -> Dict:
        conditions = data.get('conditions', {})
        if not isinstance(conditions, dict):
            conditions = {}
        standardized = {
            'organism': conditions.get('organism', 'unknown'),
            'temperature': self._normalize_temperature(conditions.get('temperature', 37.0)),
            'ph': self._normalize_ph(conditions.get('ph', 7.0)),
            'protocol_id': conditions.get('protocol_id'),
            'additional_params': conditions.get('additional_params', {}),
            'validation_warnings': []
        }
        if standardized['temperature'] < 0 or standardized['temperature'] > 100:
            standardized['validation_warnings'].append("Temperature outside normal range")
        if standardized['ph'] < 0 or standardized['ph'] > 14:
            standardized['validation_warnings'].append("pH outside valid range")
        return standardized
    def _process_image_data(self, image_base64: str) -> str:
        if not image_base64:
            return ""
        if not image_base64.startswith('data:image/'):
            return ""
        return image_base64
    def _extract_keywords(self, text: str) -> List[str]:
        if not text:
            return []
        bio_keywords = {
            'dna', 'rna', 'protein', 'gene', 'cell', 'tissue', 'organism',
            'bacteria', 'virus', 'enzyme', 'receptor', 'antibody', 'assay',
            'experiment', 'protocol', 'method', 'analysis', 'result',
            'expression', 'transcription', 'translation', 'replication',
            'mutation', 'sequencing', 'pcr', 'western', 'elisa', 'microscopy'
        }
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if word in bio_keywords]
        return list(set(keywords))[:10]
    def _classify_domain(self, data: Dict[str, Any]) -> str:
        text = data.get('text', '').lower()
        sequence = data.get('sequence', '')
        if 'dna' in text or 'rna' in text or 'gene' in text:
            return 'molecular_biology'
        elif 'protein' in text or 'enzyme' in text:
            return 'biochemistry'
        elif 'cell' in text or 'tissue' in text:
            return 'cell_biology'
        elif 'bacteria' in text or 'microorganism' in text:
            return 'microbiology'
        elif sequence and len(sequence) > 50:
            return 'genomics'
        else:
            return 'general_biology'
    def _classify_experiment_type(self, data: Dict[str, Any]) -> str:
        text = data.get('text', '').lower()
        if 'sequencing' in text or 'pcr' in text:
            return 'molecular_assay'
        elif 'western' in text or 'elisa' in text:
            return 'protein_analysis'
        elif 'microscopy' in text or 'imaging' in text:
            return 'imaging'
        elif 'culture' in text or 'growth' in text:
            return 'cell_culture'
        else:
            return 'general_experiment'
    def _calculate_confidence_score(self, data: Dict[str, Any]) -> float:
        score = 0.0
        if data.get('text') and len(data['text']) > 50:
            score += 0.3
        if data.get('sequence'):
            score += 0.2
        conditions = data.get('conditions', {})
        if conditions.get('organism') and conditions.get('organism') != 'unknown':
            score += 0.15
        if conditions.get('temperature'):
            score += 0.15
        if conditions.get('ph'):
            score += 0.1
        if data.get('image_base64'):
            score += 0.1
        return min(score, 1.0)
    def _detect_sequence_type(self, sequence: str) -> str:
        if not sequence:
            return 'unknown'
        dna_bases = set('ATCGN')
        rna_bases = set('AUCGN')
        protein_chars = set('ACDEFGHIKLMNPQRSTVWY')
        seq_set = set(sequence.upper())
        if seq_set.issubset(dna_bases):
            return 'dna'
        elif seq_set.issubset(rna_bases):
            return 'rna'
        elif seq_set.issubset(protein_chars):
            return 'protein'
        else:
            return 'mixed'
    def _calculate_gc_content(self, sequence: str) -> float:
        if not sequence:
            return 0.0
        gc_count = sequence.upper().count('G') + sequence.upper().count('C')
        total_bases = len(sequence)
        return gc_count / total_bases if total_bases > 0 else 0.0
    def _is_valid_sequence(self, sequence: str) -> bool:
        if not sequence:
            return False
        valid_chars = set('ATCGURYKMSWBDHVN-')
        return all(c.upper() in valid_chars for c in sequence)
    def _is_valid_image(self, image_base64: str) -> bool:
        if not image_base64:
            return False
        return image_base64.startswith('data:image/') and len(image_base64) > 100
    def _normalize_temperature(self, temp: Any) -> float:
        try:
            temp_float = float(temp)
            return temp_float
        except (ValueError, TypeError):
            return 37.0
    def _normalize_ph(self, ph: Any) -> float:
        try:
            ph_float = float(ph)
            return max(0.0, min(14.0, ph_float))
        except (ValueError, TypeError):
            return 7.0