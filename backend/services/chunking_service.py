from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
try:
    from chonkie import TokenChunker, SentenceChunker, SemanticChunker, SDPMChunker
    CHONKIE_AVAILABLE = True
except ImportError:
    CHONKIE_AVAILABLE = False
    print("⚠️ Chonkie not installed. Install with: pip install chonkie")
from backend.config.settings import get_settings
settings = get_settings()
@dataclass
class ChunkResult:
    chunks: List[str]
    metadata: Dict[str, Any]
    original_text: str
    strategy: str
class BiologicalChunkingService:
    def __init__(self):
        self.settings = settings
        self._validate_chonkie()
    def _validate_chonkie(self):
        if not CHONKIE_AVAILABLE:
            raise ImportError(
                "Chonkie is required for chunking. Install with: pip install chonkie"
            )
    def chunk_experiment_text(
        self,
        text: str,
        strategy: str = "semantic",
        max_chunk_size: int = 512,
        overlap: int = 50
    ) -> ChunkResult:
        if not text or len(text.strip()) == 0:
            return ChunkResult(
                chunks=[],
                metadata={"strategy": strategy, "chunk_count": 0},
                original_text=text,
                strategy=strategy
            )
        if strategy == "protocol":
            return self._chunk_protocol(text, max_chunk_size, overlap)
        elif strategy == "semantic":
            return self._chunk_semantic(text, max_chunk_size, overlap)
        elif strategy == "sentence":
            return self._chunk_sentence(text, max_chunk_size, overlap)
        elif strategy == "token":
            return self._chunk_token(text, max_chunk_size, overlap)
        elif strategy == "sdpm":
            return self._chunk_sdpm(text, max_chunk_size, overlap)
        else:
            return self._chunk_semantic(text, max_chunk_size, overlap)
    def _chunk_protocol(
        self,
        text: str,
        max_chunk_size: int,
        overlap: int
    ) -> ChunkResult:
        step_patterns = [
            r'\n\d+\.\s+',
            r'\nStep\s+\d+',
            r'\n[A-Z][a-z]+:\s+',
            r'\n
        ]
        splits = []
        for pattern in step_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                splits.append(match.start())
        splits = sorted(set(splits))
        if len(splits) == 0:
            return self._chunk_semantic(text, max_chunk_size, overlap)
        chunks = []
        splits.append(len(text))
        for i in range(len(splits) - 1):
            start = splits[i]
            end = splits[i + 1]
            chunk_text = text[start:end].strip()
            if chunk_text:
                if len(chunk_text) > max_chunk_size * 4:
                    sub_result = self._chunk_semantic(chunk_text, max_chunk_size, overlap)
                    chunks.extend(sub_result.chunks)
                else:
                    chunks.append(chunk_text)
        return ChunkResult(
            chunks=chunks,
            metadata={
                "strategy": "protocol",
                "chunk_count": len(chunks),
                "splits_detected": len(splits) - 1,
                "avg_chunk_length": sum(len(c) for c in chunks) / len(chunks) if chunks else 0
            },
            original_text=text,
            strategy="protocol"
        )
    def _chunk_semantic(
        self,
        text: str,
        max_chunk_size: int,
        overlap: int
    ) -> ChunkResult:
        try:
            chunker = SemanticChunker(
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                chunk_size=max_chunk_size,
                similarity_threshold=0.5
            )
            chunks = chunker.chunk(text)
            chunk_texts = [chunk.text for chunk in chunks]
            return ChunkResult(
                chunks=chunk_texts,
                metadata={
                    "strategy": "semantic",
                    "chunk_count": len(chunk_texts),
                    "avg_chunk_length": sum(len(c) for c in chunk_texts) / len(chunk_texts) if chunk_texts else 0
                },
                original_text=text,
                strategy="semantic"
            )
        except Exception as e:
            print(f"⚠️ Semantic chunking failed: {e}. Falling back to sentence chunking.")
            return self._chunk_sentence(text, max_chunk_size, overlap)
    def _chunk_sentence(
        self,
        text: str,
        max_chunk_size: int,
        overlap: int
    ) -> ChunkResult:
        try:
            chunker = SentenceChunker(
                tokenizer="gpt2",
                chunk_size=max_chunk_size,
                chunk_overlap=overlap
            )
            chunks = chunker.chunk(text)
            chunk_texts = [chunk.text for chunk in chunks]
            return ChunkResult(
                chunks=chunk_texts,
                metadata={
                    "strategy": "sentence",
                    "chunk_count": len(chunk_texts),
                    "avg_chunk_length": sum(len(c) for c in chunk_texts) / len(chunk_texts) if chunk_texts else 0
                },
                original_text=text,
                strategy="sentence"
            )
        except Exception as e:
            print(f"⚠️ Sentence chunking failed: {e}. Falling back to token chunking.")
            return self._chunk_token(text, max_chunk_size, overlap)
    def _chunk_token(
        self,
        text: str,
        max_chunk_size: int,
        overlap: int
    ) -> ChunkResult:
        try:
            chunker = TokenChunker(
                tokenizer="gpt2",
                chunk_size=max_chunk_size,
                chunk_overlap=overlap
            )
            chunks = chunker.chunk(text)
            chunk_texts = [chunk.text for chunk in chunks]
            return ChunkResult(
                chunks=chunk_texts,
                metadata={
                    "strategy": "token",
                    "chunk_count": len(chunk_texts),
                    "max_chunk_size": max_chunk_size,
                    "overlap": overlap
                },
                original_text=text,
                strategy="token"
            )
        except Exception as e:
            print(f"⚠️ Token chunking failed: {e}. Using manual fallback.")
            return self._chunk_manual(text, max_chunk_size * 4)
    def _chunk_sdpm(
        self,
        text: str,
        max_chunk_size: int,
        overlap: int
    ) -> ChunkResult:
        try:
            chunker = SDPMChunker(
                tokenizer="gpt2",
                chunk_size=max_chunk_size,
                chunk_overlap=overlap
            )
            chunks = chunker.chunk(text)
            chunk_texts = [chunk.text for chunk in chunks]
            return ChunkResult(
                chunks=chunk_texts,
                metadata={
                    "strategy": "sdpm",
                    "chunk_count": len(chunk_texts),
                    "avg_chunk_length": sum(len(c) for c in chunk_texts) / len(chunk_texts) if chunk_texts else 0
                },
                original_text=text,
                strategy="sdpm"
            )
        except Exception as e:
            print(f"⚠️ SDPM chunking failed: {e}. Falling back to sentence chunking.")
            return self._chunk_sentence(text, max_chunk_size, overlap)
    def _chunk_manual(
        self,
        text: str,
        max_chars: int
    ) -> ChunkResult:
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        for word in words:
            word_length = len(word) + 1
            if current_length + word_length > max_chars and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return ChunkResult(
            chunks=chunks,
            metadata={
                "strategy": "manual_fallback",
                "chunk_count": len(chunks),
                "max_chars": max_chars
            },
            original_text=text,
            strategy="manual_fallback"
        )
    def auto_select_strategy(self, text: str) -> str:
        text_lower = text.lower()
        protocol_indicators = [
            r'\bstep\s+\d+',
            r'\d+\.\s+',
            r'\bmaterial',
            r'\bmethod',
            r'\bprocedure',
            r'\bprotocol',
            r'\bincubate',
            r'\bcentrifuge',
            r'\bmix\b',
            r'\badd\b.*\bml\b'
        ]
        protocol_score = sum(
            len(re.findall(pattern, text_lower, re.IGNORECASE))
            for pattern in protocol_indicators
        )
        if protocol_score > 5:
            return "protocol"
        if re.search(r'\n
            return "sdpm"
        if len(text) < 1000:
            return "sentence"
        return "semantic"
    def chunk_with_metadata(
        self,
        text: str,
        metadata: Dict[str, Any],
        strategy: Optional[str] = None,
        max_chunk_size: int = 512,
        overlap: int = 50
    ) -> List[Dict[str, Any]]:
        if strategy is None:
            strategy = self.auto_select_strategy(text)
        result = self.chunk_experiment_text(text, strategy, max_chunk_size, overlap)
        chunks_with_metadata = []
        for i, chunk_text in enumerate(result.chunks):
            chunk_meta = {
                "text": chunk_text,
                "chunk_index": i,
                "total_chunks": len(result.chunks),
                "strategy": result.strategy,
                "original_length": len(text),
                "chunk_length": len(chunk_text),
                **metadata
            }
            chunks_with_metadata.append(chunk_meta)
        return chunks_with_metadata
_chunking_service = None
def get_chunking_service() -> BiologicalChunkingService:
    global _chunking_service
    if _chunking_service is None:
        _chunking_service = BiologicalChunkingService()
    return _chunking_service