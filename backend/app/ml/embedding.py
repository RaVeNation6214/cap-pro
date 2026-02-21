"""
GraphCodeBERT Embedding Module.
Uses microsoft/graphcodebert-base to produce node embeddings for smart contract code.
Falls back gracefully if transformers is not available.
"""
import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Try to import transformers - allow graceful failure
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed. GraphCodeBERT unavailable. Using fallback embeddings.")


GRAPHCODEBERT_MODEL = "microsoft/graphcodebert-base"
EMBEDDING_DIM = 768  # GraphCodeBERT CLS output dimension


class CodeEmbedder:
    """
    Produces GraphCodeBERT embeddings for code snippets.
    Each snippet (function body or code window) is encoded to a 768-dim vector
    using the [CLS] token output.

    Falls back to a simple TF-IDF-style bag-of-keywords embedding if
    transformers is not available.
    """

    # Solidity security keywords for fallback embedding
    SECURITY_KEYWORDS = [
        'call', 'send', 'transfer', 'delegatecall', 'staticcall',
        'selfdestruct', 'suicide', 'tx.origin', 'block.timestamp', 'now',
        'require', 'assert', 'revert', 'throw', 'SafeMath', 'overflow',
        'underflow', 'reentrancy', 'onlyOwner', 'modifier', 'payable',
        'external', 'internal', 'public', 'private', 'view', 'pure',
        'mapping', 'address', 'uint256', 'balances', 'withdraw', 'deposit',
        'emit', 'event', 'indexed', 'lock', 'mutex', 'nonReentrant',
    ]

    def __init__(self, device: Optional[str] = None):
        self.model = None
        self.tokenizer = None
        self._model_loaded = False

        if TRANSFORMERS_AVAILABLE:
            if device is None:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.device = device
            self._try_load_model()
        else:
            self.device = 'cpu'

    def _try_load_model(self):
        """Try to load GraphCodeBERT. Fail gracefully if not available."""
        try:
            import torch
            logger.info(f"Loading {GRAPHCODEBERT_MODEL}...")
            self.tokenizer = AutoTokenizer.from_pretrained(GRAPHCODEBERT_MODEL)
            self.model = AutoModel.from_pretrained(GRAPHCODEBERT_MODEL)
            self.model = self.model.to(self.device)
            self.model.eval()
            self._model_loaded = True
            logger.info("GraphCodeBERT loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load GraphCodeBERT: {e}. Using fallback embeddings.")
            self._model_loaded = False

    def embed(self, code_snippet: str) -> List[float]:
        """
        Embed a code snippet to a 768-dim vector.

        Args:
            code_snippet: Solidity code string

        Returns:
            List of 768 floats (CLS embedding from GraphCodeBERT, or fallback)
        """
        if self._model_loaded:
            return self._embed_with_model(code_snippet)
        else:
            return self._embed_fallback(code_snippet)

    def _embed_with_model(self, code_snippet: str) -> List[float]:
        """Use GraphCodeBERT to produce embedding."""
        import torch
        inputs = self.tokenizer(
            code_snippet,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1, 768]

        return cls_embedding.squeeze(0).cpu().tolist()

    def _embed_fallback(self, code_snippet: str) -> List[float]:
        """
        Fallback embedding: keyword frequency vector padded to 768 dims.
        Used when GraphCodeBERT is not available.
        """
        import math
        code_lower = code_snippet.lower()
        vec = []

        # Keyword presence/count features (len(SECURITY_KEYWORDS) = ~32 features)
        for kw in self.SECURITY_KEYWORDS:
            count = code_lower.count(kw.lower())
            vec.append(min(count / 5.0, 1.0))

        # Line-level features
        lines = code_snippet.split('\n')
        n = len(lines)
        vec.extend([
            min(n / 100.0, 1.0),
            min(len(code_snippet) / 5000.0, 1.0),
        ])

        # Pad to 768 with sinusoidal pattern
        base_len = len(vec)
        for i in range(768 - base_len):
            val = math.sin(i * 0.1) * vec[i % base_len] if vec else 0.0
            vec.append(val * 0.1)

        return vec[:768]

    def embed_batch(self, snippets: List[str], batch_size: int = 8) -> List[List[float]]:
        """Embed multiple code snippets."""
        results = []
        for i in range(0, len(snippets), batch_size):
            batch = snippets[i:i + batch_size]
            for snippet in batch:
                results.append(self.embed(snippet))
        return results

    def is_loaded(self) -> bool:
        return self._model_loaded

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_DIM
