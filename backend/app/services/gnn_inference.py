"""
GNN Inference Service.

Loads the trained HybridGNN (or SimpleMLP fallback) model and runs
real inference on Solidity contracts, replacing demo mode.

Used when DEMO_MODE=False in .env settings.
"""
import os
import re
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional

import torch

logger = logging.getLogger(__name__)

# Vulnerability classes (must match training order)
CLASS_NAMES = ["reentrancy", "arithmetic", "access_control", "unchecked_calls", "timestamp"]
CLASS_DISPLAY = ["Reentrancy", "Arithmetic", "Access Control", "Unchecked Calls", "Timestamp"]

# Detection thresholds per class
THRESHOLDS = {
    "reentrancy": 0.45,
    "arithmetic": 0.40,
    "access_control": 0.45,
    "unchecked_calls": 0.40,
    "timestamp": 0.40,
}


class GNNInferenceService:
    """
    Runs trained GNN / MLP model for vulnerability detection.
    Falls back to demo mode if model not found.
    """

    def __init__(self, model_path: str):
        self.model = None
        self.use_gnn = False
        self.cfg_builder = None
        self.embedder = None
        self._loaded = False

        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load model checkpoint."""
        path = Path(model_path)
        if not path.exists():
            logger.warning(f"Model not found at {path}. Using demo mode.")
            return

        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            config = checkpoint.get("config", {})
            self.use_gnn = config.get("use_gnn", False)

            # Import model classes
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))

            if self.use_gnn:
                from app.ml.gnn_model import HybridGNN, TORCH_GEOMETRIC_AVAILABLE
                if not TORCH_GEOMETRIC_AVAILABLE:
                    logger.warning("Model was trained with GNN but torch_geometric unavailable. Using MLP.")
                    self.use_gnn = False
                else:
                    self.model = HybridGNN(
                        node_in_dim=768,
                        static_feature_dim=12,
                        hidden_dim=256,
                        gat_heads=4,
                        num_classes=len(CLASS_NAMES),
                        dropout=0.0,  # no dropout at inference
                    )

            if not self.use_gnn:
                from app.ml.gnn_model import SimpleMLP
                self.model = SimpleMLP(input_dim=12, num_classes=len(CLASS_NAMES))

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            # Initialize CFG builder (always needed)
            from app.ml.graph_builder.cfg_builder import CFGBuilder
            self.cfg_builder = CFGBuilder()

            # Initialize embedder only for GNN
            if self.use_gnn:
                from app.ml.embedding import CodeEmbedder
                self.embedder = CodeEmbedder(device='cpu')

            self._loaded = True
            mode = "HybridGNN (GraphCodeBERT+GAT)" if self.use_gnn else "SimpleMLP"
            logger.info(f"Trained model loaded: {mode} | F1={checkpoint.get('best_f1', 0):.3f}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, code: str) -> Dict:
        """
        Run model inference on Solidity code.

        Returns dict with probabilities for each vulnerability class.
        """
        if not self._loaded or self.model is None:
            return self._fallback_predict(code)

        try:
            graph = self.cfg_builder.build(code)

            if self.use_gnn and self.embedder is not None:
                return self._predict_gnn(code, graph)
            else:
                return self._predict_mlp(graph)

        except Exception as e:
            logger.warning(f"Model inference failed: {e}. Falling back to pattern analysis.")
            return self._fallback_predict(code)

    def _predict_gnn(self, code: str, graph) -> Dict:
        """GNN inference with GraphCodeBERT embeddings."""
        from torch_geometric.data import Data, Batch

        node_embeddings = []
        for fn_name in graph.nodes:
            # Extract function body for embedding
            fn_body = self._extract_fn_body(code, fn_name)
            emb = self.embedder.embed(fn_body[:1000])
            node_embeddings.append(emb)

        if not node_embeddings:
            return self._predict_mlp(graph)

        x = torch.tensor(node_embeddings, dtype=torch.float)
        if graph.edges:
            edge_index = torch.tensor(graph.edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Mean of static features
        n = len(graph.node_features)
        static = torch.tensor(
            [[sum(f[i] for f in graph.node_features) / n for i in range(12)]],
            dtype=torch.float
        )

        data = Data(x=x, edge_index=edge_index)
        batch_obj = Batch.from_data_list([data])

        with torch.no_grad():
            logits = self.model(batch_obj.x, batch_obj.edge_index, batch_obj.batch, static)
            probs = torch.sigmoid(logits).squeeze(0).tolist()

        return {name: float(prob) for name, prob in zip(CLASS_NAMES, probs)}

    def _predict_mlp(self, graph) -> Dict:
        """MLP inference using only static features."""
        if not graph.node_features:
            return {name: 0.1 for name in CLASS_NAMES}

        n = len(graph.node_features)
        static = torch.tensor(
            [[sum(f[i] for f in graph.node_features) / n for i in range(12)]],
            dtype=torch.float
        )

        with torch.no_grad():
            logits = self.model(static)
            probs = torch.sigmoid(logits).squeeze(0).tolist()

        return {name: float(prob) for name, prob in zip(CLASS_NAMES, probs)}

    def _extract_fn_body(self, code: str, fn_name: str) -> str:
        """Extract function body for embedding."""
        idx = code.find(fn_name + '(')
        if idx == -1:
            return code[:500]
        # Find opening brace
        brace_idx = code.find('{', idx)
        if brace_idx == -1:
            return code[idx:idx + 500]
        # Extract ~500 chars of body
        return code[idx:brace_idx + 500]

    def _fallback_predict(self, code: str) -> Dict:
        """Pure regex fallback when model inference fails."""
        scores = {name: 0.05 for name in CLASS_NAMES}

        # Reentrancy
        if re.search(r'\.call\s*\{?\s*value\s*:', code):
            scores["reentrancy"] = max(scores["reentrancy"], 0.85)
        if re.search(r'msg\.sender\.call', code):
            scores["reentrancy"] = max(scores["reentrancy"], 0.75)
        if 'ReentrancyGuard' in code or 'nonReentrant' in code:
            scores["reentrancy"] *= 0.2

        # Access control
        if re.search(r'tx\.origin', code):
            scores["access_control"] = max(scores["access_control"], 0.90)

        # Unchecked calls
        if re.search(r'\.send\s*\(', code) and 'require' not in code:
            scores["unchecked_calls"] = max(scores["unchecked_calls"], 0.75)
        if re.search(r'\.delegatecall\s*\(', code):
            scores["unchecked_calls"] = max(scores["unchecked_calls"], 0.80)
        if re.search(r'\.call\s*[\(\{]', code):
            scores["unchecked_calls"] = max(scores["unchecked_calls"], 0.65)

        # Arithmetic
        if re.search(r'pragma solidity \^0\.[0-7]', code):
            if not re.search(r'SafeMath', code):
                scores["arithmetic"] = max(scores["arithmetic"], 0.65)

        return scores
