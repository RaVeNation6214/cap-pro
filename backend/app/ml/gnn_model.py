"""
Hybrid GNN Model: GraphCodeBERT + Graph Attention Network.

Architecture:
  1. Node embeddings from GraphCodeBERT (768-dim CLS per function)
  2. GAT layers to propagate vulnerability signals across the call graph
  3. Global mean pooling to get contract-level representation
  4. Concatenate with static features (12-dim)
  5. MLP classifier for multi-label prediction

Fallback mode (no torch_geometric):
  Uses a simple MLP over the mean of node features.
"""
import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Try to import PyTorch Geometric
try:
    from torch_geometric.nn import GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    logger.warning(
        "torch_geometric not installed. Using MLP fallback instead of GNN. "
        "Install with: pip install torch-geometric"
    )


NUM_CLASSES = 5  # Reentrancy, Arithmetic, Access Control, Unchecked Calls, Timestamp
STATIC_FEATURE_DIM = 12  # From CFGBuilder.extract_node_features()
GRAPHCODEBERT_DIM = 768  # From CodeEmbedder


class NodeProjection(nn.Module):
    """Projects GraphCodeBERT 768-dim embeddings to GAT input dim."""

    def __init__(self, in_dim: int = GRAPHCODEBERT_DIM, out_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class HybridGNN(nn.Module):
    """
    Hybrid Graph Attention Network for vulnerability detection.

    Accepts:
        - graph: torch_geometric.data.Data (or Batch) with node features x and edge_index
        - static_features: [B, STATIC_FEATURE_DIM] contract-level static features

    Returns:
        - logits: [B, NUM_CLASSES] raw logits (apply sigmoid for probabilities)
    """

    def __init__(
        self,
        node_in_dim: int = GRAPHCODEBERT_DIM,
        static_feature_dim: int = STATIC_FEATURE_DIM,
        hidden_dim: int = 256,
        gat_heads: int = 4,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.uses_gnn = TORCH_GEOMETRIC_AVAILABLE

        # Node feature projection
        self.node_proj = NodeProjection(node_in_dim, hidden_dim)

        if self.uses_gnn:
            # GAT layer 1: hidden_dim → hidden_dim * heads
            self.gat1 = GATConv(hidden_dim, hidden_dim, heads=gat_heads, dropout=dropout)
            # GAT layer 2: hidden_dim * heads → hidden_dim
            self.gat2 = GATConv(hidden_dim * gat_heads, hidden_dim, heads=1, dropout=dropout)
            gnn_out_dim = hidden_dim
        else:
            gnn_out_dim = hidden_dim

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(gnn_out_dim + static_feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor],
        static_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [N_total, node_in_dim] - all node features in batch
            edge_index: [2, E] - edge indices
            batch: [N_total] - maps each node to its graph index
            static_features: [B, static_feature_dim] - contract-level features

        Returns:
            logits: [B, num_classes]
        """
        # Project node features
        x = self.node_proj(node_features)  # [N, hidden_dim]

        if self.uses_gnn and edge_index.shape[1] > 0:
            # GAT forward
            x = F.elu(self.gat1(x, edge_index))   # [N, hidden * heads]
            x = self.dropout(x)
            x = F.elu(self.gat2(x, edge_index))   # [N, hidden]
            x = self.dropout(x)

            # Global pooling: graph → contract representation
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            x = global_mean_pool(x, batch)  # [B, hidden]
        else:
            # Fallback: mean pooling without GNN
            if batch is None:
                x = x.mean(dim=0, keepdim=True)  # [1, hidden]
            else:
                # Manual scatter mean
                B = batch.max().item() + 1
                out = torch.zeros(B, x.size(1), device=x.device)
                count = torch.zeros(B, 1, device=x.device)
                out.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
                count.scatter_add_(0, batch.unsqueeze(1), torch.ones_like(batch.unsqueeze(1).float()))
                x = out / count.clamp(min=1)

        # Concatenate with static features
        x = torch.cat([x, static_features], dim=1)  # [B, hidden + static]

        # Classify
        return self.classifier(x)

    def predict_proba(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor],
        static_features: torch.Tensor,
    ) -> torch.Tensor:
        """Returns sigmoid probabilities."""
        with torch.no_grad():
            logits = self.forward(node_features, edge_index, batch, static_features)
            return torch.sigmoid(logits)


class SimpleMLP(nn.Module):
    """
    Lightweight fallback model (no graph structure).
    Uses only static + keyword features.
    Used when GraphCodeBERT and GNN are not available.
    """

    def __init__(self, input_dim: int = STATIC_FEATURE_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


def build_model(
    use_gnn: bool = True,
    static_feature_dim: int = STATIC_FEATURE_DIM,
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """Factory function to build the appropriate model."""
    if use_gnn and TORCH_GEOMETRIC_AVAILABLE:
        return HybridGNN(
            static_feature_dim=static_feature_dim,
            num_classes=num_classes,
        )
    else:
        return SimpleMLP(input_dim=static_feature_dim, num_classes=num_classes)
