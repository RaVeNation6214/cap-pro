import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns: (output, attention_weights)
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.w_o(context)

        # Average attention weights across heads for interpretability
        avg_attn_weights = attn_weights.mean(dim=1)

        return output, avg_attn_weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual connection
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attn_weights


class WindowEncoder(nn.Module):
    """
    Transformer encoder for individual code windows.
    Encodes a sequence of tokens into a fixed-size embedding.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Token IDs [batch, seq_len]
            mask: Attention mask [batch, seq_len]
        Returns:
            window_embedding: [batch, d_model]
            attention_weights: [batch, seq_len] (averaged across layers and heads)
        """
        # Embedding
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)

        # Create padding mask
        if mask is None:
            mask = (x.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)

        # Transformer layers
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            all_attn_weights.append(attn_weights)

        # Average attention weights across layers
        avg_attn = torch.stack(all_attn_weights).mean(dim=0)  # [batch, seq_len, seq_len]
        avg_attn = avg_attn.mean(dim=1)  # [batch, seq_len] - average over query positions

        # Global average pooling to get window embedding
        window_embedding = x.mean(dim=1)  # [batch, d_model]

        return window_embedding, avg_attn


class ContractAttentionPooling(nn.Module):
    """
    Attention-based pooling over window embeddings.
    Uses a learnable CLS token to attend over all windows.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Cross-attention: CLS attends to windows
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        window_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            window_embeddings: [batch, num_windows, d_model]
            mask: [batch, num_windows]
        Returns:
            contract_embedding: [batch, d_model]
            attention_weights: [batch, num_windows]
        """
        batch_size = window_embeddings.size(0)

        # Expand CLS token for batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Cross-attention: CLS queries window embeddings
        attn_output, attn_weights = self.cross_attention(
            query=cls_tokens,
            key=window_embeddings,
            value=window_embeddings,
            mask=mask
        )

        # Get contract embedding
        contract_embedding = self.norm(cls_tokens + self.dropout(attn_output))
        contract_embedding = contract_embedding.squeeze(1)  # [batch, d_model]

        # Get attention weights for explainability
        window_attn_weights = attn_weights.squeeze(1)  # [batch, num_windows]

        return contract_embedding, window_attn_weights


class MultiLabelClassifier(nn.Module):
    """
    Multi-label classification head for vulnerability detection.
    Outputs independent probabilities for each vulnerability class.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Contract embedding [batch, input_dim]
        Returns:
            logits: [batch, num_classes] (before sigmoid)
        """
        return self.classifier(x)


class HierarchicalTransformer(nn.Module):
    """
    Hierarchical Transformer for Smart Contract Vulnerability Detection.

    Architecture:
    1. Split contract into windows
    2. Encode each window with WindowEncoder
    3. Fuse with static features
    4. Pool windows with attention
    5. Classify vulnerabilities

    Returns predictions and attention weights for explainability.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        num_classes: int = 4,
        static_feature_dim: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.static_feature_dim = static_feature_dim

        # Window encoder
        self.window_encoder = WindowEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout
        )

        # Static feature projection
        self.static_proj = nn.Linear(static_feature_dim, d_model // 4)

        # Contract-level attention pooling
        self.contract_pooling = ContractAttentionPooling(
            d_model=d_model + d_model // 4,  # Include static features
            n_heads=4,
            dropout=dropout
        )

        # Final projection
        self.pre_classifier = nn.Linear(d_model + d_model // 4, d_model)

        # Classification head
        self.classifier = MultiLabelClassifier(
            input_dim=d_model,
            hidden_dim=128,
            num_classes=num_classes,
            dropout=0.2
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        static_features: torch.Tensor,
        window_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            token_ids: [batch, num_windows, seq_len]
            static_features: [batch, num_windows, static_feature_dim]
            window_mask: [batch, num_windows] - which windows are valid

        Returns:
            logits: [batch, num_classes]
            window_attention: [batch, num_windows] - attention weights for explainability
            token_attention: [batch, num_windows, seq_len] - per-token attention
        """
        batch_size, num_windows, seq_len = token_ids.shape

        # Flatten for window encoding
        flat_tokens = token_ids.view(batch_size * num_windows, seq_len)

        # Encode each window
        window_embeddings, token_attention = self.window_encoder(flat_tokens)

        # Reshape back
        window_embeddings = window_embeddings.view(batch_size, num_windows, -1)
        token_attention = token_attention.view(batch_size, num_windows, seq_len)

        # Project static features
        static_proj = self.static_proj(static_features)

        # Concatenate window embeddings with static features
        combined = torch.cat([window_embeddings, static_proj], dim=-1)

        # Contract-level attention pooling
        contract_embedding, window_attention = self.contract_pooling(combined, window_mask)

        # Project to classifier input dimension
        contract_embedding = self.pre_classifier(contract_embedding)

        # Classification
        logits = self.classifier(contract_embedding)

        return logits, window_attention, token_attention

    def predict(
        self,
        token_ids: torch.Tensor,
        static_features: torch.Tensor,
        window_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with probabilities.

        Returns:
            probabilities: [batch, num_classes]
            predictions: [batch, num_classes] (binary)
            window_attention: [batch, num_windows]
            token_attention: [batch, num_windows, seq_len]
        """
        logits, window_attention, token_attention = self.forward(
            token_ids, static_features, window_mask
        )

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > threshold).float()

        return probabilities, predictions, window_attention, token_attention


def create_model(config) -> HierarchicalTransformer:
    """Create model from config."""
    return HierarchicalTransformer(
        vocab_size=config.VOCAB_SIZE,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        d_ff=config.D_FF,
        max_seq_len=config.MAX_SEQ_LEN,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT
    )
