from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass


@dataclass
class AttentionConfig:
    """Configuration for the multi-head attention mechanism.
    
    Attributes:
        d_model: Dimension of the model
        nhead: Number of attention heads
        d_k: Dimension of each head
    """
    d_model: int
    nhead: int
    d_k: int


@dataclass
class ModelConfig:
    """Configuration for the advanced model.
    
    Attributes:
        model: The underlying model (e.g., RandomForestClassifier)
        scaler: The scaler for data preprocessing
    """
    model: Any
    scaler: Any


class TimeSeriesTransformer(nn.Module):
    """Transformer-based model for time series prediction."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_heads: int):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim
            ),
            num_layers=num_layers,
        )
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(1, 0, 2)  # (seq_len, batch, features)
        x = self.encoder(x)
        x = x.permute(1, 0, 2)  # (batch, seq_len, features)
        x = self.fc(x)
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.config = AttentionConfig(
            d_model=d_model,
            nhead=nhead,
            d_k=d_model // nhead
        )

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = q.size(0)

        # Linear projections
        q = self.w_q(q).view(batch_size, -1, self.config.nhead, self.config.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.config.nhead, self.config.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.config.nhead, self.config.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.config.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Output
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.config.d_model)
        output = self.w_o(output)

        return output


class EnsembleModel(nn.Module):
    """Ensemble of different models for robust predictions."""

    def __init__(
        self,
        *,
        input_dim: int,
        transformer_config: Dict[str, Any],
        lstm_hidden_dim: int = 64,
        lstm_num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Transformer model
        self.transformer = TimeSeriesTransformer(input_dim=input_dim, **transformer_config)

        # LSTM model
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
        )
        self.lstm_fc = nn.Linear(lstm_hidden_dim, 1)

        # Attention weights
        self.attention = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Transformer prediction
        transformer_out = self.transformer(x)

        # LSTM prediction
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.lstm_fc(lstm_out)

        # Combine predictions with attention
        combined = torch.cat([transformer_out, lstm_out], dim=1)
        weights = F.softmax(self.attention(combined), dim=1)

        # Weighted ensemble prediction
        final_pred = weights[:, 0:1] * transformer_out + weights[:, 1:2] * lstm_out

        return final_pred, transformer_out, lstm_out


class AdvancedModel:
    """Advanced model for stock prediction."""

    def __init__(self) -> None:
        self.config = ModelConfig(
            model=RandomForestClassifier(n_estimators=100),
            scaler=StandardScaler()
        )
