"""
model.py
--------
Two deep learning architectures for vital sign deterioration prediction:
  1. BidirectionalLSTM  — fast, strong baseline
  2. TemporalTransformer — higher accuracy, recommended

Both take:
  x_seq    : (batch, 12, temporal_features)  — 12 hours of vitals
  x_static : (batch, static_features)        — age, comorbidity, gender, admission
And return a probability between 0 and 1 (deterioration risk score).
"""

import torch
import torch.nn as nn


# ── 1. Bidirectional LSTM ─────────────────────────────────────────────────────
class BidirectionalLSTM(nn.Module):
    def __init__(self, temporal_input_size, static_input_size,
                 hidden_size=128, num_layers=2, dropout=0.35):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=temporal_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.static_encoder = nn.Sequential(
            nn.Linear(static_input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2 + 32, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x_seq, x_static):
        lstm_out, _ = self.lstm(x_seq)
        last_step   = lstm_out[:, -1, :]
        static_emb  = self.static_encoder(x_static)
        fused       = torch.cat([last_step, static_emb], dim=1)
        return torch.sigmoid(self.classifier(fused)).squeeze(1)


# ── 2. Temporal Transformer ───────────────────────────────────────────────────
class TemporalTransformer(nn.Module):
    def __init__(self, temporal_input_size, static_input_size,
                 d_model=128, nhead=8, num_encoder_layers=3,
                 dim_feedforward=256, dropout=0.2, max_seq_len=72):
        super().__init__()

        self.input_proj    = nn.Linear(temporal_input_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.static_encoder = nn.Sequential(
            nn.Linear(static_input_size, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
        )

        self.attn_pool  = nn.Linear(d_model, 1)

        self.classifier = nn.Sequential(
            nn.Linear(d_model + 32, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x_seq, x_static):
        B, T, _ = x_seq.shape
        x        = self.input_proj(x_seq)
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        x        = x + self.pos_embedding(positions)
        enc      = self.transformer(x)
        weights  = torch.softmax(self.attn_pool(enc), dim=1)
        pooled   = (enc * weights).sum(dim=1)
        static_emb = self.static_encoder(x_static)
        fused    = torch.cat([pooled, static_emb], dim=1)
        return torch.sigmoid(self.classifier(fused)).squeeze(1)
