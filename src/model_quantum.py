"""
model_quantum.py
================
Proposed Research Architecture:
Multilingual XLM-RoBERTa encoder coupled with a Quantum-Inspired Sequence Labelling (QISL)
interaction layer and Born-rule POVM projection measurement head.

Note: Implemented using classical tensor computation in PyTorch. Does not require quantum hardware.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaModel, XLMRobertaConfig
from typing import Optional, Dict, Any, Tuple


class QuantumInspiredInteractionLayer(nn.Module):
    """
    Quantum-Inspired Sequence Labelling (QISL) Module:
    1. Transforms real transformer hidden states into complex Hilbert states |ψ⟩ = r * e^(iθ).
    2. Models contextual superposition and phase interference across neighboring tokens.
    3. Implements Born's Rule POVM projection operators for class measurement.
    """

    def __init__(self, hidden_size: int = 768, hilbert_dim: int = 64, num_labels: int = 6, temperature: float = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.hilbert_dim = hilbert_dim
        self.num_labels = num_labels
        self.temperature = temperature

        # 1. Amplitude Transformation: Maps real vector to normalized probability amplitudes r
        self.amplitude_proj = nn.Linear(hidden_size, hilbert_dim)

        # 2. Phase Transformation: Maps contextual features to linguistic phase angles θ in [-π, π]
        self.phase_proj = nn.Linear(hidden_size, hilbert_dim)

        # 3. Trainable Quantum POVM Measurement Bases: |m_c⟩ = u_c + i*v_c for each language class c
        self.measurement_real = nn.Parameter(torch.randn(num_labels, hilbert_dim) / math.sqrt(hilbert_dim))
        self.measurement_imag = nn.Parameter(torch.randn(num_labels, hilbert_dim) / math.sqrt(hilbert_dim))

        # Learnable interaction coupling and interference scale
        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.scale = nn.Parameter(torch.tensor(8.0))

        # Layer normalization to stabilize amplitude magnitudes
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
        Returns:
            logits: (batch_size, seq_len, num_labels)
        """
        B, L, _ = hidden_states.shape
        h = self.norm(hidden_states)

        # Step 1: Calculate Probability Amplitudes r_i
        raw_amp = F.relu(self.amplitude_proj(h)) + 1e-6
        r = raw_amp / torch.norm(raw_amp, p=2, dim=-1, keepdim=True)  # (B, L, hilbert_dim)

        # Step 2: Calculate Linguistic Phase Angles θ_i in [-π, π]
        theta = torch.tanh(self.phase_proj(h)) * math.pi  # (B, L, hilbert_dim)

        # Real and imaginary components of state |ψ_i⟩ = r_i * (cos θ_i + i * sin θ_i)
        psi_real = r * torch.cos(theta)  # (B, L, hilbert_dim)
        psi_imag = r * torch.sin(theta)  # (B, L, hilbert_dim)

        # Step 3: Contextual Quantum Superposition & Interference
        interf_real = torch.matmul(psi_real, psi_real.transpose(-1, -2))  # (B, L, L)
        interf_imag = torch.matmul(psi_imag, psi_imag.transpose(-1, -2))  # (B, L, L)
        interference_matrix = (interf_real + interf_imag) * self.scale

        if attention_mask is not None:
            # Mask out padding tokens from interference calculation
            mask_expanded = attention_mask.unsqueeze(1).expand(-1, L, -1)  # (B, L, L)
            interference_matrix = interference_matrix.masked_fill(mask_expanded == 0, -1e9)

        interf_weights = F.softmax(interference_matrix, dim=-1)  # (B, L, L)

        # Contextual Superposition with Residual Identity preservation
        context_real = torch.matmul(interf_weights, psi_real)  # (B, L, hilbert_dim)
        context_imag = torch.matmul(interf_weights, psi_imag)  # (B, L, hilbert_dim)

        evolved_real = psi_real + self.gamma * context_real
        evolved_imag = psi_imag + self.gamma * context_imag

        # Normalize evolved state to preserve unit trace / norm condition
        norm_factor = torch.sqrt(
            torch.sum(evolved_real ** 2 + evolved_imag ** 2, dim=-1, keepdim=True) + 1e-8
        )
        evolved_real = evolved_real / norm_factor
        evolved_imag = evolved_imag / norm_factor

        # Step 4: Born's Rule POVM Measurement
        # Normalize measurement bases |m_c⟩
        meas_norm = torch.sqrt(
            torch.sum(self.measurement_real ** 2 + self.measurement_imag ** 2, dim=-1, keepdim=True) + 1e-8
        )
        u_norm = self.measurement_real / meas_norm  # (num_labels, hilbert_dim)
        v_norm = self.measurement_imag / meas_norm  # (num_labels, hilbert_dim)

        # Compute complex overlap: ⟨m_c | Ψ_i⟩ = (u - iv) * (evolved_real + i * evolved_imag)
        # Real part R = u * evolved_real + v * evolved_imag
        # Imag part I = u * evolved_imag - v * evolved_real
        R = torch.matmul(evolved_real, u_norm.transpose(0, 1)) + torch.matmul(evolved_imag, v_norm.transpose(0, 1))
        I = torch.matmul(evolved_imag, u_norm.transpose(0, 1)) - torch.matmul(evolved_real, v_norm.transpose(0, 1))

        # Born rule measurement probability: P(y_i = c) = |⟨m_c | Ψ_i⟩|^2 = R^2 + I^2
        probabilities = (R ** 2 + I ** 2)  # (B, L, num_labels)

        # Temperature-scaled logits for stable training
        logits = torch.log(probabilities + 1e-8) / self.temperature
        return logits


class XLMRobertaQuantumInspiredForLID(nn.Module):
    """
    Proposed Model:
    Multilingual XLM-RoBERTa Encoder + Quantum-Inspired Representation & Measurement Layer (QISL).
    """

    def __init__(
        self,
        pretrained_model_name: str = "xlm-roberta-base",
        num_labels: int = 6,
        hilbert_dim: int = 64,
        temperature: float = 0.5,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.config = XLMRobertaConfig.from_pretrained(pretrained_model_name)
        self.config.num_labels = num_labels

        self.roberta = XLMRobertaModel.from_pretrained(pretrained_model_name, config=self.config)
        self.dropout = nn.Dropout(dropout_rate)

        # Quantum-Inspired Superposition and POVM Measurement Head
        self.quantum_head = QuantumInspiredInteractionLayer(
            hidden_size=self.config.hidden_size,
            hilbert_dim=hilbert_dim,
            num_labels=num_labels,
            temperature=temperature,
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with quantum measurement calculation.
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (B, L, D)

        sequence_output = self.dropout(sequence_output)
        logits = self.quantum_head(sequence_output, attention_mask=attention_mask)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}
