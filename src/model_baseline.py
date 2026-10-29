"""
model_baseline.py
=================
Baseline Sequence Labelling Model:
Pretrained XLM-RoBERTa encoder coupled with a standard linear token classification head.
"""

import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig
from typing import Optional, Dict, Any, Tuple


class XLMRobertaBaselineForLID(nn.Module):
    """
    Standard Baseline Model:
    Multilingual XLM-RoBERTa Encoder + Linear Classification Head.
    """

    def __init__(
        self,
        pretrained_model_name: str = "xlm-roberta-base",
        num_labels: int = 6,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.config = XLMRobertaConfig.from_pretrained(pretrained_model_name)
        self.config.num_labels = num_labels

        # Pretrained Multilingual Transformer
        self.roberta = XLMRobertaModel.from_pretrained(pretrained_model_name, config=self.config)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Linear Token Classification Head: Hidden Size (768) -> num_labels (3)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass:
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size, seq_len) with -100 for ignored tokens
        Returns:
            Dictionary with 'loss' (if labels provided) and 'logits' (batch_size, seq_len, num_labels)
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (B, L, D)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # (B, L, num_labels)

        loss = None
        if labels is not None:
            # Flatten to compute token-level cross-entropy loss ignoring -100
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}
