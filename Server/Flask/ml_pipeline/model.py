"""
Model Definition
================
RoBERTa-based classifier for ESG metric identification.
Multi-head architecture with Environmental/Social/Governance heads.
"""

import torch
import torch.nn as nn
from transformers import RobertaModel
from .labeling import METRIC_TO_ID, NUM_CLASSES, CATEGORY_MAP


class ESGMetricClassifier(nn.Module):
    """
    RoBERTa-based classifier for ESG metric identification.
    
    Multi-head architecture:
        RoBERTa encoder → [CLS] pooling → shared dropout →
            → Environmental head → env metrics
            → Social head → social metrics
            → Governance head → governance metrics
            → Combined logits → NUM_CLASSES
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()
        self.bert = RobertaModel.from_pretrained('roberta-base')
        hidden_size = self.bert.config.hidden_size  # 768 for roberta-base
        self.dropout = nn.Dropout(dropout)
        
        # Category-specific metric indices
        env_metrics = CATEGORY_MAP.get('environmental', [])
        soc_metrics = CATEGORY_MAP.get('social', [])
        gov_metrics = CATEGORY_MAP.get('governance', [])
        
        self.env_indices = [METRIC_TO_ID[m] for m in env_metrics if m in METRIC_TO_ID]
        self.soc_indices = [METRIC_TO_ID[m] for m in soc_metrics if m in METRIC_TO_ID]
        self.gov_indices = [METRIC_TO_ID[m] for m in gov_metrics if m in METRIC_TO_ID]
        self.no_metric_idx = METRIC_TO_ID.get('no_metric', num_classes - 1)
        
        # Multi-head classifiers
        self.env_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, len(self.env_indices)),
        )
        
        self.soc_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, len(self.soc_indices)),
        )
        
        self.gov_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, len(self.gov_indices)),
        )
        
        self.no_metric_head = nn.Linear(hidden_size, 1)
        
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        
        env_logits = self.env_head(cls_output)
        soc_logits = self.soc_head(cls_output)
        gov_logits = self.gov_head(cls_output)
        no_metric_logit = self.no_metric_head(cls_output)
        
        batch_size = input_ids.size(0)
        full_logits = torch.zeros(batch_size, self.num_classes, device=input_ids.device)
        
        for i, idx in enumerate(self.env_indices):
            full_logits[:, idx] = env_logits[:, i]
        for i, idx in enumerate(self.soc_indices):
            full_logits[:, idx] = soc_logits[:, i]
        for i, idx in enumerate(self.gov_indices):
            full_logits[:, idx] = gov_logits[:, i]
        full_logits[:, self.no_metric_idx] = no_metric_logit.squeeze(-1)
        
        return full_logits

    def predict(self, input_ids, attention_mask):
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probas = torch.softmax(logits, dim=-1)
            confidences, predicted_ids = torch.max(probas, dim=-1)
        return predicted_ids.cpu().numpy(), confidences.cpu().numpy() 

