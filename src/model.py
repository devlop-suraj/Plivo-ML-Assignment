"""
Token classification model for NER.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class NERModel(nn.Module):
    """Token classification model with BERT-style encoder."""
    
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Classifier head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        self.num_labels = num_labels
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Get sequence output (last hidden state)
        sequence_output = outputs.last_hidden_state
        
        # Apply dropout and classifier
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            # Flatten the tokens
            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.view(-1)
            )
        
        return {
            "loss": loss,
            "logits": logits,
        }
