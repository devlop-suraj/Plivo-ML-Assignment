"""
Dataset preparation: JSONL -> tokenized BIO dataset
"""
import json
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset

from labels import LABEL2ID


def align_labels_with_tokens(labels: List[int], word_ids: List[int]) -> List[int]:
    """
    Align BIO labels with subword tokens.
    - First subword of a word gets the label
    - Subsequent subwords get the same label (for I-) or are marked as -100
    - Special tokens get -100
    """
    new_labels = []
    current_word = None
    
    for word_id in word_ids:
        if word_id is None:
            # Special token
            new_labels.append(-100)
        elif word_id != current_word:
            # Start of a new word
            current_word = word_id
            new_labels.append(labels[word_id])
        else:
            # Continuation of the same word
            # For I- labels, keep them; for B- labels, convert to I-
            label = labels[word_id]
            if label != LABEL2ID["O"] and label != -100:
                # Convert B- to I- for subwords
                label_name = list(LABEL2ID.keys())[label]
                if label_name.startswith("B-"):
                    entity_type = label_name[2:]
                    label = LABEL2ID[f"I-{entity_type}"]
            new_labels.append(label)
    
    return new_labels


def tokenize_and_align_labels(examples: List[Dict], tokenizer, max_length: int = 128):
    """
    Tokenize text and align BIO labels with tokens.
    """
    tokenized_inputs = tokenizer(
        [ex["text"] for ex in examples],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_offsets_mapping=True,
        is_split_into_words=False,
    )
    
    all_labels = []
    
    for i, example in enumerate(examples):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        
        # Create character-level label array
        text = example["text"]
        char_labels = ["O"] * len(text)
        
        # Fill in entity labels
        for entity in example.get("entities", []):
            start, end, label = entity["start"], entity["end"], entity["label"]
            # Ensure entity boundaries are within text length
            start = min(start, len(text) - 1)
            end = min(end, len(text))
            if start >= end:
                continue
            # Mark first character as B-
            char_labels[start] = f"B-{label}"
            # Mark remaining characters as I-
            for j in range(start + 1, end):
                if j < len(text):
                    char_labels[j] = f"I-{label}"
        
        # Get offset mapping for this example
        offsets = tokenized_inputs["offset_mapping"][i]
        
        # Align labels with tokens
        token_labels = []
        for idx, (start, end) in enumerate(offsets):
            if word_ids[idx] is None:
                # Special token
                token_labels.append(-100)
            elif start == end:
                # Empty token
                token_labels.append(-100)
            else:
                # Use the label of the first character in the token
                label = char_labels[start]
                token_labels.append(LABEL2ID[label])
        
        all_labels.append(token_labels)
    
    tokenized_inputs["labels"] = all_labels
    # Remove offset_mapping as it's not needed for training
    del tokenized_inputs["offset_mapping"]
    
    return tokenized_inputs


class NERDataset(Dataset):
    """PyTorch Dataset for NER."""
    
    def __init__(self, jsonl_file: str, tokenizer, max_length: int = 128):
        self.examples = []
        
        # Load JSONL
        with open(jsonl_file, 'r') as f:
            for line in f:
                self.examples.append(json.loads(line))
        
        # Tokenize and align labels
        self.tokenized = tokenize_and_align_labels(
            self.examples, tokenizer, max_length
        )
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.tokenized["input_ids"][idx]),
            "attention_mask": torch.tensor(self.tokenized["attention_mask"][idx]),
            "labels": torch.tensor(self.tokenized["labels"][idx]),
        }
