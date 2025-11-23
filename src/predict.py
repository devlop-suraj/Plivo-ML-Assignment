"""
Final improved inference with entity-specific confidence thresholds for maximum PII precision.
"""
import argparse
import json
import torch
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

from model import NERModel
from labels import is_pii


# Entity-specific confidence thresholds tuned for PII precision >= 0.80
# Aggressively prioritizing precision over recall for PII entities
ENTITY_THRESHOLDS = {
    "CREDIT_CARD": 0.985,  # Maximum threshold - critical PII
    "PHONE": 0.94,
    "EMAIL": 0.84,
    "PERSON_NAME": 0.55,  # Already perfect precision
    "DATE": 0.92,
    "CITY": 0.70,
    "LOCATION": 0.70,
}


def decode_predictions_final(text: str, logits, tokenizer):
    """
    Decode BIO predictions with entity-specific confidence thresholds.
    """
    entities = []
    current_entity = None
    current_entity_logits = []
    
    # Get token to character mapping
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = encoding["offset_mapping"]
    
    # Get predictions and confidence scores
    probs = torch.softmax(torch.tensor(logits), dim=-1)
    predictions = torch.argmax(probs, dim=-1).numpy()
    confidences = torch.max(probs, dim=-1).values.numpy()
    
    from labels import ID2LABEL
    
    for idx, (pred_id, conf, (start, end)) in enumerate(zip(predictions, confidences, offsets)):
        # Skip special tokens and padding
        if start == end:
            if current_entity and current_entity_logits:
                # Check average confidence for entity
                avg_conf = np.mean(current_entity_logits)
                entity_type = current_entity["label"]
                threshold = ENTITY_THRESHOLDS.get(entity_type, 0.75)
                
                if avg_conf >= threshold:
                    entities.append(current_entity)
            
            current_entity = None
            current_entity_logits = []
            continue
        
        pred_label = pred_id
        
        if pred_label == 0:  # "O" label
            # End of entity
            if current_entity and current_entity_logits:
                avg_conf = np.mean(current_entity_logits)
                entity_type = current_entity["label"]
                threshold = ENTITY_THRESHOLDS.get(entity_type, 0.75)
                
                if avg_conf >= threshold:
                    entities.append(current_entity)
            
            current_entity = None
            current_entity_logits = []
        
        elif pred_label > 0:
            pred_label_name = ID2LABEL[int(pred_label)]
            
            if pred_label_name.startswith("B-"):
                # Start of new entity
                if current_entity and current_entity_logits:
                    avg_conf = np.mean(current_entity_logits)
                    entity_type = current_entity["label"]
                    threshold = ENTITY_THRESHOLDS.get(entity_type, 0.75)
                    
                    if avg_conf >= threshold:
                        entities.append(current_entity)
                
                entity_type = pred_label_name[2:]
                current_entity = {
                    "start": start,
                    "end": end,
                    "label": entity_type
                }
                current_entity_logits = [conf]
            
            elif pred_label_name.startswith("I-"):
                # Continuation of entity
                entity_type = pred_label_name[2:]
                
                if current_entity and current_entity["label"] == entity_type:
                    # Extend current entity
                    current_entity["end"] = end
                    current_entity_logits.append(conf)
                else:
                    # Start new entity
                    if current_entity and current_entity_logits:
                        avg_conf = np.mean(current_entity_logits)
                        old_entity_type = current_entity["label"]
                        threshold = ENTITY_THRESHOLDS.get(old_entity_type, 0.75)
                        
                        if avg_conf >= threshold:
                            entities.append(current_entity)
                    
                    current_entity = {
                        "start": start,
                        "end": end,
                        "label": entity_type
                    }
                    current_entity_logits = [conf]
    
    # Add final entity
    if current_entity and current_entity_logits:
        avg_conf = np.mean(current_entity_logits)
        entity_type = current_entity["label"]
        threshold = ENTITY_THRESHOLDS.get(entity_type, 0.75)
        
        if avg_conf >= threshold:
            entities.append(current_entity)
    
    return entities


def predict(model, tokenizer, text, device, max_length=128):
    """Make prediction for a single text."""
    model.eval()
    
    # Tokenize
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
    
    return logits[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Using entity-specific confidence thresholds:")
    for entity, threshold in sorted(ENTITY_THRESHOLDS.items()):
        print(f"  {entity}: {threshold}")
    
    # Load config
    with open(f"{args.model_dir}/config.json", "r") as f:
        config = json.load(f)
    
    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    # Load model
    print(f"Loading model...")
    model = NERModel(
        model_name=config["model_name"],
        num_labels=config["num_labels"],
        dropout=config["dropout"]
    )
    model.load_state_dict(torch.load(f"{args.model_dir}/model.pt", map_location=device))
    model.to(device)
    model.eval()
    
    # Load input
    with open(args.input, "r") as f:
        examples = [json.loads(line) for line in f]
    
    # Predict
    predictions = []
    
    for example in tqdm(examples, desc="Predicting"):
        text = example["text"]
        
        # Get predictions with logits
        logits = predict(model, tokenizer, text, device, config["max_length"])
        
        # Decode to spans with entity-specific confidence filtering
        entities = decode_predictions_final(text, logits, tokenizer)
        
        predictions.append({
            "id": example["id"],
            "text": text,
            "entities": entities
        })
    
    # Save predictions
    with open(args.output, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    
    print(f"\nPredictions saved to {args.output}")


if __name__ == "__main__":
    main()
