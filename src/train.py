"""
Training script for NER model.
"""
import argparse
import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

from dataset import NERDataset
from model import NERModel
from labels import LABELS, LABEL2ID, ID2LABEL


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs["loss"]
        total_loss += loss.item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs["loss"].item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--dev", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load datasets
    print(f"Loading datasets...")
    train_dataset = NERDataset(args.train, tokenizer, args.max_length)
    dev_dataset = NERDataset(args.dev, tokenizer, args.max_length)
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Dev examples: {len(dev_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)
    
    # Create model
    print(f"Creating model...")
    model = NERModel(
        model_name=args.model_name,
        num_labels=len(LABELS),
        dropout=args.dropout
    )
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    best_dev_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        dev_loss = evaluate(model, dev_loader, device)
        
        print(f"Train loss: {train_loss:.4f}")
        print(f"Dev loss: {dev_loss:.4f}")
        
        # Save best model
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            print(f"Saving best model...")
            
            # Save model
            torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))
            
            # Save tokenizer
            tokenizer.save_pretrained(args.out_dir)
            
            # Save config
            config = {
                "model_name": args.model_name,
                "num_labels": len(LABELS),
                "dropout": args.dropout,
                "max_length": args.max_length,
                "labels": LABELS,
                "label2id": LABEL2ID,
                "id2label": ID2LABEL,
            }
            with open(os.path.join(args.out_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Best dev loss: {best_dev_loss:.4f}")


if __name__ == "__main__":
    main()
