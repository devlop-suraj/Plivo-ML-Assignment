"""
Measure inference latency for NER model.
"""
import argparse
import json
import time
import torch
import numpy as np
from transformers import AutoTokenizer

from model import NERModel


def measure_latency(model, tokenizer, texts, device, max_length=128, runs=50):
    """
    Measure inference latency.
    Returns p50 and p95 latencies in milliseconds.
    """
    model.eval()
    latencies = []
    
    for _ in range(runs):
        # Sample a random text
        text = texts[np.random.randint(0, len(texts))]
        
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
        
        # Measure inference time
        start_time = time.perf_counter()
        
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
        end_time = time.perf_counter()
        
        # Convert to milliseconds
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
    
    # Compute percentiles
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    mean = np.mean(latencies)
    
    return {
        "p50": p50,
        "p95": p95,
        "mean": mean,
        "latencies": latencies,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()
    
    # Device (force CPU for latency measurement)
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load config
    with open(f"{args.model_dir}/config.json", "r") as f:
        config = json.load(f)
    
    # Load tokenizer
    print(f"Loading tokenizer...")
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
    
    # Load texts
    with open(args.input, "r") as f:
        examples = [json.loads(line) for line in f]
    
    texts = [ex["text"] for ex in examples]
    
    print(f"\nMeasuring latency over {args.runs} runs...")
    print(f"Dataset: {len(texts)} utterances")
    
    # Warm-up
    print("Warming up...")
    for _ in range(5):
        text = texts[0]
        encoding = tokenizer(text, truncation=True, padding="max_length", 
                           max_length=config["max_length"], return_tensors="pt")
        with torch.no_grad():
            _ = model(input_ids=encoding["input_ids"].to(device), 
                     attention_mask=encoding["attention_mask"].to(device))
    
    # Measure
    results = measure_latency(model, tokenizer, texts, device, 
                             config["max_length"], args.runs)
    
    # Print results
    print("\n" + "=" * 60)
    print("LATENCY RESULTS (batch_size=1, CPU)")
    print("=" * 60)
    print(f"Mean:  {results['mean']:.2f} ms")
    print(f"p50:   {results['p50']:.2f} ms")
    print(f"p95:   {results['p95']:.2f} ms")
    print("=" * 60)
    
    # Check if latency target is met
    if results['p95'] <= 20.0:
        print(f"✓ Latency target MET: p95 = {results['p95']:.2f} ms ≤ 20 ms")
    else:
        print(f"✗ Latency target NOT MET: p95 = {results['p95']:.2f} ms > 20 ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
