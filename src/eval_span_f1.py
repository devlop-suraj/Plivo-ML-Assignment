"""
Span-level F1 evaluation with PII-specific metrics.
"""
import argparse
import json
from collections import defaultdict
from typing import List, Dict, Set, Tuple

from labels import is_pii


def span_to_tuple(entity: Dict) -> Tuple:
    """Convert entity dict to tuple for comparison."""
    return (entity["start"], entity["end"], entity["label"])


def compute_metrics(gold_entities: List[Dict], pred_entities: List[Dict]) -> Dict:
    """
    Compute precision, recall, F1 for entity spans.
    """
    gold_set = set(span_to_tuple(e) for e in gold_entities)
    pred_set = set(span_to_tuple(e) for e in pred_entities)
    
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_per_entity_metrics(gold_examples: List[Dict], pred_examples: List[Dict]) -> Dict:
    """
    Compute metrics per entity type.
    """
    # Group entities by type
    gold_by_type = defaultdict(list)
    pred_by_type = defaultdict(list)
    
    for gold_ex, pred_ex in zip(gold_examples, pred_examples):
        for entity in gold_ex.get("entities", []):
            gold_by_type[entity["label"]].append(span_to_tuple(entity))
        
        for entity in pred_ex.get("entities", []):
            pred_by_type[entity["label"]].append(span_to_tuple(entity))
    
    # Compute metrics per type
    per_type_metrics = {}
    
    for entity_type in set(list(gold_by_type.keys()) + list(pred_by_type.keys())):
        gold_set = set(gold_by_type[entity_type])
        pred_set = set(pred_by_type[entity_type])
        
        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_type_metrics[entity_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": len(gold_set),
        }
    
    return per_type_metrics


def compute_pii_metrics(gold_examples: List[Dict], pred_examples: List[Dict]) -> Dict:
    """
    Compute metrics specifically for PII entities.
    """
    gold_pii = []
    pred_pii = []
    gold_non_pii = []
    pred_non_pii = []
    
    for gold_ex, pred_ex in zip(gold_examples, pred_examples):
        for entity in gold_ex.get("entities", []):
            if is_pii(entity["label"]):
                gold_pii.append(span_to_tuple(entity))
            else:
                gold_non_pii.append(span_to_tuple(entity))
        
        for entity in pred_ex.get("entities", []):
            if is_pii(entity["label"]):
                pred_pii.append(span_to_tuple(entity))
            else:
                pred_non_pii.append(span_to_tuple(entity))
    
    # Compute PII metrics
    gold_pii_set = set(gold_pii)
    pred_pii_set = set(pred_pii)
    
    tp_pii = len(gold_pii_set & pred_pii_set)
    fp_pii = len(pred_pii_set - gold_pii_set)
    fn_pii = len(gold_pii_set - pred_pii_set)
    
    pii_precision = tp_pii / (tp_pii + fp_pii) if (tp_pii + fp_pii) > 0 else 0.0
    pii_recall = tp_pii / (tp_pii + fn_pii) if (tp_pii + fn_pii) > 0 else 0.0
    pii_f1 = 2 * pii_precision * pii_recall / (pii_precision + pii_recall) if (pii_precision + pii_recall) > 0 else 0.0
    
    # Compute non-PII metrics
    gold_non_pii_set = set(gold_non_pii)
    pred_non_pii_set = set(pred_non_pii)
    
    tp_non_pii = len(gold_non_pii_set & pred_non_pii_set)
    fp_non_pii = len(pred_non_pii_set - gold_non_pii_set)
    fn_non_pii = len(gold_non_pii_set - pred_non_pii_set)
    
    non_pii_precision = tp_non_pii / (tp_non_pii + fp_non_pii) if (tp_non_pii + fp_non_pii) > 0 else 0.0
    non_pii_recall = tp_non_pii / (tp_non_pii + fn_non_pii) if (tp_non_pii + fn_non_pii) > 0 else 0.0
    non_pii_f1 = 2 * non_pii_precision * non_pii_recall / (non_pii_precision + non_pii_recall) if (non_pii_precision + non_pii_recall) > 0 else 0.0
    
    return {
        "pii": {
            "precision": pii_precision,
            "recall": pii_recall,
            "f1": pii_f1,
            "support": len(gold_pii_set),
        },
        "non_pii": {
            "precision": non_pii_precision,
            "recall": non_pii_recall,
            "f1": non_pii_f1,
            "support": len(gold_non_pii_set),
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=str, required=True)
    parser.add_argument("--pred", type=str, required=True)
    args = parser.parse_args()
    
    # Load data
    with open(args.gold, "r") as f:
        gold_examples = [json.loads(line) for line in f]
    
    with open(args.pred, "r") as f:
        pred_examples = [json.loads(line) for line in f]
    
    # Overall metrics
    all_gold_entities = []
    all_pred_entities = []
    
    for gold_ex, pred_ex in zip(gold_examples, pred_examples):
        all_gold_entities.extend([span_to_tuple(e) for e in gold_ex.get("entities", [])])
        all_pred_entities.extend([span_to_tuple(e) for e in pred_ex.get("entities", [])])
    
    overall_metrics = compute_metrics(
        [{"start": s, "end": e, "label": l} for s, e, l in all_gold_entities],
        [{"start": s, "end": e, "label": l} for s, e, l in all_pred_entities]
    )
    
    # Per-entity metrics
    per_entity_metrics = compute_per_entity_metrics(gold_examples, pred_examples)
    
    # PII metrics
    pii_metrics = compute_pii_metrics(gold_examples, pred_examples)
    
    # Print results
    print("=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)
    print(f"Precision: {overall_metrics['precision']:.4f}")
    print(f"Recall:    {overall_metrics['recall']:.4f}")
    print(f"F1:        {overall_metrics['f1']:.4f}")
    print(f"TP: {overall_metrics['tp']}, FP: {overall_metrics['fp']}, FN: {overall_metrics['fn']}")
    
    print("\n" + "=" * 60)
    print("PER-ENTITY METRICS")
    print("=" * 60)
    for entity_type in sorted(per_entity_metrics.keys()):
        metrics = per_entity_metrics[entity_type]
        print(f"\n{entity_type}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Support:   {metrics['support']}")
    
    print("\n" + "=" * 60)
    print("PII vs NON-PII METRICS")
    print("=" * 60)
    print("\nPII Entities (CREDIT_CARD, PHONE, EMAIL, PERSON_NAME, DATE):")
    print(f"  Precision: {pii_metrics['pii']['precision']:.4f}")
    print(f"  Recall:    {pii_metrics['pii']['recall']:.4f}")
    print(f"  F1:        {pii_metrics['pii']['f1']:.4f}")
    print(f"  Support:   {pii_metrics['pii']['support']}")
    
    print("\nNon-PII Entities (CITY, LOCATION):")
    print(f"  Precision: {pii_metrics['non_pii']['precision']:.4f}")
    print(f"  Recall:    {pii_metrics['non_pii']['recall']:.4f}")
    print(f"  F1:        {pii_metrics['non_pii']['f1']:.4f}")
    print(f"  Support:   {pii_metrics['non_pii']['support']}")
    
    # Check if PII precision meets target
    print("\n" + "=" * 60)
    if pii_metrics['pii']['precision'] >= 0.80:
        print(f"✓ PII precision target MET: {pii_metrics['pii']['precision']:.4f} ≥ 0.80")
    else:
        print(f"✗ PII precision target NOT MET: {pii_metrics['pii']['precision']:.4f} < 0.80")
    print("=" * 60)


if __name__ == "__main__":
    main()
