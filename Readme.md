APPROACH:
- Used prajjwal1/bert-tiny (4.4M parameters) for fast CPU inference
- Trained on 800 synthetic examples with realistic STT noise patterns (spelled-out numbers, "at"/"dot" for emails)
- Dataset features Indian context: Indian names (raj kumar, priya sharma), cities (mumbai, bangalore), and locations (gateway of india, marine drive)

KEY OPTIMIZATIONS:
1. Entity-specific confidence thresholds to maximize PII precision:
   - CREDIT_CARD: 0.80 (high threshold to reduce false positives)
   - PHONE: 0.60
   - EMAIL: 0.55
   - PERSON_NAME: 0.75
   - DATE: 0.98 (very high to ensure precision)
   - CITY/LOCATION: 0.55

2. Training hyperparameters tuned for balance:
   - 6 epochs with dropout 0.2 to prevent overfitting
   - Learning rate 3e-4, batch size 24
   - Achieved final dev loss: 0.1148

RESULTS:
- PII Precision: 0.8529 (exceeds 0.80 target by 6.6%)
- p95 Latency: 3.06ms (6.5x faster than 20ms target)
- Overall F1: 0.7600

TRADE-OFFS:
- Prioritized precision over recall for PII entities as per requirements
- PERSON_NAME achieved perfect 1.0 F1 score
- DATE has lower recall (0.3636 F1) due to very high confidence threshold, but maintains perfect precision
- CREDIT_CARD precision improved to 0.68 through threshold tuning

TECHNICAL DECISIONS:
- Chose bert-tiny over distilbert for 5x faster inference with minimal accuracy loss
- BIO tagging scheme for robust span detection
- Post-processing with confidence thresholds instead of complex ensemble methods for simplicity and speed
