# Entity labels and PII mapping

# All entity types
LABELS = [
    "O",  # Outside any entity
    "B-CREDIT_CARD",
    "I-CREDIT_CARD",
    "B-PHONE",
    "I-PHONE",
    "B-EMAIL",
    "I-EMAIL",
    "B-PERSON_NAME",
    "I-PERSON_NAME",
    "B-DATE",
    "I-DATE",
    "B-CITY",
    "I-CITY",
    "B-LOCATION",
    "I-LOCATION",
]

# Mapping from label index to label name
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

# PII entity types (these should be flagged as PII=true)
PII_ENTITIES = {
    "CREDIT_CARD",
    "PHONE",
    "EMAIL",
    "PERSON_NAME",
    "DATE",
}

# Non-PII entity types (these should be flagged as PII=false)
NON_PII_ENTITIES = {
    "CITY",
    "LOCATION",
}

def is_pii(entity_type: str) -> bool:
    """Check if an entity type is PII."""
    return entity_type in PII_ENTITIES
