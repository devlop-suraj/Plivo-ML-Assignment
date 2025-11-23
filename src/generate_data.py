"""
Generate synthetic noisy STT datasets with PII entities.
Simulates STT errors: spelled-out numbers, "dot", "at", missing punctuation, etc.
"""
import json
import random
from typing import List, Dict, Tuple

# Entity templates with noisy STT patterns
PERSON_NAMES = [
    "raj kumar", "priya sharma", "amit singh", "neha patel", "rahul gupta",
    "anjali verma", "vikram reddy", "kavya iyer", "arjun mehta", "pooja desai",
    "rohan joshi", "sneha nair", "aditya rao", "divya krishnan", "karthik bhat",
    "ishita agarwal", "sanjay kulkarni", "deepika menon", "manish trivedi", "riya kapoor"
]

CITIES = [
    "mumbai", "delhi", "bangalore", "hyderabad", "chennai",
    "kolkata", "pune", "ahmedabad", "jaipur", "lucknow",
    "surat", "kanpur", "nagpur", "indore", "thane",
    "bhopal", "visakhapatnam", "vadodara", "ghaziabad", "ludhiana"
]

LOCATIONS = [
    "gateway of india", "india gate", "marine drive", "connaught place", "m g road",
    "janpath", "brigade road", "banjara hills", "juhu beach", "lake palace",
    "charminar", "victoria memorial", "hawa mahal", "qutub minar", "lotus temple"
]

DATES_PATTERNS = [
    ("january {day} {year}", "MONTH_DAY_YEAR"),
    ("february {day} {year}", "MONTH_DAY_YEAR"),
    ("march {day} {year}", "MONTH_DAY_YEAR"),
    ("april {day} {year}", "MONTH_DAY_YEAR"),
    ("may {day} {year}", "MONTH_DAY_YEAR"),
    ("june {day} {year}", "MONTH_DAY_YEAR"),
    ("july {day} {year}", "MONTH_DAY_YEAR"),
    ("august {day} {year}", "MONTH_DAY_YEAR"),
    ("september {day} {year}", "MONTH_DAY_YEAR"),
    ("october {day} {year}", "MONTH_DAY_YEAR"),
    ("november {day} {year}", "MONTH_DAY_YEAR"),
    ("december {day} {year}", "MONTH_DAY_YEAR"),
]

NUMBER_WORDS = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
}

def number_to_words(num_str: str) -> str:
    """Convert numeric string to spelled-out words (STT style)."""
    return ' '.join(NUMBER_WORDS[d] for d in num_str)

def generate_credit_card() -> Tuple[str, str]:
    """Generate noisy STT credit card number."""
    # Generate 16-digit card number
    card_digits = ''.join(random.choices('0123456789', k=16))
    
    # Convert to spoken format with variations
    variations = [
        # Fully spelled out
        number_to_words(card_digits),
        # Groups of 4 (common pattern)
        ' '.join(number_to_words(card_digits[i:i+4]) for i in range(0, 16, 4)),
    ]
    
    return random.choice(variations), card_digits

def generate_phone() -> Tuple[str, str]:
    """Generate noisy STT phone number."""
    # Generate 10-digit phone
    area = ''.join(random.choices('0123456789', k=3))
    prefix = ''.join(random.choices('0123456789', k=3))
    line = ''.join(random.choices('0123456789', k=4))
    
    # Convert to spoken format
    variations = [
        f"{number_to_words(area)} {number_to_words(prefix)} {number_to_words(line)}",
        f"{area[0]} {area[1]} {area[2]} {prefix[0]} {prefix[1]} {prefix[2]} {line}",
    ]
    
    return random.choice(variations), f"{area}-{prefix}-{line}"

def generate_email() -> Tuple[str, str]:
    """Generate noisy STT email address."""
    usernames = ["raj", "priya", "info", "contact", "support", "admin", "sales", "service"]
    domains = ["gmail", "yahoo", "rediffmail", "company", "business", "mail"]
    tlds = ["com", "in", "co dot in"]
    
    user = random.choice(usernames)
    domain = random.choice(domains)
    tld = random.choice(tlds)
    
    # STT transcribes as "at" and "dot"
    spoken = f"{user} at {domain} dot {tld}"
    actual = f"{user}@{domain}.{tld}"
    
    return spoken, actual

def generate_date() -> str:
    """Generate noisy STT date."""
    pattern, _ = random.choice(DATES_PATTERNS)
    day = random.randint(1, 28)
    year = random.randint(2020, 2025)
    
    # Spell out numbers sometimes
    if random.random() < 0.5:
        day_str = number_to_words(str(day))
    else:
        day_str = str(day)
    
    if random.random() < 0.3:
        year_str = number_to_words(str(year))
    else:
        year_str = str(year)
    
    return pattern.format(day=day_str, year=year_str)

TEMPLATES = [
    "my credit card number is {credit_card}",
    "you can reach me at {phone}",
    "send it to {email}",
    "my name is {person}",
    "i was born on {date}",
    "i live in {city}",
    "meet me at {location}",
    "call {person} at {phone}",
    "email {person} at {email}",
    "the transaction on {date} was for {credit_card}",
    "contact {person} in {city}",
    "{person} from {city} called about {credit_card}",
    "my card {credit_card} was used on {date}",
    "please call {phone} or email {email}",
    "{person} will be at {location} on {date}",
    "i visited {city} on {date}",
    "the event is at {location} in {city}",
    "card ending in {credit_card} phone {phone}",
    "{person} lives in {city} near {location}",
    "contact details are {phone} and {email}",
]

def generate_example() -> Dict:
    """Generate a single training example with entities."""
    template = random.choice(TEMPLATES)
    text = template
    entities = []
    
    # Track what placeholders are in this template
    placeholders = {
        'credit_card': 'CREDIT_CARD',
        'phone': 'PHONE',
        'email': 'EMAIL',
        'person': 'PERSON_NAME',
        'date': 'DATE',
        'city': 'CITY',
        'location': 'LOCATION'
    }
    
    # Replace placeholders
    for placeholder, label in placeholders.items():
        if '{' + placeholder + '}' in text:
            if placeholder == 'credit_card':
                value, _ = generate_credit_card()
            elif placeholder == 'phone':
                value, _ = generate_phone()
            elif placeholder == 'email':
                value, _ = generate_email()
            elif placeholder == 'person':
                value = random.choice(PERSON_NAMES)
            elif placeholder == 'date':
                value = generate_date()
            elif placeholder == 'city':
                value = random.choice(CITIES)
            elif placeholder == 'location':
                value = random.choice(LOCATIONS)
            
            # Find position and replace
            start = text.find('{' + placeholder + '}')
            text = text.replace('{' + placeholder + '}', value, 1)
            end = start + len(value)
            
            entities.append({
                'start': start,
                'end': end,
                'label': label
            })
    
    # Sort entities by start position
    entities.sort(key=lambda x: x['start'])
    
    return {
        'text': text,
        'entities': entities
    }

def generate_dataset(num_examples: int, start_id: int = 0) -> List[Dict]:
    """Generate a dataset with the specified number of examples."""
    dataset = []
    for i in range(num_examples):
        example = generate_example()
        example['id'] = f"utt_{start_id + i:04d}"
        dataset.append(example)
    return dataset

def main():
    """Generate train, dev, and test datasets."""
    random.seed(42)
    
    # Generate datasets
    train_data = generate_dataset(800, start_id=0)
    dev_data = generate_dataset(150, start_id=800)
    test_data = generate_dataset(100, start_id=950)
    
    # Remove labels from test data
    test_data_unlabeled = [{'id': ex['id'], 'text': ex['text']} for ex in test_data]
    
    # Save datasets
    with open('data/train.jsonl', 'w') as f:
        for example in train_data:
            f.write(json.dumps(example) + '\n')
    
    with open('data/dev.jsonl', 'w') as f:
        for example in dev_data:
            f.write(json.dumps(example) + '\n')
    
    with open('data/test.jsonl', 'w') as f:
        for example in test_data_unlabeled:
            f.write(json.dumps(example) + '\n')
    
    # Also save test with labels for final evaluation
    with open('data/test_labeled.jsonl', 'w') as f:
        for example in test_data:
            f.write(json.dumps(example) + '\n')
    
    print(f"Generated {len(train_data)} training examples")
    print(f"Generated {len(dev_data)} dev examples")
    print(f"Generated {len(test_data)} test examples")
    
    # Show a sample
    print("\nSample example:")
    print(json.dumps(train_data[0], indent=2))

if __name__ == '__main__':
    main()
