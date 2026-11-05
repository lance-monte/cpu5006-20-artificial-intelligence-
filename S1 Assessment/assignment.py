import random
from collections import Counter 

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
TARGET = "class"
CLASSES = ["unacc", "acc", "good", "vgood"] 

df = pd.read_csv("car.data", header=None, names=COLUMNS, dtype=str)

print(f"Number of cars: {len(df)}")
print("\nFirst 5 cars:")
print(df.head())

print("\nClass distribution:")
print(df[TARGET].value_counts())
print("\nClass distribution (percentage):")
print(df[TARGET].value_counts(normalize=True) * 100)

baseline = df[TARGET].value_counts(normalize=True).max()
print(f"\nBaseline accuracy: {baseline:.2f}")

first_three_rules = [
    ({"safety": "low"}, "unacc"),
    ({"persons": "2"}, "unacc"),
    ({"safety": "high"}, "acc")
]

DEFAULT = "unacc"

def classify(row):
    for conditions, label in first_three_rules:
        matches = True
        for attribute, value in conditions.items():
            if row[attribute] != value:
                matches = False
        if matches:
            return label
    return DEFAULT

rows = df.to_dict(orient="records")
three_rule_accuracy = sum(classify(r) == r[TARGET] for r in rows) / len(rows)

print(f"3-rule accuracy: {three_rule_accuracy:.2f} (baseline: {baseline:.2f})")
print("Better than baseline, but not perfect")

