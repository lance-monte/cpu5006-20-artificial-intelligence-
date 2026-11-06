import random #random shuffles the data for the train/test split. 
from collections import Counter #Counter counts how often each class appears in a list.

import pandas as pd #reads car.data and prints tables
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix #calculates accuracy, precision, recall, and F1 score

#literally just a list of the column names in the dataset, so we can refer to them by name instead of index.
COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
TARGET = "class"
CLASSES = ["unacc", "acc", "good", "vgood"] #the four possible answers, from worst to best. use this list so the table always prints the classes in the same order, even if some of them don't appear in the dataset.

df = pd.read_csv("car.data", header=None, names=COLUMNS, dtype=str) 
#header=none means first row is data, not column names. 
#names=columns means use the list of column names we defined above. 
#dtype=str means treat all columns as strings, even if they look like numbers. Means i can use string comparison instead of numeric comparison.

print(f"Number of cars: {len(df)}") #shows how many cars are in the dataset.
print("\nFirst 5 cars:")
print(df.head()) #shows the first 5 rows of the dataset

#the classes are not evenly distributed, so we need to know how many of each class there are. 
#around 70% are "unacc", around 22% are "acc", around 4% are "good" and "vgood". 
print("\nClass distribution:")
print(df[TARGET].value_counts())
print("\nClass distribution (percentage):")
print(df[TARGET].value_counts(normalize=True) * 100)

# baseline is basically the mode, the most common class which is "unacc"
baseline = df[TARGET].value_counts(normalize=True).max()
print(f"\nBaseline accuracy: {baseline:.2f}")


#each rule is a (condition, answer) pair. the first matching rule is used to classify the car. if no rules match, the default answer is used.
#order matters, so the first rule that matches is used. Rules 1 and 2 are always right, every low-safety car is unacc and every 2-person car is unacc.
#rule 3 is a bit more complicated, it says that every high-safety car is acc. This is not always true, but it is true often enough to be useful.
#its kinda common sense, if a car is safe, it is probably at least acceptable. The rules are not perfect, but they are better than the baseline.
first_three_rules = [
    ({"safety": "low"}, "unacc"),
    ({"persons": "2"}, "unacc"),
    ({"safety": "high"}, "acc")
]

DEFAULT = "unacc" #when a car matches none of the rules

#loops through the rules and checks if the car matches any of them
#if it does, return the answer for that rule. if it doesn't match any rules, return the default answer.
def classify(row):
    for conditions, label in first_three_rules:
        matches = True
        for attribute, value in conditions.items():
            if row[attribute] != value:
                matches = False
        if matches:
            return label
    return DEFAULT

#list of dictionaries, one per car. 
#calculates the accuracy of the 3-rule classifier by comparing the predicted class to the actual class for each car in the dataset.
rows = df.to_dict(orient="records")
three_rule_accuracy = sum(classify(r) == r[TARGET] for r in rows) / len(rows)

print(f"3-rule accuracy: {three_rule_accuracy:.2f} (baseline: {baseline:.2f})")
print("Better than baseline, but not perfect")

#Train and test
#the reason for the test is to see how well the rules generalize to new data. 
#If the accuracy on the test set is much lower than the accuracy on the training set, it means the rules are overfitting to the training data and not generalizing well.

#car.data is not randomly ordered, so we need to shuffle it before splitting into train and test sets.
#the seed makes the shuffle the same every run, so the results are reproducible. 
random.seed(42)
shuffled = rows[:]
random.shuffle(shuffled)

cut = int(0.7 * len(shuffled)) #works out 70% of the dataset for train, the rest is for test.
train, test = shuffled[:cut], shuffled[cut:]

print(f"Train rows: {len(train)}, Test rows: {len(test)}")
print("Test set class counts:", dict(Counter(r[TARGET] for r in test)))

def row_to_facts(row): #turns a car into a set of facts, which is the form the rules use.
    return {f"{attribute}={value}" for attribute, value in row.items() if attribute != TARGET}

RULES = [
    #price rules combine buying and maint to determine price.
    ({"buying=vhigh", "maint=vhigh"}, "price=unaffordable"),
    ({"buying=vhigh", "maint=high"}, "price=unaffordable"),
    ({"buying=high", "maint=vhigh"}, "price=unaffordable"),
    ({"buying=med", "maint=low"}, "price=low"),
    ({"buying=low", "maint=med"}, "price=low"),
    ({"buying=low", "maint=low"}, "price=low"),
    ({"buying=med", "maint=med"}, "price=medium"),
    ({"buying=low", "maint=high"}, "price=medium"),

    #comfort rules combine lug_boot, doors, and persons to determine comfort.
    ({"lug_boot=big"}, "comfort=high"),
    ({"lug_boot=med", "doors=4"}, "comfort=high"),
    ({"lug_boot=med", "doors=5more"}, "comfort=high"),
    ({"lug_boot=med", "doors=3", "persons=more"}, "comfort=high"),
    ({"lug_boot=small"}, "comfort=medium"),
    ({"lug_boot=med", "doors=2"}, "comfort=medium"),
    ({"lug_boot=med", "doors=3", "persons=4"}, "comfort=medium"),

    #tech rules combine safety and comfort to determine tech.
    ({"safety=high", "comfort=high"}, "tech=excellent"),
    ({"safety=high", "comfort=medium"}, "tech=good"),
    ({"safety=med", "comfort=high"}, "tech=good"),
    ({"safety=med", "comfort=medium"}, "tech=acceptable"),

    #class rules combine price and tech to determine class.
    ({"safety=low"}, "class=unacc"),
    ({"persons=2"}, "class=unacc"),
    ({"persons=more", "doors=2", "lug_boot=small"}, "class=unacc"),
    ({"price=unaffordable"}, "class=unacc"),
    ({"tech=excellent", "price=low"}, "class=vgood"),
    ({"tech=excellent", "price=medium"}, "class=vgood"),
    ({"tech=good", "price=low"}, "class=good"),
    ({"tech=excellent"}, "class=acc"),
    ({"tech=good"}, "class=acc"),
    ({"tech=acceptable", "price=low"}, "class=acc"),
    ({"tech=acceptable", "price=medium"}, "class=acc"),
]

DEFAULT_CLASS = "unacc" #when a car matches none of the rules
CLASS_GOALS = ["class=unacc", "class=acc", "class=good", "class=vgood"] #the four possible answers, from worst to best.

def forward_chain(facts, rules):
    known = set(facts)
    changed = True 
    while changed:
        changed = False
        for conditions, conclusion in rules:
            if conditions <= known and conclusion not in known:
                known.add(conclusion)
                changed = True
    return known


