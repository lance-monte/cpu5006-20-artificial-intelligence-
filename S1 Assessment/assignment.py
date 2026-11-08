import random #random shuffles the data for the train/test split. 
from collections import Counter #Counter counts how often each class appears in a list.

import pandas as pd #reads car.data and prints tables
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix #calculates accuracy, precision, recall, and F1 score

#literally just a list of the column names in the dataset, so we can refer to them by name instead of index.
COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
TARGET = "class"
CLASSES = ["unacc", "acc", "good", "vgood"] #the four possible answers, from worst to best. use this list so the table always prints the classes in the same order, even if some of them don't appear in the predictions.

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
    #order matters meaning that the first class rule that matches wins
    #deal-breakers would go first because those cars are always unacceptable
    #then vgood, good and acc so a car that matches both vgood and acc gets vgood
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
CLASS_GOALS = ["class=unacc", "class=vgood", "class=good", "class=acc"] #the priority order.
#it must match the order of the rules otherwise backward chain and forward chain will get different answers.

#forward chaining is a way to apply rules to a set of facts to infer new facts.
#passes through the rules repeatedly until no new facts can be inferred.
def forward_chain(facts, rules):
    known = set(facts)
    changed = True 
    while changed:
        changed = False
        for conditions, conclusion in rules:
            if conditions <= known and conclusion not in known: #conditions <= known means all conditions are in known. conclusion not in known means we haven't already inferred this fact and stops infinite loops.
                known.add(conclusion)
                changed = True
    return known

#same as forward_chain, but also returns a trace of which rules were applied to infer which facts.
#it can explain how it got to its conclusion, which is useful for debugging and understanding the rules.
def forward_chain_trace(facts, rules):
    known, trace= set(facts), []
    changed = True 
    while changed:
        changed = False
        for conditions, conclusion in rules:
            if conditions <= known and conclusion not in known:
                known.add(conclusion)
                trace.append((sorted(conditions), conclusion))
                changed = True
    return known, trace

#predict_forward takes a row of data and a set of rules, and returns the predicted class for that row.
#forward chaining can find more than one class and this function picks the first matching class rule.
def predict_forward(row,rules):
    known = forward_chain(row_to_facts(row), rules)
    for conditions, conclusion in rules:
        if conclusion.startswith("class=") and conditions <= known:
            return conclusion.split("=")[1]
    return DEFAULT_CLASS

#backward chaining is a way where it proves a goal by proving that all the conditions for a rule that concludes the goal are true.
#each condition then becomes a sub-goal so that the function calls itself which is recursion
#if a known fact matches the goal, it returns true. if not, it looks for rules that conclude the goal and tries to prove all their conditions.
def backward_chain(goal, facts, rules):
    if goal in facts:
        return True
    for conditions, conclusion in rules:
        if conclusion == goal:
            if all(backward_chain(c, facts, rules) for c in sorted(conditions)):
                return True
    return False

#prove is the same as backward_chain, but it also prints a trace of which rules were applied to prove which goals.
def prove(goal, facts, rules, depth=0):
    indent = "  " * depth
    if goal in facts:
        print(f"{indent}{goal}: known fact")
        return True
    for conditions, conclusion in rules:
        if conclusion == goal:
            print(f"{indent}{goal}: try rule needing {sorted(conditions)}")
            if all(prove(c, facts, rules, depth +1) for c in sorted(conditions)):
                print(f"{indent}{goal}: proved")
                return True
    print(f"{indent}{goal}: cannot prove")
    return False

#using backward_chain it classifies the car by proving each of the possible classes in order, and returning the first one that can be proved. if none can be proved, it returns the default class.
def predict_backward(row, rules):
    row_facts = row_to_facts(row)
    proven = [g for g in CLASS_GOALS if backward_chain(g, row_facts, rules)]
    predicted = proven[0] if proven else "class=" + DEFAULT_CLASS
    return predicted.split("=")[1]

y_true = [r[TARGET] for r in test] #the actual classes for the test set, used to calculate accuracy, precision, recall, and F1 score.
majority = Counter(r[TARGET] for r in train).most_common(1)[0][0] #the most common class in the train set, used by baseline accuracy.

#This shows each approach paired to its predictions for the test set. The baseline is just the majority class, the first three rules are the simple classifier, and forward and backward chaining are the more complex classifiers.
#this way all four approaches can be compared on the same test set, and their accuracy, precision, recall, and F1 score can be calculated.
methods = {
    f"Baseline (always '{majority}')": [majority for r in test],
    "First three rules": [classify(r) for r in test],
    "Forward chaining": [predict_forward(r, RULES) for r in test],
    "Backward chaining": [predict_backward(r, RULES) for r in test]
}

#this prints out the different methods within "methods".
#it prints the accuracy, precision, recall, f1 score(for each class) and a confusion matrix. 
for name, y_pred in methods.items():
    print (f"\n{name}:")
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2f}\n")
    print(classification_report(y_true, y_pred, labels=CLASSES, zero_division=0)) #zero_division=0 means if a class is not predicted at all, don't throw an error, just say 0 precision and recall.
    print("Confusion matrix (rows = actual class, columns = predicted class,")
    print("both in the order unacc, acc, good, vgood):")
    print(confusion_matrix(y_true, y_pred, labels=CLASSES))

#tests how many test cars forward and backward chaining gave the same answer for. 
#checks both methods are consistent, they use same rules and same priority order, so they should always agree; they  only differ in how they reach the answer
agree = sum(f == b for f, b in zip(methods["Forward chaining"], methods["Backward chaining"]))
print(f"\nForward and backward chaining agree on {agree} out of {len(test)} test cases ({agree/len(test):.2%})")

#makes a list of the test cars the first three rules got wrong 
#looking at the mistakes show what the rules are missing, the first car printed on code line 229 is predicted unacc but the actual class is acc because the first three rules dont consider price.
#this is why the rules section include price rules.
print("\nThree test cars the first three rules got wrong:")
wrong = [r for r in test if classify(r) != r[TARGET]]
for r in wrong[:3]:
    car = {a: v for a, v in r.items() if a != TARGET}
    print(f"{car} predicted: {classify(r)}, actual: {r[TARGET]}")
print(f"(The first three rules got {len(wrong)} out of {len(test)} wrong.)")
print(f"Forward chaining got {sum(p != t for p, t in zip(methods['Forward chaining'], y_true))} out of {len(test)} wrong.")

#Demo section: shows how each method reaches its answers for a few test cars 
for number, row in enumerate(test[:5], start=1):
    car = {a: v for a, v in row.items() if a != TARGET}
    print(f"\nCar {number}: {car}")
    print(f"True Class: {row[TARGET]}")
    print("FORWARD CHAINING - rules that fired:")
    known, trace = forward_chain_trace(row_to_facts(car), RULES)
    for conditions, conclusion in trace:
        print(f" {' + '.join(conditions)} -> {conclusion}")
    print(f"Answer: {predict_forward(row, RULES)}")

    print("\nBACKWARD CHAINING - proof:")
    facts = row_to_facts(car)
    for goal in CLASS_GOALS:
        print(f"\nGoal: {goal}")
        if prove(goal, facts, RULES):
            break
    print(f"\nAnswer: {predict_backward(car, RULES)}")