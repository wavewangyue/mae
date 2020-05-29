# baseline: most_common_value

import numpy as np

attr2value2num = {}

with open("./data/train/input.attr") as f:
    attrs = f.read().strip().split("\n")
with open("./data/train/output.value") as f:
    values = f.read().strip().split("\n")
print(len(attrs), len(values))
        
for attr, value in zip(attrs, values):
    if attr not in attr2value2num: attr2value2num[attr] = {}
    if value not in attr2value2num[attr]: attr2value2num[attr][value] = 0
    attr2value2num[attr][value] += 1

attr2value = {}
for attr, value2num in attr2value2num.items():
    attr2value[attr] = sorted(value2num.keys(), key=lambda v:value2num[v], reverse=True)[0]
    
with open("./data/test/input.attr") as f:
    attrs = f.read().strip().split("\n")
with open("./data/test/output.value") as f:
    values = f.read().strip().split("\n")
print(len(attrs), len(values))

hits = 0
for attr, value in zip(attrs, values):
    if attr in attr2value:
        pred = attr2value[attr]
        if pred == value:
            hits += 1
print(hits)
print(hits / len(attrs))

