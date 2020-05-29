# make vocab_word

lines = set()
with open("./train/input.seq") as f:
    for line in f.read().strip().split("\n"):
        lines.add(line)
with open("./test/input.seq") as f:
    for line in f.read().strip().split("\n"):
        lines.add(line)
print(len(lines))
        
w2num = {}
for line in lines:
    for word in line.split(" "):
        if word not in w2num: w2num[word] = 0
        w2num[word] += 1
        
print("all words:", len(w2num))
w2num = {w: num for w, num in w2num.items() if num > 20}
print("remain words:", len(w2num))
words = ["[PAD]", "[UNK]", "[SEP]"] + sorted(w2num.keys(), key=lambda v:w2num[v], reverse=True)

with open("./vocab_word.txt", "w") as w:
    for word in words:
        w.write(word + "\n")
        
# make vocab_attr

attrs = set()
with open("./train/input.attr") as f:
    for line in f.read().strip().split("\n"):
        attrs.add(line)
with open("./test/input.attr") as f:
    for line in f.read().strip().split("\n"):
        attrs.add(line)
attrs = sorted(attrs)
print("attrs:", len(attrs))

with open("./vocab_attr.txt", "w") as w:
    for attr in attrs:
        w.write(attr + "\n")
        
# make vocab_value

values = set()
with open("./train/output.value") as f:
    for line in f.read().strip().split("\n"):
        values.add(line)
with open("./test/output.value") as f:
    for line in f.read().strip().split("\n"):
        values.add(line)
print("values:", len(values))

with open("./vocab_value.txt", "w") as w:
    for value in values:
        w.write(value + "\n")