import os
import json

item_num_all = 0
triple_num_all = 0
    
for fin_dir, fout_dir in [
    ("./origin/train_top100attr", "./train"),
    ("./origin/test_top100attr", "./test"),
]:
    
    seqs_cls = []
    attrs = []
    values = []
    
    fnames = os.listdir(fin_dir)
    for i, fname in enumerate(fnames):
        fin = fin_dir + "/" + fname
        with open(fin) as f:
            infos = json.load(f)
        if i % 100 == 0: print("{}/{}: load {} items from {}, processing...".format(i, len(fnames), len(infos), fin))
        
        for ii, info in enumerate(infos):
            item_num_all += 1
            words = [word.lower() for word in info["tokens"][:150]]
            doc = " ".join(words)
            bios = ["O"] * len(words)
            attr_group = set()
            
            for attr, value in info["specs"].items():
                triple_num_all += 1
                attr = attr.replace(" ", "_")
                value = value.lower()
                seqs_cls.append(doc)
                attrs.append(attr)
                values.append(value)

    print("writing data")
    
    with open(fout_dir + "/input.seq", "w") as w:
        print("seqs_cls:", len(seqs_cls))
        for seq in seqs_cls:
            w.write(seq + "\n")
    
    with open(fout_dir + "/input.attr", "w") as w:
        print("attrs:", len(attrs))
        for attr in attrs:
            w.write(attr + "\n")

    with open(fout_dir + "/output.value", "w") as w:
        print("values:", len(values))
        for value in values:
            w.write(value + "\n")
            
print("items:", item_num_all)
print("triples:", triple_num_all)