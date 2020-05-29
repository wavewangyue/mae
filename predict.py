import tensorflow as tf
import numpy as np
import json

from utils import DataProcessor
from utils import VectorContainer, VectorContainer_ZERO
from utils import load_vocabulary

img_embedding_size = 4096

paths = {
    "ckpt": "./ckpt/mae.ckpt.batch2500",
    "test_data": "./data/test",
    "vocab_word": "./data/vocab_word.txt",
    "vocab_attr": "./data/vocab_attr.txt",
    "vocab_value": "./data/vocab_value.txt",
    "image_vector": "./data/image_fc_vectors.npy"
}

use_image = False

print("load data...")

w2i_word, i2w_word = load_vocabulary(paths["vocab_word"])
w2i_attr, i2w_attr = load_vocabulary(paths["vocab_attr"])
w2i_value, i2w_value = load_vocabulary(paths["vocab_value"])

data_processor = DataProcessor(
    paths["test_data"] + "/input.seq",
    paths["test_data"] + "/input.imageindex",
    paths["test_data"] + "/input.attr",
    paths["test_data"] + "/output.value",
    w2i_word,
    w2i_attr, 
    w2i_value, 
    shuffling=False
)

if use_image:
    image_vector_container = VectorContainer(paths["image_vector"], img_embedding_size)
else:
    image_vector_container = VectorContainer_ZERO(img_embedding_size)

print("loading checkpoint from", paths["ckpt"], "...")
    
tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)

saver = tf.train.import_meta_graph(paths["ckpt"] + ".meta")
saver.restore(sess, paths["ckpt"])
graph = tf.get_default_graph()

# for n in graph.as_graph_def().node:
#     t = n.name
#     if (not t.startswith("opt/")) and (not t.startswith("save/")):
#         print(t)

print("predicting...")

golds = []
preds = []

while not data_processor.end_flag:
    (inputs_seq_batch, 
     inputs_seq_len_batch,
     inputs_image_batch,
     inputs_attr_batch,
     outputs_value_batch) = data_processor.get_batch(512, image_vector_container)

    feed_dict = {
        graph.get_tensor_by_name('inputs_seq:0'): inputs_seq_batch,
        graph.get_tensor_by_name('inputs_seq_len:0'): inputs_seq_len_batch,
        graph.get_tensor_by_name('inputs_image:0'): inputs_image_batch,
        graph.get_tensor_by_name('inputs_attr:0'): inputs_attr_batch
    }

    preds_value_batch = sess.run(graph.get_tensor_by_name("outputs:0"), feed_dict)
    preds.extend(preds_value_batch.tolist())
    golds.extend(outputs_value_batch.tolist())

hits = 0
for gold, pred in zip(golds, preds):
    if gold == pred:
        hits += 1
acc = round(hits*100/len(preds), 2)
print("acc: {}({}/{})".format(acc, hits, len(preds)))


