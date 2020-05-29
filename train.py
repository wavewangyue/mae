import logging
import tensorflow as tf
import numpy as np

from model import MAEModel
from utils import DataProcessor
from utils import VectorContainer, VectorContainer_ZERO
from utils import load_vocabulary

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

config = {
    "txt_embedding_size": 200,
    "txt_hidden_size": 600,
    "img_embedding_size": 4096,
    "img_hidden_size": 1024,
    "attr_embedding_size": 200,
    "fusion_hidden_size": 200,
    "batch_size": 256
}

paths = {
    "ckpt": "./ckpt/mae.ckpt",
    "train_data": "./data/train",
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

data_processor_train = DataProcessor(
    paths["train_data"] + "/input.seq",
    paths["train_data"] + "/input.imageindex",
    paths["train_data"] + "/input.attr",
    paths["train_data"] + "/output.value",
    w2i_word,
    w2i_attr, 
    w2i_value, 
    shuffling=True
)

data_processor_test = DataProcessor(
    paths["test_data"] + "/input.seq",
    paths["test_data"] + "/input.imageindex",
    paths["test_data"] + "/input.attr",
    paths["test_data"] + "/output.value",
    w2i_word,
    w2i_attr, 
    w2i_value, 
    shuffling=True
)

if use_image:
    image_vector_container = VectorContainer(paths["image_vector"], config["img_embedding_size"])
else:
    image_vector_container = VectorContainer_ZERO(config["img_embedding_size"])

print("build model...")

model = MAEModel(
    config["txt_embedding_size"],
    config["txt_hidden_size"],
    config["img_embedding_size"],
    config["img_hidden_size"],
    config["attr_embedding_size"],
    config["fusion_hidden_size"],
    len(w2i_word),
    len(w2i_attr),
    len(w2i_value)
)

print("start training...")

saver = saver = tf.train.Saver(max_to_keep=500)
tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True

with tf.Session(config=tf_config) as sess:
    sess.run(tf.global_variables_initializer())
    
    epoches = 0
    losses = []
    batches = 0
    best_acc = 0

    while epoches < 5:
        
        (inputs_seq_batch,
         inputs_seq_len_batch,
         inputs_image_batch,
         inputs_attr_batch,
         outputs_value_batch) = data_processor_train.get_batch(config["batch_size"], image_vector_container)
		
        feed_dict = {
            model.inputs_seq: inputs_seq_batch,
            model.inputs_seq_len: inputs_seq_len_batch,
            model.inputs_image: inputs_image_batch,
            model.inputs_attr: inputs_attr_batch,
            model.outputs_value: outputs_value_batch
        }
        
        if batches == 0: 
            print("###### shape of a batch #######")
            print("input_seq:", inputs_seq_batch.shape)
            print("input_seq_len:", inputs_seq_len_batch.shape)
            print("input_attr:", inputs_attr_batch.shape)
            print("output_value:", outputs_value_batch.shape)
            print("###### preview a sample #######")
            print("input_seq:", " ".join([i2w_word[i] for i in inputs_seq_batch[0]]))
            print("input_seq_len:", inputs_seq_len_batch[0])
            print("input_attr:", i2w_attr[inputs_attr_batch[0]])
            print("output_value:", i2w_value[outputs_value_batch[0]])
            print("###############################")
        
        loss, _ = sess.run([model.loss, model.train_op], feed_dict)
        losses.append(loss)
        batches += 1
        
        if data_processor_train.end_flag:
            data_processor_train.refresh()
            epoches += 1

        def valid(data_processor, batch_size=1024, max_batches=None):
            pred_num = 0
            hits = 0
            preds = []
            while True:
                (inputs_seq_batch, 
                 inputs_seq_len_batch,
                 inputs_image_batch,
                 inputs_attr_batch,
                 outputs_value_batch) = data_processor.get_batch(config["batch_size"], image_vector_container)

                feed_dict = {
                    model.inputs_seq: inputs_seq_batch,
                    model.inputs_seq_len: inputs_seq_len_batch,
                    model.inputs_image: inputs_image_batch,
                    model.inputs_attr: inputs_attr_batch,
                    model.outputs_value: outputs_value_batch
                }

                preds_value_batch = sess.run(model.outputs, feed_dict)

                pred_num += preds_value_batch.shape[0]
                hits += (preds_value_batch == outputs_value_batch).astype(int).sum()

                if data_processor.end_flag:
                    data_processor.refresh()
                    break
                
                if (max_batches is not None) and (pred_num >= max_batches * batch_size):
                    break

            acc = round(hits*100/pred_num, 2)
            logging.info("valid acc: {} ({}/{})".format(acc, hits, pred_num))

            return acc
            
        if batches % 100 == 0:
            logging.info("")
            logging.info("Epoches: {}".format(epoches))
            logging.info("Batches: {}".format(batches))
            logging.info("Loss: {}".format(sum(losses) / len(losses)))
            losses = []

            ckpt_save_path = paths["ckpt"] + ".batch{}".format(batches)
            logging.info("Path of ckpt: {}".format(ckpt_save_path))
            saver.save(sess, ckpt_save_path)
            
            acc = valid(data_processor_test, max_batches=100)
            if acc > best_acc:
                logging.info("############# best performance now : {} ###############".format(acc))
                best_acc = acc
                
#             print("train dataset pointer:", data_processor_train.pointer)
#             print("test dataset pointer:", data_processor_test.pointer)
