import random
import numpy as np

##########################
####### Vocabulary #######
##########################
            
def load_vocabulary(path):
    vocab = open(path, "r", encoding="utf-8").read().strip().split("\n")
    print("load vocab from: {}, containing words: {}".format(path, len(vocab)))
    w2i = {}
    i2w = {}
    for i, w in enumerate(vocab):
        w2i[w] = i
        i2w[i] = w
    return w2i, i2w

##############################
####### DataProcessor ########
##############################

class DataProcessor(object):
    def __init__(self, 
                 input_seq_path, 
                 input_image_index_path,
                 input_attr_path,
                 output_value_path,
                 w2i_word, 
                 w2i_attr, 
                 w2i_value, 
                 shuffling=False):
        
        inputs_seq = []
        with open(input_seq_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                seq = []
                for word in line.split(" "):
                    seq.append(w2i_word[word] if word in w2i_word else w2i_word["[UNK]"])
                inputs_seq.append(seq)
        
        inputs_image_index = []
        with open(input_image_index_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                inputs_image_index.append([int(index) for index in line.split(" ")])
        
        inputs_attr = []
        with open(input_attr_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                inputs_attr.append(w2i_attr[line])
        
        outputs_value = []
        with open(output_value_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                outputs_value.append(w2i_value[line])
    
        assert len(inputs_seq) == len(inputs_image_index)
        assert len(inputs_seq) == len(inputs_attr)
        assert len(inputs_seq) == len(outputs_value)
        
        self.w2i_word = w2i_word
        self.inputs_seq = inputs_seq
        self.inputs_image_index = inputs_image_index
        self.inputs_attr = inputs_attr
        self.outputs_value = outputs_value
        self.ps = list(range(len(inputs_seq)))
        self.shuffling = shuffling
        if shuffling: random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False
        print("DataProcessor load data num: " + str(len(inputs_seq)), "shuffling:", shuffling)
        
    def refresh(self):
        if self.shuffling: random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False
    
    def get_batch(self, batch_size, image_vector_container=None):
        inputs_seq_batch = []
        inputs_seq_len_batch = []
        inputs_image_batch = []
        inputs_attr_batch = []
        outputs_value_batch = []
        
        while (len(inputs_seq_batch) < batch_size) and (not self.end_flag):
            p = self.ps[self.pointer]
            inputs_seq_batch.append(self.inputs_seq[p].copy())
            inputs_seq_len_batch.append(len(self.inputs_seq[p]))
            inputs_attr_batch.append(self.inputs_attr[p].copy())
            outputs_value_batch.append(self.outputs_value[p].copy())
            inputs_image_batch.append(image_vector_container.get_vectors(self.inputs_image_index[p], padding_num=3))
            self.pointer += 1
            if self.pointer >= len(self.ps): self.end_flag = True
        
        max_seq_len = max(inputs_seq_len_batch)
        for seq in inputs_seq_batch:
            seq.extend([self.w2i_word["[PAD]"]] * (max_seq_len - len(seq)))
    
        return (np.array(inputs_seq_batch, dtype="int32"),
                np.array(inputs_seq_len_batch, dtype="int32"),
                np.array(inputs_image_batch, dtype="float32"),
                np.array(inputs_attr_batch, dtype="int32"),
                np.array(outputs_value_batch, dtype="int32"))
    
############################
####### VectorContrainer #######
############################

class VectorContainer(object):
    def __init__(self,image_vector_path, vector_dim):
        self.vectors = np.load(image_vector_path) # N * D
        self.vector_dim = vector_dim
        assert self.vector_dim == self.vectors.shape[1]
        print("VectorContrainer load vectors from", image_vector_path, "shape:", self.vectors.shape)
    
    def get_vectors(self, indexes, padding_num):
        vecs = self.vectors[indexes]
        if vecs.shape[0] > padding_num:
            vecs = vecs[:padding_num, :]
        if vecs.shape[0] < padding_num:
            vecs = np.concatenate((vecs, np.zeros((padding_num-vecs.shape[0], self.vector_dim))), axis=0)
        return vecs

class VectorContainer_ZERO(object):
    def __init__(self, vector_dim):
        self.vector_dim = vector_dim
    
    def get_vectors(self, indexes, padding_num):
        return np.zeros((padding_num,  self.vector_dim))
