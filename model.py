import tensorflow as tf

class MAEModel(object):
    
    def __init__(self, 
                 txt_embedding_size,
                 txt_hidden_size,
                 img_embedding_size,
                 img_hidden_size,
                 attr_embedding_size,
                 fusion_hidden_size,
                 vocab_size_word, 
                 vocab_size_attr,
                 vocab_size_value,
                 fusion_mode="concat"):
        self.inputs_seq = tf.placeholder(tf.int32, [None, None], name="inputs_seq")
        self.inputs_seq_len = tf.placeholder(tf.int32, [None], name="inputs_seq_len")
        self.inputs_image = tf.placeholder(tf.float32, [None, 3, img_embedding_size], name="inputs_image")
        self.inputs_attr = tf.placeholder(tf.int32, [None], name="inputs_attr")
        self.outputs_value = tf.placeholder(tf.int32, [None], name='outputs_value')
        
        with tf.variable_scope('word_embedding'):
            embedding_matrix = tf.get_variable("embedding_matrix", [vocab_size_word, txt_embedding_size], dtype=tf.float32)
            txt_embedded = tf.nn.embedding_lookup(embedding_matrix, self.inputs_seq) # B * S * D
            
        with tf.variable_scope('txt_encoder'):
            conv = tf.layers.conv1d(txt_embedded,
                                filters=txt_hidden_size,
                                kernel_size=5,
                                strides=1,
                                padding="valid") # B * (S-5+1) * D
            pool = tf.reduce_max(conv, axis=1)
            pool_drop = tf.layers.dropout(pool, 0.5)
            txt_contents = tf.layers.dense(pool_drop, txt_hidden_size, activation="relu") # B * D
            
        with tf.variable_scope('img_encoder'):
            img_input = tf.layers.dropout(self.inputs_image, 0.5)
            img_embedded = tf.layers.dense(img_input, img_hidden_size, activation="relu") # B * 3 * D
            img_contents = tf.reduce_max(img_embedded, axis=1) # B * D
        
        with tf.variable_scope('attr_encoder'):
            embedding_matrix = tf.get_variable("embedding_matrix", [vocab_size_attr, attr_embedding_size], dtype=tf.float32)
            attr_contents = tf.nn.embedding_lookup(embedding_matrix, self.inputs_attr) # B * D
        
        with tf.variable_scope('fusion_layer'):
            if fusion_mode == "concat":
                fusion_all = tf.concat([txt_contents, img_contents, attr_contents], axis=1)
                fusion_contents = tf.layers.dense(fusion_all, fusion_hidden_size) # B * D
            elif fusion_mode == "gmu":
                fusion_ta = tf.concat([txt_contents, attr_contents], axis=1)
                fusion_contents_ta = tf.layers.dense(fusion_ta, fusion_hidden_size)  # B * D
                fusion_ia = tf.concat([img_contents, attr_contents], axis=1)
                fusion_contents_ia = tf.layers.dense(fusion_ia, fusion_hidden_size)  # B * D
                fusion_ti = tf.concat([txt_contents, img_contents], axis=1)
                gate = tf.nn.sigmoid(tf.layers.dense(fusion_ti, 1)) # B * 1
                fusion_contents = gate * fusion_contents_ta + (1 - gate) * fusion_contents_ia  # B * D
            else:
                print("wrong fusion mode:", fusion_mode, "(only support ['concat', 'gmu'])")
        
        distribute_matrix = tf.get_variable("distribute_matrix", [fusion_hidden_size, vocab_size_value], dtype=tf.float32) # D * V

        # loss by cos_similarity
        
        dot_part1 = tf.expand_dims(fusion_contents, axis=-1) # B * D * 1
        dot_part2 = tf.expand_dims(distribute_matrix, axis=0) # 1 * D * V
        dot = tf.reduce_sum(dot_part1 * dot_part2, axis=1) # B * V
        norm_part1 = tf.expand_dims(tf.sqrt(tf.reduce_sum(fusion_contents * fusion_contents, axis=1)), axis=-1) # B * 1
        norm_part2 = tf.expand_dims(tf.sqrt(tf.reduce_sum(distribute_matrix * distribute_matrix, axis=0)), axis=0) # 1 * V
        norm = norm_part1 * norm_part2 # B * V
        norm = tf.clip_by_value(norm, clip_value_min=1e-5, clip_value_max=1e5)
        sims = dot / norm # B * V
        outputs = tf.argmax(sims, 1, name="outputs") # B
        self.outputs = outputs
        
        pos_value_onehot = tf.one_hot(self.outputs_value, vocab_size_value) # B * V
        loss_pos = sims * pos_value_onehot # B * V
        loss_pos = tf.square(1 - tf.reduce_sum(loss_pos, axis=1)) # B
        loss_neg = sims * (1 - pos_value_onehot) # B * V
        loss_neg = tf.clip_by_value(loss_neg, clip_value_min=0, clip_value_max=1) # B * V
        loss_neg = tf.reduce_sum(tf.square(loss_neg), axis=1) # B
        loss = loss_pos + loss_neg
        loss = tf.reduce_mean(loss)
        
        self.loss = loss

        # loss by cross_entropy
        
#         logits = tf.matmul(fusion_contents, distribute_matrix) # B * V
#         self.logits = logits
#         probs = tf.nn.softmax(logits, name="probs") # B * V
#         self.probs = probs
#         outputs = tf.argmax(probs, 1, name="outputs") # B
#         self.outputs = outputs
          
#         loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.outputs_value)
#         loss = tf.reduce_mean(loss)
#         self.loss = loss
        
        with tf.variable_scope('opt'):
            self.train_op = tf.train.AdamOptimizer().minimize(loss)
        
        print("model params:")
        params_num_all = 0
        for variable in tf.trainable_variables():
            params_num = 1
            for dim in variable.shape:
                params_num *= dim
            print("\t {} {}".format(variable.name, variable.shape))
            params_num_all += params_num
        print("all params num:", params_num_all)