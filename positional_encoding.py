import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import string, convert_to_tensor
from keras.layers import TextVectorization, Layer, Embedding

# output_sequence_length = 5
# vocab_size = 10
# sentences = [
#     ["I am a robot"], 
#     ["you too robot"]
# ]
# sentence_data = tf.data.Dataset.from_tensor_slices(sentences)

# vectorize_layer = TextVectorization(output_sequence_length = output_sequence_length, max_tokens = vocab_size)
# vectorize_layer.adapt(sentence_data)

# word_tensors = convert_to_tensor(sentences, dtype = tf.string)

# vectorize_words = vectorize_layer(word_tensors)

# # print("Vocabulary:", vectorize_layer.get_vocabulary())

# # print("Vocabulary_words: ", vectorize_words)

# output_length = 6
# word_embedding_layer = Embedding(vocab_size, output_length)
# # print(word_embedding_layer)

# embedded_words = word_embedding_layer(vectorize_words)
# # print(embedded_words)

# position_embedding_layer = Embedding(output_sequence_length, output_length)
# position_indices = tf.range(output_sequence_length)

# embedded_indices = position_embedding_layer(position_indices)

# print(embedded_indices)

# final_output_embedding = embedded_words + embedded_indices
# print('Final_output: ', final_output_embedding)


class PositionEmbeddingLayer(Layer):
    def __init__(self, output_sequence_length, vocab_size, output_length, **kwargs):
        super(PositionEmbeddingLayer, self).__init__(**kwargs)
        self.word_embedding_layer = Embedding(vocab_size, output_length)
        self.position_embedding_layer = Embedding(output_sequence_length, output_length)

    def call(self, inputs):
        position_indices = tf.range(tf.shape(inputs)[-1])

        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices = self.position_embedding_layer(position_indices)

        return embedded_words + embedded_indices
    
    def positional_encoding(self, seq_len, d, n = 10000):
        p = np.zeroes((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d / 2)):
                denom = np.power(n, 2 * i / d)
                p[k, 2 * i] = np.sin(k / denom)
                p[k, 2 * i + 1] = np.cos(k / denom)
        return p






