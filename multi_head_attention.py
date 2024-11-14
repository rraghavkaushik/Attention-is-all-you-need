import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from keras.backend import softmax
from numpy import random

class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)

    def calc(self, queries, keys, values, dk, mask = None):
        
        key = np.matrix(key)

        scores = (queries @ keys.transpose()) / np.sqrt(float(dk))

        if mask is not None:
            scores += -1e9 * mask

        weights = softmax(scores)

        return weights @ values


class MultiheadAttention(Layer):
    def __init__(self, h, dk, dv, d_o, **kwargs):
        super(MultiheadAttention, self).__init__(**kwargs)

        self.attention = DotProductAttention()
        self.heads = h
        self.dk = dk
        self.dv = dv
        self.d_o = d_o
        self.W_Q = Dense(dk)
        self.W_K = Dense(dk)
        self.W_V = Dense(dv)
        self.W_O = Dense(d_o)
    

    def reshape_tensor(self, x, heads, flag):
        if flag:
            x = np.reshape(x, (x.shape[0], x.shape[1], heads, -1))
            x = np.transpose(x, (0, 2, 1, 3))
        else:
            x = np.transpose(x, (0, 2, 1, 3))
            x = np.reshape(x, (x.shape[0], x.shape[1], -1))
        return x
    
    def cal(self, queries, keys, values, mask = None):

        q_reshaped = self.reshape_tensor(self.W_Q(queries), self.heads, True)

        k_reshaped = self.reshape_tensor(self.W_K(keys), self.heads, True)

        v_reshaped = self.reshape_tensor(self.W_K(values), self.heads, True)

        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.dk, mask)

        output = self.reshape_tensor(o_reshaped, self.heads, False)

        return self.W_o(output)
    

#testing out the code

'''
    The below parameters are from the paper
'''


h = 8 
dk = 64  
dv = 64 
d_o = 512
batch_size = 64  
input_seq_length = 5

queries = random.random((batch_size, input_seq_length, dk))
keys = random.random((batch_size, input_seq_length, dk))
values = random.random((batch_size, input_seq_length, dv))
 
multihead_attention = MultiheadAttention(h, dk, dv, d_o)
print(multihead_attention(queries, keys, values))