from keras.layers import Layer, ReLU, Dense, LayerNormalization, Dropout
from multi_head_attention import MultiheadAttention
from positional_encoding import PositionEmbeddingLayer

class FeedForwardLayer(Layer):
    def __init__(self, dff, dmodel, **kwargs):
        super(FeedForwardLayer).__init__(**kwargs)
        self.activation = ReLU()
        self.fully_conn_layer1 = Dense(dff)
        self.fully_conn_layer2 = Dense(dmodel)

    def call(self, x):
        x_fc1 = self.fully_conn_layer1(x)
        x_fc2 = self.fully_conn_layer2(ReLU(x_fc1))
        return x_fc2
    

class AddNorm(Layer):
    def __init__(self, **kwargs):
        self(AddNorm).__init__(**kwargs)
        self.layer_norm = LayerNormalization()

    def call(self, x, sublayer_x):
        return self.layer_norm(x + sublayer_x)
    

class EncodingLayer(Layer):
    def __init__(self, h, dk, dv, dmodel, rate, **kwargs):

        ''' rate - frequency at which the input units are set to 0'''
        super(EncodingLayer).__init__(**kwargs)
        self.multihead_attention = MultiheadAttention(h, dk, dv, dmodel)
        self.dropout1 = Dropout(rate)
        self.addnorm1 = AddNorm()
        self.feed_forward_layer = FeedForwardLayer()
        self.dropout2 = Dropout(rate)
        self.addnorm2 = AddNorm()
    
    def call(self, x, mask, train):
        multi_head = self.multihead_attention(x, x, x, mask)
        multi_head = self.dropout1(multi_head, train = train)

        addnorm1 = self.addnorm1(x, multi_head)

        feedforw_layer = FeedForwardLayer(addnorm1)
        feedforw_layer = self.drpout(feedforw_layer, train = train)

        addnorm2 = self.addnorm2(addnorm1, feedforw_layer)

        return addnorm2

        

