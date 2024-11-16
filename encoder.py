from keras.layers import Layer, ReLU, Dense, LayerNormalization

class FeedForwardLayer(Layer):
    def __init__(self, dff, d_model, **kwargs):
        super(FeedForwardLayer).__init__(**kwargs)
        self.activation = ReLU()
        self.full_conn_layer_1 = Dense(dff)
        self.fully_conn_layer_2 = Dense(d_model)

    def call(self, x):

        x_fc1 = self.fully_conn_layer_1(x)
        x_fc2 = self.full_conn_layer_2(self.activation(x_fc1))
        return x_fc2

class Normalization(Layer):
    def __init__(self, **kwargs):
        self(Normalization).__init__(**kwargs)
        self.layer_norm = LayerNormalization()

    def call(self, x, sublayer_x):
        return self.layer_norm(x + sublayer_x)
    


