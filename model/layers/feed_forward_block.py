import tensorflow as tf
from tensorflow.keras import Sequential
from typing import Optional
from tensorflow.keras.layers import Add, Layer, LayerNormalization, Dropout, Dense
from tensorflow.keras.regularizers import l2

class FeedForwardBlock(Layer):
    def __init__(self, dim_ff: int, dim_model: int, scope_name: str, drop_out: float = 0.0, **kwargs):
        super().__init__()

        self.scope_name = scope_name
        self.seq = Sequential([
            Dense(units=dim_ff, activation='relu', kernel_regularizer=l2(0.01)), 
            Dense(units=dim_model, kernel_regularizer=l2(0.01)), 
            Dropout(rate=drop_out)
        ], name=f'{scope_name}_sequential')

        self.add = Add()
        self.layernorm = LayerNormalization()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        with tf.name_scope(self.scope_name):
            x = self.add([x, self.seq(x)])
            return self.layernorm(x)