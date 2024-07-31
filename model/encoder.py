import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout
from tensorflow.keras import Sequential
from model.layers.feed_forward_block import FeedForwardBlock
from model.layers.attention import CausalSelfAttention
from model.layers.positional_encoding import PositionalEmbedding

class Encoder(Layer):
    def __init__(self, num_layers: int, dim_model: int, dim_ff: int, vocab_size: int, max_position_encoding: int, dropout: float = 0.0, 
                 num_heads: int = 4, scope_name: str = 'encoder'):
        super().__init__()
        self.positional_embedding = PositionalEmbedding(
            vocabSize=vocab_size,
            dModel=dim_model,
            maximumPositionEncoding=max_position_encoding,
        )
        self.dropout = Dropout(rate=dropout)

        self.encoder_layers = [
            Sequential([
                CausalSelfAttention(scope_name=f'causal_self_attention{i}', num_heads=num_heads, key_dim=dim_model // num_heads, dropout=dropout), 
                FeedForwardBlock(dim_ff=dim_ff, dim_model=dim_model, scope_name=f'ffn{i}', drop_out=dropout)
            ])
            for i in range(num_layers)
        ]
        self.scope_name = scope_name

    def call(self, x):
        with tf.name_scope(self.scope_name):
            # apply positional embedding to the target token ids
            x = self.positional_embedding(x)

            # apply dropout to the embedded targets
            x = self.dropout(x)

            # iterate over the stacks of decoder layer
            for encoder_layer in self.encoder_layers:
                x = encoder_layer(x)

            # get the attention scores and cache it
            # self.lastAttentionScores = self.decoder_layers[-1].last_attn_scores

            return x