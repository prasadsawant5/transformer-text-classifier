import tensorflow as tf
from tensorflow.keras.layers import Add, Layer, LayerNormalization, MultiHeadAttention

class BaseAttention(Layer):
    def __init__(self, scope_name: str, **kwargs):
        super().__init__()

        self.mha = MultiHeadAttention(name=f'{scope_name}_mha', **kwargs)
        self.layernorm = LayerNormalization(name=f'{scope_name}_layernorm')
        self.add = Add(name=f'{scope_name}_add')

        self.scope_name = scope_name

class CrossAttention(BaseAttention):
    def call(self, x, context):
        with tf.name_scope(self.scope_name):
            # apply multihead attention to the query and the context inputs
            (attentionOutputs, attentionScores) = self.mha(
                query=x,
                key=context,
                value=context,
                return_attention_scores=True,
            )

            # store the attention scores that will be later visualized
            self.lastAttentionScores = attentionScores

            # apply residula connection and layer norm
            x = self.add([x, attentionOutputs])
            x = self.layernorm(x)

            # return the processed query
            return x


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        with tf.name_scope(self.scope_name):
            # apply self multihead attention
            attentionOutputs = self.mha(
                query=x,
                key=x,
                value=x,
            )

            # apply residual connection and layer norm
            x = self.add([x, attentionOutputs])
            x = self.layernorm(x)

            # return the processed query
            return x


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        with tf.name_scope(self.scope_name):
            # apply self multi head attention with causal masking (look-ahead-mask)
            attentionOutputs, attn_scores = self.mha(
                query=x,
                key=x,
                value=x,
                use_causal_mask=True, 
                return_attention_scores=True
            )

            # Cache the attention scores for plotting later.
            self.last_attn_scores = attn_scores

            # apply residual connection and layer norm
            x = self.add([x, attentionOutputs])
            x = self.layernorm(x)

            # return the processed query
            return x