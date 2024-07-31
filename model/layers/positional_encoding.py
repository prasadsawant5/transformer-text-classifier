import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Layer

def positional_encoding(length, depth):
    """
    Function to build the positional encoding as per the
    "Attention is all you need" paper.

    Args:
        length: The length of each sentence (target or source)
        depth: The depth of each token embedding
    """
    # divide the depth of the positional encoding into two for
    # sinusoidal and cosine embeddings
    depth = depth / 2

    # define the positions and depths as numpy arrays
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth

    # build the angle rates and radians
    angleRates = 1 / (10000**depths)
    angleRads = positions * angleRates

    # build the positional encoding, cast it to float32 and return it
    posEncoding = np.concatenate([np.sin(angleRads), np.cos(angleRads)], axis=-1)
    return tf.cast(posEncoding, dtype=tf.float32)


class PositionalEmbedding(Layer):
    def __init__(self, vocabSize, dModel, maximumPositionEncoding, **kwargs):
        """
        Args:
            vocabSize: The vocabulary size of the target or source dataset
            dModel: The dimension of the transformer model
            maximumPositionEncoding: The maximum length of a sentence in the dataset
        """
        super().__init__(**kwargs)

        # initialize an embedding layer
        self.embedding = Embedding(
            input_dim=vocabSize, output_dim=dModel, mask_zero=True, name='embedding'
        )

        # initialize the positional encoding function
        self.posEncoding = positional_encoding(
            length=maximumPositionEncoding, depth=dModel
        )

        # define the dimensions of the model
        self.dModel = dModel

    def compute_mask(self, *args, **kwargs):
        # return the padding mask from the inputs
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        # get the length of the input sequence
        seqLen = tf.shape(x)[1]

        # embed the input and scale the embeddings
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dModel, tf.float32))

        # add the positional encoding with the scaled embeddings
        x += self.posEncoding[tf.newaxis, :seqLen, :]

        # return the encoded input
        return x