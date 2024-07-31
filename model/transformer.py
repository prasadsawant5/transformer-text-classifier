from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
from model.encoder import Encoder
from config import N_CLASSES

class Transformer(Model):
    def __init__(self, encoder_layers: int, dim_model: int, dim_ff: int, num_heads: int, vocab_size: int, maximum_position_encoding: int, 
                dropout: float, **kwargs):
        super().__init__(**kwargs)

        self.encoder = Encoder(
            num_layers=encoder_layers, 
            dim_model=dim_model, 
            dim_ff=dim_ff, 
            vocab_size=vocab_size, 
            max_position_encoding=maximum_position_encoding, 
            dropout=dropout, 
            num_heads=num_heads
        )

        self.pooling = GlobalAveragePooling1D(name='global_avg_pooling')
        self.dropout0 = Dropout(dropout, name='dropout0')
        # self.dropout1 = Dropout(dropout, name='dropout1')
        self.dense = Dense(20, activation='relu', name='dense0', kernel_regularizer=l2(0.01))

        self.outputs = Dense(units=N_CLASSES, activation='softmax', name='outputs')

    def call(self, inputs):
        encoder_output = self.encoder(x=inputs)
        encoder_output = self.pooling(encoder_output)
        encoder_output = self.dropout0(encoder_output)
        encoder_output = self.dense(encoder_output)
        # encoder_output = self.dropout1(encoder_output)

        # apply a dense layer to the deocder output to formulate the logits
        logits = self.outputs(encoder_output)

        # drop the keras mask, so it doesn't scale the losses/metrics.
        try:
            del logits._keras_mask
        except AttributeError:
            pass

        # return the final logits
        return logits