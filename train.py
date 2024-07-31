import os
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam

from model.transformer import Transformer
from custom_schedule import CustomSchedule
from dataset import get_dataset, get_sentence_from_vec, get_label_from_one_hot
from config import (
    BATCH_SIZE, DIM_FF, DIM_MODEL, DROPOUT, EPOCHS, LOGS, LR, NUM_HEADS, NUM_LAYERS, PATHS, SAVED_MODELS, VOCAB_SIZE, MAX_LEN
)

def make_batches(ds):
    return (
        ds.shuffle(10000)
            .batch(BATCH_SIZE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

# Define the weighted categorical cross-entropy loss function
def weighted_categorical_crossentropy(weights):
    def loss(y_true, y_pred):
        # Scale predictions so that the class probabilities of each sample sum to 1
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        # Clip the predictions to prevent log(0) errors
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        # Calculate the cross-entropy loss
        loss = y_true * tf.math.log(y_pred)
        # Scale the loss by the class weights
        loss = loss * weights
        # Return the mean loss over all samples
        return -tf.reduce_mean(loss, axis=-1)
    return loss

if not os.path.exists(LOGS):
    os.mkdir(LOGS)

if not os.path.exists(SAVED_MODELS):
    os.mkdir(SAVED_MODELS)

for path in PATHS:
    if not os.path.exists(path):
        print('Dataset does not exists')
        sys.exit(-1)

features, labels, weights = get_dataset()
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

train_ds = make_batches(train_ds)
val_ds = make_batches(val_ds)

early_stopping = EarlyStopping(
    monitor='val_loss',     # Metric to monitor
    patience=4,             # Number of epochs with no improvement after which training will be stopped
    verbose=1,              # Verbosity mode (0 or 1)
    mode='min',             # 'min' for metrics that should decrease (e.g., loss), 'max' for metrics that should increase (e.g., accuracy)
    restore_best_weights=True  # Whether to restore model weights from the epoch with the best value of the monitored quantity
)

callbacks = [
    TensorBoard(LOGS), early_stopping
]

inputs = Input(shape=(MAX_LEN, ), name='inputs')

transformer = Transformer(
    encoder_layers=NUM_LAYERS, 
    dim_model=DIM_MODEL,
    dim_ff=DIM_FF, 
    dropout=DROPOUT,
    num_heads=NUM_HEADS, 
    vocab_size=VOCAB_SIZE,
    maximum_position_encoding=MAX_LEN
)

outputs = transformer(inputs)
model = Model(inputs=inputs, outputs=outputs, name='text_classifier') 

# transformer.summary()
optimizer = Adam(learning_rate=LR, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
# learning_rate = CustomSchedule(dModel=DIM_MODEL)

# Create a list of class weights corresponding to the order of the classes
weights_array = np.array([weights[i] for i in range(len(weights))])

# Create the weighted loss function
# weighted_loss = weighted_categorical_crossentropy(weights_array)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    train_ds, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks
)

model.save(SAVED_MODELS)

predictions = model.predict(X_test)

data = []
labels = []
preds = []
for x_test, y_label, pred in zip(X_test, y_test, predictions):
    ground_truth = get_label_from_one_hot(y_label)
    pred = get_label_from_one_hot(pred)
    sentence = get_sentence_from_vec(x_test)

    data.append([sentence, ground_truth, pred])
    labels.append(ground_truth)
    preds.append(pred)

df = pd.DataFrame(data=data, columns=['sentence', 'ground_truth', 'prediction'])
df.to_csv('inf.csv')

# Calculate accuracy
accuracy = accuracy_score(labels, preds)

# Print the accuracy
print(f"Accuracy: {accuracy * 100:.2f}%")
