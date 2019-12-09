import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

# Workaround numpy > 1.16.1 incompatible with keras data loading.
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz")

x_train = pad_sequences(x_train, padding="post")
maxlen = x_train.shape[1]
x_test = pad_sequences(x_test, padding="post", maxlen=maxlen)
vocab_size = x_train.max() + 1  # +1 for padding index at 0.

model_file = "models/rnn.h5"

embedding_size = 64
model = keras.Sequential([
  keras.layers.Embedding(
    vocab_size, embedding_size, input_length=maxlen,
    mask_zero=False,  # Masking is not support by plaidml.
    name="word_embedding"),
  keras.layers.GRU(64, dropout=.2, name="GRU"),
  keras.layers.Dense(1, activation="sigmoid", name="sigmoid")
  ], name="rnn_classifier"
)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

metrics = model.fit(
  x=x_train, y=y_train,
  batch_size=32, epochs=10,
  validation_data=(x_test, y_test),
  callbacks=[
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
    keras.callbacks.ModelCheckpoint(model_file, monitor="val_loss", save_best_only=True)
  ],
  verbose=1)

yhat = np.squeeze(model.predict(x_test))
pred = (yhat > .5).astype(int)
acc = (pred == y_test).sum() / len(y_test)
print(acc)
