"""Use pre-trained BERT from tensorflow-hub for moive review sentiment classification."""

import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tokenization import FullTokenizer
# https://github.com/tensorflow/models/blob/master/official/nlp/bert/tokenization.py


# Load the data as tf.data.Dataset.
imdb = tfds.load(name="imdb_reviews", as_supervised=True)

# Extract all texts as list since we want to use libraries other than tensorflow as well.
# And since this is a small dataset, we don't care about memory usage.
imdb_reviews_train = []
imdb_reviews_test = []
y_train = []
y_test = []
for x, y in imdb["train"].batch(128):
  imdb_reviews_train.extend(x.numpy())
  y_train.extend(y.numpy())
for x, y in imdb["test"].batch(128):
  imdb_reviews_test.extend(x.numpy())
  y_test.extend(y.numpy())
y_train = np.array(y_train)
y_test = np.array(y_test)


# Extract pre-trained BERT as a Keras layer.
bert_model_path = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
bert_layer = hub.KerasLayer(bert_model_path, trainable=False)

# Build tokenizer from pre-trained BERT vocabulary.
bert_tokenizer = FullTokenizer(
  vocab_file=bert_layer.resolved_object.vocab_file.asset_path.numpy(),
  do_lower_case=bert_layer.resolved_object.do_lower_case.numpy()
)

# TODO:
# Document longer than 512 words wont be able to be encoded by BERT,
# since its positional encoding has a hard limit for 512 words.
# For better results we may need to summarize the document into <= 512 tokens,
# or encode sentence by sentence then pool together.
maxlen = 256

# TODO:
# We need to manually handle CLS and SEP special token for sentence beginning and ending.

# Encode text with padding, masking, and segmentation (required by BERT even if we don't use it).
tok_seq_train = [bert_tokenizer.tokenize(text) for text in imdb_reviews_train]
wid_seq_train = [bert_tokenizer.convert_tokens_to_ids(toks)[:maxlen] for toks in tok_seq_train]
wid_seq_train_padded = pad_sequences(wid_seq_train, padding="post", maxlen=maxlen)
wid_seq_train_mask = (wid_seq_train_padded > 0).astype(int)
segment_ids_train = np.zeros_like(wid_seq_train_mask)

tok_seq_test = [bert_tokenizer.tokenize(text) for text in imdb_reviews_test]
wid_seq_test = [bert_tokenizer.convert_tokens_to_ids(toks)[:maxlen] for toks in tok_seq_test]
wid_seq_test_padded = pad_sequences(wid_seq_test, padding="post", maxlen=maxlen)
wid_seq_test_mask = (wid_seq_test_padded > 0).astype(int)
segment_ids_test = np.zeros_like(wid_seq_test_mask)

# NOTE:
# This won't fit into the Sequential API since bert layer has multiple inputs.
input_word_ids = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32, name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32, name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32, name="segment_ids")
pooled_output, _ = bert_layer([input_word_ids, input_mask, segment_ids])
relu_1 = tf.keras.layers.Dense(192, activation="relu")(pooled_output)
relu_2 = tf.keras.layers.Dense(48, activation="relu")(relu_1)
sigmoid = tf.keras.layers.Dense(1, activation="sigmoid")(relu_2)
bert = tf.keras.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=sigmoid, name="bert")
bert.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print(bert.summary())

model_file = "bert.h5"
metrics = bert.fit(
  x=[wid_seq_train_padded, wid_seq_train_mask, segment_ids_train],
  y=y_train,
  batch_size=32,  # Keep this small unless we have enough GPU memory.
  epochs=20,
  validation_data=([wid_seq_test_padded, wid_seq_test_mask, segment_ids_test], y_test),
  validation_steps=20,
  callbacks=[
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
    tf.keras.callbacks.ModelCheckpoint(model_file, monitor="val_loss", save_best_only=True)
  ],
  verbose=1)

# Only reach val_accuracy: 0.7766. (A simple network with embeddings trained from scratch can reach ~0.88.)
# We may need to unfreeze BERT layer for better result.
# Also the missing CLS and SEP token can be important?

# To load back the model:
#bert = tf.keras.models.load_model("bert.h5", custom_objects={"KerasLayer":hub.KerasLayer})
