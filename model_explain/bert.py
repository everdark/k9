import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tokenization import FullTokenizer
# https://github.com/tensorflow/models/blob/master/official/nlp/bert/tokenization.py

# Extract pre-trained BERT as a Keras layer.
bert_model_path = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
bert_layer = hub.KerasLayer(model_path, trainable=True)

# Build tokenizer from pre-trained BERT vocabulary.
bert_tokenizer = FullTokenizer(
  vocab_file=bert_layer.resolved_object.vocab_file.asset_path.numpy(),
  do_lower_case=bert_layer.resolved_object.do_lower_case.numpy()
)

# TODO:
# Document longer than 512 words wont be able to be encoded by BERT,
# since its positional encoding has a hard limit for 512 words.
# We need to summarize the document into <= 512 tokens, or encode sentence by sentence then pool together.

# Encode text with padding, masking, and segmentation (required by BERT even if we don't use it).
tok_seq_train = [bert_tokenizer.tokenize(text) for text in newsgroups_train.data]
wid_seq_train = [bert_tokenizer.convert_tokens_to_ids(toks) for toks in tok_seq_train]
wid_seq_train_padded = pad_sequences(wid_seq_train, padding="post")
wid_seq_train_mask = (wid_seq_train_padded > 0).astype(int)
segment_ids_train = np.zeros_like(wid_seq_train_mask)
maxlen = wid_seq_train_padded.shape[1]

tok_seq_test = [bert_tokenizer.tokenize(text) for text in newsgroups_test.data]
wid_seq_test = [bert_tokenizer.convert_tokens_to_ids(toks) for toks in tok_seq_test]
wid_seq_test_padded = pad_sequences(wid_seq_test, padding="post", maxlen=maxlen)
wid_seq_test_mask = (wid_seq_test_padded > 0).astype(int)
segment_ids_test = np.zeros_like(wid_seq_test_mask)

# NOTE:
# This won't fit into the Sequential API since bert layer has multiple inputs.
# Then we won't be able to use scikit-learn wrapper, then lime won't support.
input_word_ids = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32, name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32, name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32, name="segment_ids")
pooled_output, _ = bert_layer([input_word_ids, input_mask, segment_ids])
relu = tf.keras.layers.Dense(128, activation="relu")(pooled_output)
sigmoid = tf.keras.layers.Dense(1, activation="sigmoid")(relu)
bert = tf.keras.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=sigmoid, name="bert")
bert.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
print(bert.summary())

bert.fit(
  x=[wid_seq_train_padded, wid_seq_train_mask, segment_ids_train],
  y=newsgroups_train.target,
  batch_size=32, epochs=10,
  #validation_data=([wid_seq_test_padded, wid_seq_test_mask, segment_ids_test], newsgroups_test.target),
  #validation_steps=20,
  verbose=1)