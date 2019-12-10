import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print(tf.__version__)
if tf.test.is_gpu_available():
  print(tf.test.gpu_device_name())


# Load the data as tf.data.Dataset.
imdb = tfds.load(name="imdb_reviews", as_supervised=True)

imdb_reviews_train = []
imdb_reviews_test = []
imdb_y_train = []
imdb_y_test = []
for x, y in imdb["train"].batch(128):
  imdb_reviews_train.extend(x.numpy())
  imdb_y_train.extend(y.numpy())
for x, y in imdb["test"].batch(128):
  imdb_reviews_test.extend(x.numpy())
  imdb_y_test.extend(y.numpy())

imdb_y_train = np.array(imdb_y_train)
imdb_y_test = np.array(imdb_y_test)

# Take one review.
print(imdb_reviews_train[87])

print(imdb_y_train[87])  # Label. 0 as negative and 1 as positive.

# Build vocabulary. We use similar size as in our previous TfidfVectorizer.
# Since we will use zero padding, 0 cannot be used as OOV index.
# Keras tokenizer by default reserves 0 already. OOV token, if used, will be indexed at 1.
# Note that len(tokenizer.index_word) will be all vocabulary instead of `num_words`.
vocab_size = 20001  # +1 for 0 index used for padding.
oov_token = "<unk>"
tokenizer = Tokenizer(lower=True, oov_token=oov_token, num_words=vocab_size)
tokenizer.fit_on_texts(imdb_reviews_train)

# Encode text with padding to ensure fixed-length input.
maxlen = 256
seq_train = tokenizer.texts_to_sequences(imdb_reviews_train)
seq_train_padded = pad_sequences(seq_train, padding="post", truncating="post", maxlen=maxlen,)

seq_test = tokenizer.texts_to_sequences(imdb_reviews_test)
seq_test_padded = pad_sequences(seq_test, padding="post", truncating="post", maxlen=maxlen)

assert tokenizer.index_word[1] == oov_token
assert seq_train_padded.max() == vocab_size - 1

model_file = "models/imdb_rnn.h5"

embedding_size = 64
model = tf.keras.Sequential([
  tf.keras.layers.Embedding(
    vocab_size, embedding_size, input_length=maxlen,
    mask_zero=True, name="word_embedding"),
  tf.keras.layers.GRU(64, dropout=.2, name="GRU"),
  tf.keras.layers.Dense(1, activation="sigmoid", name="sigmoid")
], name="rnn_classifier")
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

print(model.summary())

metrics = model.fit(
  x=seq_train_padded, y=imdb_y_train,
  batch_size=32, epochs=10,
  validation_data=(seq_test_padded, imdb_y_test),
  validation_steps=20,
  callbacks=[
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
    tf.keras.callbacks.ModelCheckpoint(model_file, monitor="val_loss", save_best_only=True)
  ],
  verbose=1)
