"""Labs for training word2vec with tensorflow Keras API."""

import os

import sentencepiece as spm
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.preprocessing.sequence import make_sampling_table
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Sentence file.
outfile = "data/enwiki_sents.txt"


# Learn vocabulary.
# Make it small so the training time is manageable.
spm_args = "--input={}".format(outfile)
spm_args += " --model_prefix=enwiki"
spm_args += " --vocab_size=30000"
spm_args += " --model_type=unigram"
spm.SentencePieceTrainer.Train(spm_args)

sp = spm.SentencePieceProcessor()
sp.Load("enwiki.model")


# Model architecture.
k = 128  # Embedding dimension.

input_targets = tf.keras.layers.Input(shape=(1,), name="targets")
input_contexts = tf.keras.layers.Input(shape=(1,), name="contexts")

embeddings = tf.keras.layers.Embedding(len(sp), k, name="word_embeddings")
target_embeddings = embeddings(input_targets)
context_embeddings = embeddings(input_contexts)

dots = tf.keras.layers.Dot(axes=-1, name="logits")([target_embeddings, context_embeddings])
outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="sigmoid")(dots)
outputs = tf.keras.layers.Reshape((1,), input_shape=(1, 1))(outputs)

optimizer = tf.keras.optimizers.SGD(lr=.5, decay=1e-5, momentum=.9)
model = tf.keras.Model(inputs=[input_targets, input_contexts],
                       outputs=outputs,
                       name="word2vec")
model.compile(loss="binary_crossentropy", optimizer=optimizer)

print(model.summary())


# Skipgram generation.
# Keras skipgrams generates too many pairs. We need to downsample it for efficient training.
# The only flaw here is that if the target word appears more than once in the sentence,
# we will also extract the training window from those appearance.
# This still can already reduce lots of training pairs.
def downsample_skipgrams(ids, vocab_size, subsample=1e-3, window=2, neg=2):
  w = []
  y = []
  sampling_table = make_sampling_table(vocab_size, sampling_factor=subsample)
  span = 2 * window + 1
  targets = ids[window::span]
  pairs, labels = skipgrams(
    ids, vocabulary_size=vocab_size, window_size=np.random.randint(window - 1) + 1,
    negative_samples=neg, sampling_table=sampling_table, shuffle=True)
  for (t, c), l in zip(pairs, labels):
    if t in targets:
      w.append([t, c])
      y.append(l)
  return w, y


# In-memory training.
infile = "data/enwiki_sents.txt"
x = []
y = []
with open(infile, "r", encoding="utf-8") as f:
  for i, line in enumerate(f):
    ids = sp.EncodeAsIds(line)
    pairs, labels = downsample_skipgrams(ids, len(sp), window=2, neg=2)
    x.extend(pairs)
    y.extend(labels)
x = np.array(x)
y = np.array(y)

x.shape  # This is huge!

model.fit(x=[x[:,0], x[:,1]], y=y, batch_size=512, epochs=1, verbose=1)


# On-the-fly. Batch by sentence.
# The result is worse for one epoch, and also take longer to finish training.
sent_ids = []
with open(infile, "r", encoding="utf-8") as f:
  for i, line in enumerate(f):
    sent_ids.append(sp.EncodeAsIds(line))

len(sent_ids)

for epoch in range(1):
  total_loss = 0
  for i, sent in enumerate(sent_ids):
    if len(sent) >= 5:
      pairs, labels = downsample_skipgrams(sent, len(sp), window=2, neg=2)
      if len(pairs):  # May be empty due to dropouts.
          x = np.array(pairs)
          loss = model.train_on_batch(x=[x[:,0], x[:,1]], y=np.array(labels))
          total_loss += loss
          if i % 1000 == 0:
            # Batch loss will fluctuate a lot.
            print("Step {}, loss={}".format(i, loss), end="\r")
  print("Epoch {}, total_loss={}".format(epoch, loss))


# On-disk training.
def parse_line(text_tensor):
  """Convert a raw text line (in tensor) into skp-gram training examples."""
  ids = sp.EncodeAsIds(text_tensor.numpy())
  pairs, labels = downsample_skipgrams(ids, len(sp), window=2, neg=2)
  if len(pairs):
    targets, contexts = list(zip(*pairs))
  else:
    targets, contexts = [], []
  return targets, contexts, labels


# Since each text line can result in different number of training pairs,
# we need to use flat_map to flatten the parsed results before batching.
def parse_line_map_fn(text_tensor):
  targets, contexts, labels = tf.py_function(
    parse_line, inp=[text_tensor], Tout=[tf.int64, tf.int64, tf.int64])
  return tf.data.Dataset.from_tensor_slices(
    ({"targets": targets, "contexts": contexts}, labels))


# For simplicity we drop text lines that are too short.
batch_size = 512
dataset = tf.data.TextLineDataset(infile)
dataset = dataset.filter(lambda line: tf.greater(tf.strings.length(line), 20))
dataset = dataset.flat_map(parse_line_map_fn)
dataset = dataset.shuffle(buffer_size=10000).repeat(None)
dataset = dataset.batch(batch_size, drop_remainder=True)
dataset = dataset.prefetch(batch_size)

n_steps = 30000
losses = []
for i, (x, y) in enumerate(dataset):
  if i < n_steps:
    loss = model.train_on_batch(x, y, reset_metrics=True)
    if i % 1000 == 0:
      print("Step {}, loss={}".format(i, loss))#, end="\r")
      losses.append(loss)
  else:
    break

# check if the learning rate is indeed decayed.
model.optimizer._decayed_lr("float32").numpy()
model.save("enwiki_w2v.h5")


# Test similarity.
word_vectors = model.get_layer("word_embeddings").weights[0].numpy()
print(word_vectors.shape)


def find_similar_words(w, wv, top_k=10):# Use a larger batch size.# Use a larger batch size.
  ws_meta = "\u2581"  # The sentencepiece special meta char.
  i = sp.PieceToId(ws_meta + w)
  scores = cosine_similarity(wv[i,np.newaxis], wv)
  scores = np.squeeze(scores)
  sim_ind = np.squeeze(scores).argsort()[-top_k:][::-1]
  for i, s in zip(sim_ind, scores[sim_ind]):
    print("{:10} | {}".format(sp.IdToPiece(int(i)).replace(ws_meta, ""), s))


find_similar_words("man", wv=word_vectors)
find_similar_words("computer", wv=word_vectors)
find_similar_words("taiwan", wv=word_vectors)
find_similar_words("1", wv=word_vectors)

find_similar_words("love", wv=word_vectors)
find_similar_words("girl", wv=word_vectors)
find_similar_words("elephants", wv=word_vectors)
find_similar_words("elephant", wv=word_vectors)
find_similar_words("and", wv=word_vectors)
find_similar_words("or", wv=word_vectors)
