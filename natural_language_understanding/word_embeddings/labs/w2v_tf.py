
import os
import re
import string

import jsonlines
import sentencepiece as spm
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.preprocessing.sequence import make_sampling_table
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


outfile = "data/enwiki_sents.txt"
infile = "data/enwiki_texts/AA/wiki_00"
def parse_wiki_sent(infile):
  paragraphs = []
  with jsonlines.open(infile) as jlines:
    for j in jlines:
      paragraphs.append(j["text"])  # One paragraph per line.

  # Break by newline.
  lines = []
  for p in paragraphs:
    lines.extend(p.lower().split("\n"))

  # Further break by sentence. We use a naive approach here just for simplicity.
  sents = []
  for l in lines:
    if len(l) >= 20:
      sents.extend(re.split(r"\. |! |\? ", l))

  return sents

sents = parse_wiki_sent(infile)
print(len(sents))

for s in sents[:3]:
  print(s + "\n")

sp = spm.SentencePieceProcessor()
sp.Load("m.model")


vocab_size = len(sp)
batch_size = 512
sampling_table = make_sampling_table(vocab_size, sampling_factor=1e-3)


def parse_line(text_tensor):
  """Convert a raw text line (in tensor) into skp-gram training examples."""
  ids = sp.EncodeAsIds(text_tensor.numpy())
  pairs, labels = skipgrams(
    ids, vocabulary_size=vocab_size,
    window_size=3, negative_samples=5,
    shuffle=True, sampling_table=sampling_table
  )
  targets, contexts = list(zip(*pairs))
  return targets, contexts, labels


# Since each text line can result in different number of training pairs,
# we need to use flat_map to flatten the parsed results before batching.
def parse_line_map_fn(text_tensor):
  targets, contexts, labels = tf.py_function(
    parse_line, inp=[text_tensor], Tout=[tf.int64, tf.int64, tf.int64])
  return tf.data.Dataset.from_tensor_slices(
    ({"targets": targets, "contexts": contexts}, labels))


# For simplicity we drop text lines that are too short.
dataset = tf.data.TextLineDataset(outfile)
dataset = dataset.filter(lambda line: tf.greater(tf.strings.length(line), 20))
dataset = dataset.flat_map(parse_line_map_fn)
dataset = dataset.shuffle(buffer_size=10000).repeat(count=2)
dataset = dataset.batch(batch_size, drop_remainder=True)
dataset = dataset.prefetch(batch_size)

# Test.
for x, y in dataset:
  print(x)
  print(y)
  break


k = 128

# We separate target and context word indices as two input layers.
input_targets = tf.keras.layers.Input(shape=(1,), name="targets")
input_contexts = tf.keras.layers.Input(shape=(1,), name="contexts")

# Word embeddings are looked up separately for target and context words.
embeddings = tf.keras.layers.Embedding(vocab_size, k, name="word_embeddings")
target_embeddings = embeddings(input_targets)
context_embeddings = embeddings(input_contexts)

# Dot-product of the target and context word embeddings with sigmoid activation.
dots = tf.keras.layers.Dot(axes=-1, name="logits")([target_embeddings, context_embeddings])
outputs = tf.keras.layers.Activation("sigmoid", name="sigmoid")(dots)
optimizer = tf.keras.optimizers.SGD(lr=.025, decay=1e-6, momentum=.9)
model = tf.keras.Model(inputs=[input_targets, input_contexts], outputs=outputs, name="word2vec")
model.compile(loss="binary_crossentropy", optimizer=optimizer)

print(model.summary())


x1 = embeddings(x["targets"])
x2 = embeddings(x["contexts"])
dot_op = tf.keras.layers.Dot(axes=-1, name="logits")
act_op = tf.keras.layers.Activation("sigmoid", name="sigmoid")

d = dot_op([x1, x2])
print(d.numpy()[:3,:])

a = act_op(d)
print(a.numpy()[:3,:])

print((x1.numpy() * x2.numpy()).sum(axis=1)[:3])  # Dot products.

print(tf.sigmoid(d.numpy()[:3]).numpy())  # Activation with sigmoid.


model.fit_generator(dataset, epochs=1, shuffle=True, workers=6, use_multiprocessing=True, verbose=1)
word_vectors = model.get_layer("word_embeddings").weights[0].numpy()
print(word_vectors.shape)
print(word_vectors)


def find_similar_words(w, wv, top_k=10):
  ws_meta = "\u2581"  # The sentencepiece special meta char.
  i = sp.PieceToId(ws_meta + w)
  scores = cosine_similarity(wv[i,np.newaxis], wv)
  scores = np.squeeze(scores)
  sim_ind = np.squeeze(scores).argsort()[-top_k:][::-1]
  for i, s in zip(sim_ind, scores[sim_ind]):
    print("{:10} | {}".format(sp.IdToPiece(int(i)).replace(ws_meta, ""), s))

find_similar_words("love", wv=word_vectors)
find_similar_words("girl", wv=word_vectors)
find_similar_words("computer", wv=word_vectors)
