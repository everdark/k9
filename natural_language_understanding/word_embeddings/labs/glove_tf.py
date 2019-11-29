"""Labs for training GloVe with tensorflow Keras API."""

from collections import defaultdict

import sentencepiece as spm
import tensorflow as tf
from tensorflow.keras import backend as K
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


# Create co-occurrence training triplets.
def count_cooccurrence(seq_list, window_size):
  cooccur_dict = defaultdict(int)
  for seq in seq_list:
    seq_len = len(seq)
    for i, wi in enumerate(seq):
      window_start = max(0, i - window_size)
      window_end = min(seq_len, i + window_size + 1)
      for j in range(window_start, window_end):
        if j != i:
          dist = abs(i - j)
          wj = seq[j]
          cooccur_dict[(wi, wj)] += 1 / dist
  return cooccur_dict

# Pre-parse corpus into word ids.
with open(outfile, "r", encoding="utf-8") as f:
  wiki_sents = f.read().split("\n")
wiki_sents_wids = [sp.EncodeAsIds(s) for s in wiki_sents]

len(wiki_sents_wids)


# Create co-occurence training triplets.
cooccur_dict = count_cooccurrence(wiki_sents_wids, 3)
triplets = np.array([(*k, v) for k, v in cooccur_dict.items()])

print(triplets.shape)


# Training pipeline.
batch_size = 512

dataset = tf.data.Dataset.from_tensor_slices(
  ({"targets": triplets[:,0].astype(int),
    "contexts": triplets[:,1].astype(int)},
  triplets[:,2]
))
dataset = dataset.shuffle(buffer_size=10000).repeat(count=None)
dataset = dataset.batch(batch_size, drop_remainder=True)
dataset = dataset.prefetch(batch_size)


# Model architecture.
def weighted_sum_squared_error(y_true, y_pred):
  """Custom loss for GloVe.
  y_true is the co-occurrence counts.
  y_pred is the model predicted scores (embedding dot-products plus biases).
  """
  weights = K.pow(K.clip(y_true / 100, 0, 1), 3/4)
  squared_err = K.square(y_pred - K.log(y_true))
  return K.sum(weights * squared_err, axis=-1)


embedding_size = 64  # Dimension of embeddings.
vocab_size = len(sp)

# We separate target and context word indices as two input layers.
input_targets = tf.keras.layers.Input(shape=(1,), name="targets")
input_contexts = tf.keras.layers.Input(shape=(1,), name="contexts")

# Two set of embeddings are leared: target and context word embeddings.
# Theoretically they are the same thing since the problem is symmetric.
# But empirically by learning two separate sets of embeddings the results are better.
embeddings_targets = tf.keras.layers.Embedding(
  vocab_size, embedding_size, name="target_embeddings")
embeddings_contexts = tf.keras.layers.Embedding(
  vocab_size, embedding_size, name="context_embeddings")
target_embed = embeddings_targets(input_targets)
context_embed = embeddings_contexts(input_contexts)

# Dot-product of the target and context word embeddings.
dots = tf.keras.layers.Dot(axes=-1, name="dots")([target_embed, context_embed])

# Bias.
target_biases = tf.keras.layers.Embedding(
  input_dim=vocab_size, output_dim=1, name="target_biases")(input_targets)
context_biases = tf.keras.layers.Embedding(
  input_dim=vocab_size, output_dim=1, name="context_biases")(input_contexts)

# Model outputs.
scores = tf.keras.layers.Add(name="scores")([dots, target_biases, context_biases])

glove = tf.keras.Model(inputs=[input_targets, input_contexts], outputs=scores, name="glove")
glove.compile(loss=weighted_sum_squared_error, optimizer="sgd")

print(glove.summary())


# Train.
n_epoch = 5
n_steps = len(triplets) // batch_size
for epoch in range(n_epoch):
  losses = []
  for i, (x, y) in enumerate(dataset):
    if i < n_steps:
      loss = glove.train_on_batch(x, y, reset_metrics=True)
      if i % 1000 == 0:
        print("Epoch {} Step {}, current batch loss = {}".format(epoch, i, loss), end="\r")
        losses.append(loss)
    else:
      break
  print("\nEpoch {}, mean batch loss = {}\n".format(epoch, np.mean(losses)))



# Save model.
glove.save("enwiki_glove.h5")


# Test word similarity.
def find_similar_words(w, wv, top_k=10):# Use a larger batch size.# Use a larger batch size.
  ws_meta = "\u2581"  # The sentencepiece special meta char.
  i = sp.PieceToId(ws_meta + w)
  scores = cosine_similarity(wv[i,np.newaxis], wv)
  scores = np.squeeze(scores)
  sim_ind = np.squeeze(scores).argsort()[-top_k:][::-1]
  for i, s in zip(sim_ind, scores[sim_ind]):
    print("{:10} | {}".format(sp.IdToPiece(int(i)).replace(ws_meta, ""), s))



target_word_vectors = glove.get_layer("target_embeddings").weights[0].numpy()
context_word_vectors = glove.get_layer("context_embeddings").weights[0].numpy()
word_vectors = target_word_vectors + context_word_vectors
print(word_vectors.shape)


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
