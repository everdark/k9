#!/usr/bin/env python

import itertools

from mxnet import gluon
import sentencepiece as spm
from gluonnlp.data import IMDB, SentencepieceTokenizer, PadSequence
from mxnet.gluon.data import SimpleDataset


# Load IMDB movie review dataset.
imdb_train = IMDB("train")
imdb_test = IMDB("test")
imdb_review_train, imdb_s_train = zip(*imdb_train)
imdb_y_train = SimpleDataset([1 if s > 5 else 0 for s in imdb_s_train])


# Train a sentencepiece tokenizer.
train_text_file = "train.txt"
model_prefix = "spm"
with open(train_text_file, "w") as f:
    for review in imdb_review_train:
        f.write(review.lower() + "\n")
spm_args = "--input={}".format(train_text_file)
spm_args += " --model_prefix={}".format(model_prefix)
spm_args += " --vocab_size=20000"
spm_args += " --pad_id=0"
spm_args += " --unk_id=1"
spm_args += " --bos_id=2"
spm_args += " --eos_id=3"
spm_args += " --model_type=unigram"
spm.SentencePieceTrainer.Train(spm_args)
tokenizer = SentencepieceTokenizer(model_prefix + ".model")


# Build vocabulary.
imdb_tok_train = [tokenizer(t.lower()) for t in imdb_review_train]
counter = gluonnlp.data.count_tokens(itertools.chain.from_iterable(imdb_tok_train))
vocab = gluonnlp.Vocab(counter, bos_token="<s>", eos_token="</s>", min_freq=10)

def encode(toks):
    return [vocab[tok] for tok in toks]

imdb_x_train = [encode(toks) for toks in imdb_tok_train]


# Build data pipeline.
# TODO: Wrap x and y before making a dataset?
maxlen = max([len(x) for x in imdb_x_train])
dataset = SimpleDataset(imdb_x_train)


dataset = dataset.transform(PadSequence(maxlen))
dataset = dataset.transform(mxnet.nd.array)


# Build the model.
model_ctx = mxnet.cpu()
model = mxnet.gluon.nn.Sequential()
with model.name_scope():
    model.add(mxnet.gluon.nn.Embedding(len(vocab), embedding_size))
    model.add(mxnet.gluon.rnn.GRU(64, dropout=.2))
    model.add(mxnet.gluon.nn.Dense(1))
model.initialize(ctx=model_ctx)
loss = mxnet.gluon.loss.SigmoidBinaryCrossEntropyLoss()
opt = mxnet.gluon.Trainer(model.collect_params(), "sgd", {"learning_rate": .01})


# Implement the training loop.
smoothing_constant = .01
epochs = 2
for e in range(epochs):
    cumulative_loss = 0
    for i, (X, y) in enumerate(zip(dataset, imdb_y_train)):
        X = X.as_in_context(model_ctx)
        y = y.as_in_context(model_ctx)
        with autograd.record():
            output = model(X)
            l = loss(output, label)
        l.backward()
        opt.step(X.shape[0])
        cumulative_loss += mxnet.nd.sum(l).asscalar()
    test_accuracy = evaluate_accuracy(test_data, model)
    train_accuracy = evaluate_accuracy(train_data, model)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
          (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))


# TODO: Explore Fit API.
from mxnet.gluon.contrib.estimator import estimator

est = estimator.Estimator(
    net=model,
    loss=loss,
    metrics=mxnet.metric.Accuracy(),
    trainer=opt,
    context=model_ctx)

est.fit(train_data=train_data_loader,
        epochs=2)
