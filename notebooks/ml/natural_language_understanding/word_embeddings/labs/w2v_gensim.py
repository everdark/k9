from gensim.models import Word2Vec
import sentencepiece as spm

# Encode sentences into list of tokens as training input.
# Here we use our pre-trained sentencepiece model as the tokenizer.
outfile = "data/enwiki_sents.txt"
sp = spm.SentencePieceProcessor()
sp.Load("enwiki.model")
sp_sents = []
with open(outfile, "r", encoding="utf-8") as f:
  for _, line in enumerate(f):
    sp_sents.append(sp.EncodeAsPieces(line))

len(sp_sents)
sp_sents[0]

# Train word2vec.
gensim_w2v = Word2Vec(sp_sents, sg=1, size=128, window=2, hs=0, negative=2, min_count=5, workers=4)

len(gensim_w2v.wv.vocab)

# Check some similarity.
ws_meta = "\u2581"  # The sentencepiece special meta char.

gensim_w2v.wv.most_similar(ws_meta + "love")

gensim_w2v.wv.most_similar(ws_meta + "man")

gensim_w2v.wv.most_similar(ws_meta + "taiwan")

gensim_w2v.wv.most_similar(ws_meta + "computer")

gensim_w2v.wv.most_similar(ws_meta + "elephants")

gensim_w2v.wv.most_similar(ws_meta + "1")

gensim_w2v.wv.most_similar(ws_meta + "one")

gensim_w2v.wv.most_similar(ws_meta + "king")

gensim_w2v.wv.most_similar(ws_meta + "money")

gensim_w2v.wv.most_similar(ws_meta + "freedom")
