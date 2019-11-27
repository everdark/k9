from gensim.models import Word2Vec
import sentencepiece as spm

# Encode sentences into list of tokens as training input.
# Here we use our pre-trained sentencepiece model as the tokenizer.
outfile = "data/enwiki_sents_1000.txt"
sp = spm.SentencePieceProcessor()
sp.Load("m.model")
sp_sents = []
with open(outfile, "r", encoding="utf-8") as f:
  for _, line in enumerate(f):
    sp_sents.append(sp.EncodeAsPieces(line))

len(sp_sents)
sp_sents[0]

# Train word2vec.
gensim_w2v = Word2Vec(sp_sents, sg=1, size=128, window=3, hs=0, negative=5, min_count=5, workers=4)

# Check some similarity.
ws_meta = "\u2581"  # The sentencepiece special meta char.

gensim_w2v.wv.most_similar(ws_meta + "love")

gensim_w2v.wv.most_similar(ws_meta + "man")

gensim_w2v.wv.most_similar(ws_meta + "taiwan")

gensim_w2v.wv.most_similar(ws_meta + "computer")

gensim_w2v.wv.most_similar(ws_meta + "elephants")

