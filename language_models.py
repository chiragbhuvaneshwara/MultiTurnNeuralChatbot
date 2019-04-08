"""
Language Models
"""

from collections import defaultdict
from functools import partial
from itertools import chain
import pickle

import nltk
from nltk.lm import MLE, KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
from nltk.util import everygrams, pad_sequence

from load_data import Voc, SOS_token, EOS_token

flatten = chain.from_iterable


def my_padded_everygram_pipeline(order, text, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'):
    """
    Modified version of padded_everygram_pipeline that passes optional arguments to pad_both_ends.

    ######
    Default preprocessing for a sequence of sentences.

    Creates two iterators:
    - sentences padded and turned into sequences of `nltk.util.everygrams`
    - sentences padded as above and chained together for a flat stream of words

    :param order: Largest ngram length produced by `everygrams`.
    :param text: Text to iterate over. Expected to be an iterable of sentences:
    Iterable[Iterable[str]]
    :return: iterator over text as ngrams, iterator over text as vocabulary data
    """
    padding_fn = partial(pad_both_ends, n=order, pad_left=pad_left, pad_right=pad_right, left_pad_symbol=left_pad_symbol, right_pad_symbol=right_pad_symbol)
    return (
        (everygrams(list(padding_fn(sent)), max_len=order) for sent in text),
        flatten(map(padding_fn, text)),
    )


N = 2

# LM = MLE(N)
LM = KneserNeyInterpolated(N)
corpus = "data/cornell movie-dialogs corpus/formatted_movie_lines.txt"
# corpus = "test_corpus.txt"
with open(corpus) as f:
    raw = f.read()
print("corpus read")
tokens = nltk.word_tokenize(raw)
sents = [nltk.word_tokenize(s) for s in nltk.sent_tokenize(raw)]

voc = Voc(corpus)
print(voc)
for s in sents:
    for w in s:
        voc.addWord(w)
print(voc)
sents = [[str(SOS_token)] + [str(voc.word2index[w]) for w in s] + [str(EOS_token)] for s in sents]
print(sents[0])

# train, vocab = my_padded_everygram_pipeline(N, sents, left_pad_symbol="SOS", right_pad_symbol="EOS")
train, vocab = my_padded_everygram_pipeline(N, sents, left_pad_symbol="0", right_pad_symbol="0")
print("preprocessing ready")
LM.fit(train, vocab)
print("LM ready")
# LM.generate()
# print("how")
# print(LM.score("How"))
# print("are you")
# print(LM.score("you", ["are"]))

out_file = f"token_KneserNey_{N}_lm.pkl"
with open(out_file, "wb") as f:
    pickle.dump(LM, f)

print(LM.score("6"))
print(LM.score("4", ["8"]))
