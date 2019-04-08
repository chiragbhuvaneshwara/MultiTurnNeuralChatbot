from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  

import torch
import torch.nn as nn

from load_data import PAD_token, SOS_token, EOS_token
from train_chatbot import beam_decode


class IDFSearchDecoder(nn.Module):

    def __init__(self, encoder, decoder, corpus, voc, penalty=0.5, length_penalty=-1, beam_size=10):
        super(IDFSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # initialize Vectorizer
        tfidf_vectorizer = TfidfVectorizer()
        train_set = get_train_text(corpus)
        tfidf_matrix = tfidf_vectorizer.fit_transform(train_set)
        idf = tfidf_vectorizer.idf_
        self.idf = dict(zip(tfidf_vectorizer.get_feature_names(), idf))
        self.penalty = penalty
        self.length_penalty = length_penalty
        self.voc = voc
        self.beam_size = beam_size

    def forward(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        best_list = beam_decode(self.decoder, decoder_hidden, encoder_outputs, self.voc, self.beam_size, max_length)
        best = (float("-inf"), None)
        for candidate, score in best_list:
            idf_scores = [self.idf.get(word, 0) for word in candidate]
            idf_score = sum(idf_scores) / len(idf_scores)
            combined_score = score + self.penalty * idf_score + self.length_penalty * len(candidate)
            if combined_score > best[0]:
                best = (combined_score, candidate)
        all_tokens = [torch.tensor(self.voc.word2index.get(word, 0)) for word in candidate]
        all_scores = None  # TODO: how to get these scores
        return all_tokens, all_scores


def get_train_text(filename):
    texts = []
    with open(filename) as f:
        for line in f:
            text = line.split("\t")
            texts.append(text[0])
            texts.append(text[1])
    return texts


if __name__ == "__main__":
    tfidf_vectorizer = TfidfVectorizer()
    train_set = get_train_text("data/cornell movie-dialogs corpus/formatted_movie_lines.txt")
    tfidf_matrix = tfidf_vectorizer.fit_transform(train_set)
    # print(tfidf_matrix)
    print(type(tfidf_matrix))

    idf = tfidf_vectorizer.idf_
    print(dict(zip(tfidf_vectorizer.get_feature_names(), idf)))
    # cosine = cosine_similarity(tfidf_matrix[length-1], tfidf_matrix)
    # print(cosine)
