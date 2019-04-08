"""
Search Decoder after Li et al. (2016)
"""

import logging
from math import log
import pickle

import nltk
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
import torch
import torch.nn as nn

from load_data import PAD_token, SOS_token, EOS_token
from train_chatbot import GreedySearchDecoder, beam_decode

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


class MMISearchDecoder(nn.Module):

    def __init__(self, encoder, decoder, language_model, lm_order, penalty=0.5, cutoff=5, length_penalty=-1, voc=None):
        """
        Initializes a SearchDecoder

        Args:
            encoder: RNN that encodes an input string
            decoder: RNN that decodes an encoded input
            language_model: file with a pickled nltk.lm model
            lm_order: int, order of the given language model
            penalty: float, weight for language model score
            cutoff: int, up to this word index, the language model is used
            length_penalty: float, weight for discouraging long answers
            voc: Vocabulary that maps indices to words
        """
        super(MMISearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lm_order = lm_order
        self.penalty = penalty
        self.cutoff = cutoff
        self.length_penalty = length_penalty
        self.voc = voc
        with open(language_model, "rb") as f:
            self.lm = pickle.load(f)

    def forward(self, input_seq, input_length, max_length):
        """Alternative forward method with shorter run-time.

        Args:
            input_seq: a long-tensor with the input sequence
            input_length: a long-tensor with their lengths
            max_length: int, maximum sentence length of target
        Returns:
            A long-tensor of tokens and a float-tensor of their scores.
        """
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        decoder_input = torch.ones(1, 1, device=DEVICE, dtype=torch.long) * SOS_token
        all_tokens = torch.zeros([0], device=DEVICE, dtype=torch.long)
        sentence = [str(SOS_token)]  # initialize with SOS_token?
        all_scores = torch.zeros([0], device=DEVICE)
        for i in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # shape of decoder_output torch.Size([1, 7826]): 1 * len(vocabulary)
            lm_scores = torch.zeros([0])
            logging.debug(f"all tokens {all_tokens} {all_tokens.shape}")
            logging.debug(f"all scores {all_scores}")
            # hard cutoff
            if all_tokens.shape[0] > self.cutoff:
                lm_scores = 0
            else:
                best = (float("-inf"), None)
                for word, score in enumerate(decoder_output[0]):
                    logging.debug(f"word {word}, sentence {sentence}")
                    lm_score = self.lm.score(str(word), sentence[-(self.lm_order - 1):])
                    logging.debug(f"score {lm_score}")
                    if lm_score:
                        lm_score = log(lm_score)
                    else:
                        # never seen, favor unlikely but possible
                        lm_score = 0.0
                    current = torch.tensor([lm_score])
                    lm_scores = torch.cat((lm_scores, current), dim=0)
            combined_scores = decoder_output.log() - self.penalty * lm_scores
            decoder_scores, decoder_input = torch.max(combined_scores, dim=1)

            logging.debug(f"input {decoder_input}")
            sentence.append(str(decoder_input.item()))
            decoder_input = torch.tensor([decoder_input])

            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)

            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
            # Stop decoding when EOS is reached
            logging.debug(f"decoder_input {decoder_input.item()}")
            if decoder_input.item() == EOS_token:
                break

        # Return collections of word tokens and scores
        return all_tokens, all_scores

    def __str__(self):
        res = f"MMi-antilm with penalty {self.penalty}, cutoff {self.cutoff}"
        res += f", length penalty {self.length_penalty}"
        res += f" based on language model {self.lm} with order {self.lm_order}"
        return res


class MMIBidiSearchDecoder(nn.Module):

    def __init__(self, encoder, decoder, penalty=0.5, length_penalty=-1, voc=None, beam_size=10):
        """
        Initializes a SearchDecoder

        Args:
            encoder: RNN that encodes an input string
            decoder: RNN that decodes an encoded input
            penalty: float, weight of p(S|T)
            length_penalty: float weight of answer length (should be negative)
            voc: Vocabulary
            beam_size: int, beam size for decoding
        """
        super(MMIBidiSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.penalty = penalty
        self.length_penalty = length_penalty
        self.voc = voc
        self.beam_size = beam_size

    def forward(self, input_seq, input_length, max_length):
        # forward from Beamsearch, get n-best list
        # for each of the candidates in the n-best list
        #   - get p(T|S) (already computed during Beamsearch
        #   - compute p(S|T)
        #       - use T as encoder input
        #       - use decoder for each word get prob of the word that was in source
        #       - combines these collected word probs
        #   - compute length penalty of candidate
        #   - combine these three linearly (p(S|T) and N(T) have weights
        # selected the candidate with maximum combined prob
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # decoder_input = torch.ones(1, 1, device=DEVICE, dtype=torch.long) * SOS_token
        best_list = beam_decode(self.decoder, decoder_hidden, encoder_outputs, self.voc, self.beam_size, max_length)
        # best_list: list of tuples:
        # 1. list of strings
        # 2. float tensor of torch.Size([])

        best = (float("-inf"), None)
        for candidate, score in best_list:
            s_t_score = 0
            # candidate = " ".join(candidate)
            logging.debug(f"candidate {candidate}")
            indexes_batch = [[self.voc.word2index.get(word, 0) for word in candidate]]
            logging.debug(f"batch {indexes_batch}")
            lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
            input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
            input_batch = input_batch.to(DEVICE)
            lengths = lengths.to(DEVICE)
            for word in input_seq:
                logging.debug(f"word {word}")
                # input_step, last_hidden, encoder_outputs
                pseudo_encoder_outputs, pseudo_encoder_hidden = self.encoder(input_batch, lengths)
                pseudo_decoder_hidden = pseudo_encoder_hidden[:self.decoder.n_layers]
                pseudo_decoder_input = torch.ones(1, 1, device=DEVICE, dtype=torch.long) * SOS_token
                decoder_last, decoder_all = self.decoder(pseudo_decoder_input , pseudo_decoder_hidden, pseudo_encoder_outputs)
                logging.debug(f"new (decoder output) {decoder_last}")
                new = decoder_last[0][word.item()]
                s_t_score += new.item()
            logging.debug(f"p(S|T) score: {s_t_score}")
            combined_score = score + self.penalty * s_t_score + self.length_penalty * len(candidate)
            if combined_score > best[0]:
                best = (combined_score, candidate)
        all_tokens = [torch.tensor(self.voc.word2index.get(word, 0)) for word in candidate]
        all_scores = None  # TODO: how to get these scores
        return all_tokens, all_scores

    def __str__(self):
        res = f"MMi-bidi with penalty {self.penalty}, "
        res += f"length penalty {self.length_penalty} and "
        res += f"beam size {self.beam_size}, based on decoder {self.decoder}"
        return res


# logging.basicConfig(level=logging.DEBUG, filename="log_debug.txt")
logging.basicConfig(level=logging.INFO)
