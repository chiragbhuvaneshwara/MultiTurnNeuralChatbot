"""
Interface for talking to the chatbot
"""

import argparse
import configparser
import logging

import torch
import torch.nn as nn

from load_data import Voc
from chatbot_model import EncoderRNN, LuongAttnDecoderRNN
from train_chatbot import GreedySearchDecoder, BeamSearchDecoder
from MMI_search_decoder import MMISearchDecoder, MMIBidiSearchDecoder
from IDF_search_decoder import IDFSearchDecoder
from evaluate_chatbot import evaluateInput, evaluate_automatic


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description="Talk to a pretrained chatbot.")
    parser.add_argument("model", help="File where the pretrained model is saved")
    parser.add_argument("-l", "--log", nargs="?", help="File where the conversation is recorded")
    parser.add_argument("--max_sent_length", nargs="?", default=10, type=int,
                        help="Maximum sentence length to consider")
    parser.add_argument("--hidden_size", nargs="?", type=int, default=500, help="")
    parser.add_argument("--attention_model", nargs="?", default="dot", choices=["dot", "general", "concat"], help="")
    parser.add_argument("--encoder_n_layers", nargs="?", type=int, default=2, help="")  # TODO: extract n_layers from filename
    parser.add_argument("--decoder_n_layers", nargs="?", type=int, default=2, help="")
    # parser.add_argument("--searcher", nargs="?", default="greedy", choices=["greedy", "mmi-antilm", "mmi-bidi", "beam-search", "idf-beam"], help="")
    parser.add_argument("--input", nargs="?", default=None, help="input file for pre-written human responses")
    parser.add_argument("--searcher_config", nargs="?", default=None, help="Path to the configuration file for the used searcher, if no file is specified use greedy search with default parameters")
    args = parser.parse_args()

    print(args)
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    if USE_CUDA:
        checkpoint = torch.load(args.model)
    else:
        # If loading a model trained on GPU to CPU
        checkpoint = torch.load(args.model, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc = Voc("evaluation")
    voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, args.hidden_size)
    embedding.load_state_dict(embedding_sd)

    # Initialize encoder & decoder models
    encoder = EncoderRNN(args.hidden_size, embedding, args.encoder_n_layers)
    decoder = LuongAttnDecoderRNN(args.attention_model, embedding, args.hidden_size, voc.num_words, args.decoder_n_layers)
    logging.debug(f"Encoder keys: {encoder_sd.keys()}")
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

    # Use appropriate device
    encoder = encoder.to(DEVICE)
    decoder = decoder.to(DEVICE)
    print('Models built and ready to go!')

    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()

    if args.searcher_config is None:
        searcher = GreedySearchDecoder(encoder, decoder)
    else:
        config = configparser.ConfigParser()
        config.read(args.searcher_config)
        searcher_type = config["CHOICE"]["Searcher"]
        if searcher_type == "mmi-antilm":
            penalty = float(config["MMI-ANTILM"]["Penalty"])
            cutoff = int(config["MMI-ANTILM"]["CutOff"])
            length_penalty = float(config["MMI-ANTILM"]["LengthPenalty"])
            language_model = config["MMI-ANTILM"]["LanguageModel"]
            order = int(config["MMI-ANTILM"]["LMOrder"])
            searcher = MMISearchDecoder(encoder, decoder, language_model, order, penalty, cutoff, length_penalty, voc)
        elif searcher_type == "mmi-bidi":
            penalty = float(config["MMI-BIDI"]["Penalty"])
            length_penalty = float(config["MMI-BIDI"]["LengthPenalty"])
            beam_size = int(config["MMI-BIDI"]["BeamSize"])
            searcher = MMIBidiSearchDecoder(encoder, decoder, penalty, length_penalty, voc, beam_size)
        elif searcher_type == "idf-beam":
            penalty = float(config["IDF-BEAM"]["Penalty"])
            length_penalty = float(config["IDF-BEAM"]["LengthPenalty"])
            beam_size = int(config["IDF-BEAM"]["BeamSize"])
            corpus = config["IDF-BEAM"]["Corpus"]
            searcher = IDFSearchDecoder(encoder, decoder, corpus, voc, penalty, length_penalty, beam_size)
        elif searcher_type == "beam":
            beam_size = int(config["BEAM"]["BeamSize"])
            searcher = BeamSearchDecoder(encoder, decoder, voc, beam_size)
        else:
            raise NotImplementedError

    # Begin chatting
    if args.input is None:
        evaluateInput(searcher, voc, args.max_sent_length, args.log)
    else:
        evaluate_automatic(args.input, searcher, voc, args.max_sent_length, args.log)
