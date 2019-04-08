"""
Interface for training the chatbot
"""

import argparse
import os

import torch
from torch import nn
from torch import optim

# from load_data import load_and_prepare_data, trim_rare_words
# from load_data import trim_rare_words
from cmd_pytorch_train import load_and_prepare_data, trim_rare_words
from chatbot_model import EncoderRNN, LuongAttnDecoderRNN
from train_chatbot import trainIters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a chatbot")

    parser.add_argument("-e", "--embeddings", nargs="?", default=None, type=str,
                        choices=[None, "word2vec", "glove50"], help="Specifies pre-trained \
                        word embeddings, None if no pre-trained embeddings are used")

    # Path to data
    parser.add_argument("corpus", help="Path to the preprocessed corpus directory")
    parser.add_argument("--save_dir", nargs="?", default=".", help="Path to directory where \
                        models should be saved")

    # Data options
    parser.add_argument("--max_sent_length", nargs="?", default=10, type=int,
                        help="Maximum sentence length to consider")
    parser.add_argument("--min_word_count", nargs="?", default=1, type=int,
                        help="Minimum occurence of a word to not be trimmed")

    # Model configuration
    parser.add_argument("--model_name", nargs="?", default="cb_model",
                        choices=["cb_model"], help="")
    parser.add_argument("--attention_model", nargs="?", default="dot",
                        choices=["dot", "general", "concat"],
                        help="Specifies the used attention model.")
    parser.add_argument("--hidden_size", nargs="?", type=int, default=500,
                        help="Dimension of the hidden layers (including embedding layer)")
    parser.add_argument("--encoder_n_layers", nargs="?", type=int, default=2,
                        help="Number of hidden layers of the encoder.")
    parser.add_argument("--decoder_n_layers", nargs="?", type=int, default=2,
                        help="Number of hidden layers of the decoder.")
    parser.add_argument("--dropout", nargs="?", type=float, default=0.1,
                        help="Dropout. Specifies the proportion of connections\
                        between layes which should be dropped.")
    parser.add_argument("--batch_size", nargs="?", type=int, default=64,
                        help="number of training instances in one batch (default=64)")

    # For retraining older models
    parser.add_argument("--model_file", nargs="?", default=None,
                        help="Path to model/checkpoint to load from; \
                        set to None if starting from scratch")
    # parser.add_argument("--checkpoint_iter", nargs="?", type=int, default=4000, help="")  # TODO: not used

    # Training and optimization configuration
    parser.add_argument("--clip", nargs="?", default=50.0, type=float, help="")
    parser.add_argument("--teacher_forcing_ratio", nargs="?", default=1.0,
                        type=float, help="Determines what proportion of the \
                        training examples will use teacher forcing.")
    parser.add_argument("-lr", "--learning_rate", nargs="?", default=0.0001,
                        type=float, help="Learning rate. Determines how much \
                        the parameters will be changed in each iteration.")
    parser.add_argument("--decoder_learning_ratio", nargs="?", default=5.0,
                        type=float, help="Adjustment of the learning rate for the decoder.")
    parser.add_argument("--n_iteration", nargs="?", default=4000, type=int,
                        help="Number of iteration the model will be trained.")
    parser.add_argument("--print_every", nargs="?", default=1, type=int,
                        help="Number of iterations after which progress will be printed.")
    parser.add_argument("--save_every", nargs="?", default=500, type=int,
                        help="Number of iterations after which the model will be saved.")

    args = parser.parse_args()
    print(args)

    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    # ===== load data =====
    voc, pairs = load_and_prepare_data(args.corpus)
    if args.min_word_count > 1:
        pairs = trim_rare_words(voc, pairs, args.min_word_count)
    if not pairs:
        raise ValueError(f"Training corpus contained no valid sentences for "
                         f"min_word_count {args.min_word_count} and max_sentence_length "
                         f"{args.max_sent_length}")
    corpus_name = args.corpus.split("/")[-2]

    # ===== set up model =====
    # Load model if a filename is provided
    if args.model_file:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, args.hidden_size)  # shouldn't it be embedding size?
    if args.model_file:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(args.hidden_size, embedding, args.encoder_n_layers, args.dropout)
    decoder = LuongAttnDecoderRNN(args.attention_model, embedding,
                                  args.hidden_size, voc.num_words,
                                  args.decoder_n_layers, args.dropout)
    if args.model_file:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(DEVICE)
    decoder = decoder.to(DEVICE)
    print('Models built and ready to go!')

    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate * args.decoder_learning_ratio)
    if args.model_file:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # Run training iterations
    print("Starting Training!")
    trainIters(args.model_name, voc, pairs, encoder, decoder, encoder_optimizer,
               decoder_optimizer, embedding, args.encoder_n_layers, args.decoder_n_layers,  # why are layers needed again?
               args.hidden_size, 
               args.save_dir, args.n_iteration, 1, args.print_every,
               args.save_every, args.clip, args.teacher_forcing_ratio,
               corpus_name, args.model_file)
