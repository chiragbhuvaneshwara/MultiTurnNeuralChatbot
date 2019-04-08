"""
Manual evaluation / chatting functions for a trained model.
"""

# MAX_LENGTH = 10

import os
import time

import torch

from load_data import normalizeString, indexesFromSentence


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

######################################################################
# Evaluate my text
# ~~~~~~~~~~~~~~~~
#
# Now that we have our decoding method defined, we can write functions for
# evaluating a string input sentence. The ``evaluate`` function manages
# the low-level process of handling the input sentence. We first format
# the sentence as an input batch of word indexes with *batch_size==1*. We
# do this by converting the words of the sentence to their corresponding
# indexes, and transposing the dimensions to prepare the tensor for our
# models. We also create a ``lengths`` tensor which contains the length of
# our input sentence. In this case, ``lengths`` is scalar because we are
# only evaluating one sentence at a time (batch_size==1). Next, we obtain
# the decoded response sentence tensor using our ``GreedySearchDecoder``
# object (``searcher``). Finally, we convert the response’s indexes to
# words and return the list of decoded words.
#
# ``evaluateInput`` acts as the user interface for our chatbot. When
# called, an input text field will spawn in which we can enter our query
# sentence. After typing our input sentence and pressing *Enter*, our text
# is normalized in the same way as our training data, and is ultimately
# fed to the ``evaluate`` function to obtain a decoded output sentence. We
# loop this process, so we can keep chatting with our bot until we enter
# either “q” or “quit”.
#
# Finally, if a sentence is entered that contains a word that is not in
# the vocabulary, we handle this gracefully by printing an error message
# and prompting the user to enter another sentence.
#


def evaluate(searcher, voc, sentence, max_length):
    # ## Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(DEVICE)
    lengths = lengths.to(DEVICE)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(searcher, voc, max_length, log_file=None):
    input_sentence = ''
    print(log_file)
    if log_file is None:
        log_file = os.devnull
    print("To chat with the bot, enter a sentence. 'q' or 'quit' will stop the dialogue")
    with open(log_file, "a") as f:
        f.write(f"Started new conversation at {time.asctime(time.localtime())}\n")
        while(1):
            try:
                # Get input sentence
                input_sentence = input('> ')
                # Check if it is quit case
                if input_sentence == 'q' or input_sentence == 'quit':
                    break
                inp = f'Human:\t{input_sentence}\n'
                f.write(inp)
                # Normalize sentence
                input_sentence = normalizeString(input_sentence)
                # Evaluate sentence
                output_words = evaluate(searcher, voc, input_sentence, max_length)
                # Format and print response sentence
                output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
                print_output = ' '.join(output_words)
                output = f'Bot:\t{print_output}\n'
                print('Bot: ', print_output)
                f.write(output)
            except KeyError as e:
                print("Error: Encountered unknown word.", e)
                f.write(f"Error: Encountered unknown word. {e}")


def evaluate_automatic(input_file, searcher, voc, max_length, log_file=None):
    # set up logging
    if log_file is None:
        log_file = os.devnull

    # read input
    lines = []
    with open(input_file) as f:
        lines = [line for line in f]
    with open(log_file, "a") as f:
        f.write(f"\nStarted new conversation at {time.asctime(time.localtime())}\n")
        f.write(f"current searcher is {searcher}\n")
        for line in lines:
            line = line.strip()
            try:
                inp = f'Human:\t{line}\n'
                f.write(inp)
                # Normalize sentence
                input_sentence = normalizeString(line)
                # Evaluate sentence
                output_words = evaluate(searcher, voc, input_sentence, max_length)
                # Format and print response sentence
                output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
                output = f"Bot:\t{ ' '.join(output_words) }\n"
                f.write(output)
            except KeyError:
                f.write("Error: Encountered unknown word.")
    print("Finished.")
