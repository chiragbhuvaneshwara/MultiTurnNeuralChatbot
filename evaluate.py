'''

This evaluation script is to be used for evaluating the performance
of the neural multi-turn chatbot. The algorithm below is used to
calculate the BLEU score values for each sentence/reply/text generated
by the neural chatbot. The text answer from the chatbot is evaluated
against test data. For testing purposes for general chat, human responses
will be used as the ground truth reference. 

'''

#   importing libraries for RegExp Tokenization and BLEU score

from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

#   this is completely optional. However, this two lines of codes
#   are included in order to avoid the UserWarning warnings which
#   might be generated in case of higher order n-grams calculation
#   for unequal sentence lengths.

import warnings
warnings.filterwarnings("ignore")

def evaluate_bleu(reference_text, chatbot_text):

    # tokenize both reference and candidate texts,
    # tokenizing only the words without punctuations
    
    tokenizer = RegexpTokenizer(r'\w+')
    reference, candidate = tokenizer.tokenize(reference_text), tokenizer.tokenize(chatbot_text)

    # calculate the BLEU score for 1-gram, 2-grams, 3-grams and 4-grams

    bleu_1 = sentence_bleu([reference], candidate, weights=(1, 0, 0, 0))
    bleu_2 = sentence_bleu([reference], candidate, weights=(0.5, 0.5, 0, 0))
    bleu_3 = sentence_bleu([reference], candidate, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25))

    print('BLEU 1-gram: %.2f %%' % (bleu_1 * 100))
    print('BLEU 2-gram: %.2f %%' % (bleu_2 * 100))
    print('BLEU 3-gram: %.2f %%' % (bleu_3 * 100))
    print('BLEU 4-gram: %.2f %%' % (bleu_4 * 100))
    
    return bleu_1, bleu_2, bleu_3, bleu_4

# sample text for testing

reference = 'I am going to the school now.'
text = 'I am coming back from the school now.'

print('Reference:', reference)
print('Candidate:', text, '\n')

# call the function for testing

bleu_1, bleu_2, bleu_3, bleu_4 = evaluate_bleu(reference, text)

# sample results

'''

Reference: I am going to the school now.
Candidate: I am coming back from the school now. 

BLEU 1-gram: 62.50 %
BLEU 2-gram: 51.75 %
BLEU 3-gram: 35.84 %
BLEU 4-gram: 0.00 %

'''