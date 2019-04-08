# coding: utf-8

# In[3]:


# Preparations
# ------------
#
# To start, Download the data ZIP file
# `here <https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html>`__
# and put in a ``data/`` directory under the current directory.
#
# After that, let’s import some necessities.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import ast


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


# In[4]:


# Load & Preprocess Data
# ----------------------
#
# The next step is to reformat our data file and load the data into
# structures that we can work with.
#
# The `Cornell Movie-Dialogs
# Corpus <https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html>`__
# is a rich dataset of movie character dialog:
#
# -  220,579 conversational exchanges between 10,292 pairs of movie
#    characters
# -  9,035 characters from 617 movies
# -  304,713 total utterances
#
# This dataset is large and diverse, and there is a great variation of
# language formality, time periods, sentiment, etc. Our hope is that this
# diversity makes our model robust to many forms of inputs and queries.
#
# First, we’ll take a look at some lines of our datafile to see the
# original format.
#

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("cmd_corpus", corpus_name)

def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

printLines(os.path.join(corpus, "../movie_lines.txt"))


# In[5]:


# Create formatted data file
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# The following functions facilitate the parsing of the raw
# *movie_lines.txt* data file.
#
# -  ``loadLines`` splits each line of the file into a dictionary of
#    fields (lineID, characterID, movieID, character, text)
# -  ``loadConversations`` groups fields of lines from ``loadLines`` into
#    conversations based on *movie_conversations.txt*
# -  ``extractSentencePairs`` extracts pairs of sentences from
#    conversations
#

# Splits each line of the file into a dictionary of fields
def loadLines(fileName, fields):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            line = line.replace("\n","")
            line = line + "<EOS>"
            values = line.split(" +++$+++ ")
            
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj 
    
    return lines


# Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            lineIds = eval(convObj["utteranceIDs"])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
            
            # Drop the last line for conversation with odd lines
            if len(lineIds) % 2 == 1:
                dropLine = convObj["lines"].pop()     
            
    return conversations




# Extract conversation sequence to text
def extractConversationSequence(conversations):
    c_seqs = []
    for conversation in conversations:
        # <EOC> token
        conversation["lines"][-1]["text"] = conversation["lines"][-1]["text"]+"\n<EOC>"
        
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"])):  
            conversationLine = conversation["lines"][i]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if conversationLine:
                c_seqs.append([conversationLine])
    return c_seqs


# In[6]:


# Now we’ll call these functions and create the file. We’ll call it
# *cmd_movie_lines.txt*.
#

# Define path to new file
datafile = os.path.join(corpus, "../formatted_movie_lines.txt")

delimiter = '\t'
# Unescape the delimiter
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

# Initialize lines dict, conversations list, and field ids
lines = {}
conversations = []
MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

# Load lines and process conversations
print("\nProcessing corpus...")
lines = loadLines(os.path.join(corpus, "../movie_lines.txt"), MOVIE_LINES_FIELDS)
print("\nLoading conversations...")
conversations = loadConversations(os.path.join(corpus, "../movie_conversations.txt"),
                                  lines, MOVIE_CONVERSATIONS_FIELDS)

#print(conversations[1:10])


# Write new csv file
print("\nWriting newly formatted file...")
with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
    for c_seq in extractConversationSequence(conversations):
        writer.writerow(c_seq)

# Get rid of \r\n before <EOS>
#datafile = [x.replace("\r\n","") for x in datafile]


# Print a sample of lines
print("\nSample lines from file:")
printLines(datafile)

######################################################################


# In[8]:


# Chop data in training and test set
# ~~~~~~~~~~~~~~~~~~
#
# 80% training 20% test; Total:83096 conversations
# 66477 16619
# Now we’ll split the samples from cmd_movie_lines.txt. We’ll call it
# *cmd_train.txt* and *cmd_test.txt*.

# Define path to new file
cmd_train = os.path.join(corpus, "../cmd_train.txt")
cmd_test = os.path.join(corpus, "../cmd_test.txt")


# Extract training set and test set
def extractTrain(conversations, num_train):
    c_train_seq = []
    c_train = random.sample(conversations, num_train)
    c_test_seq = []
    #type: c_train, c_test, conversations: lists of lists of dictionsaries
    
    for conversation in conversations:
        #conversation["lines"][-1]["text"] = conversation["lines"][-1]["text"]

        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"])):  
            conversationLine = conversation["lines"][i]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if conversationLine:
                if conversation in c_train:
                    #print("checkpoint1...")
                    c_train_seq.append([conversationLine])
                    #print(c_train_seq)
                else:
                    #print("checkpoint2...")
                    c_test_seq.append([conversationLine])
                    #print(c_test_seq)
                
        #totalData = extractConversationSequence(conversations)
        #print("checkpoint3...")
        #print(totalData[1:10])
        #print("train_Sample............")
        #print(c_train_seq[1:10])
        #print("test_Sample............")
        #print(c_test_seq[1:10])
    

    return c_train_seq, c_test_seq


# Write new csv file
print("\nSample lines from a,b:...")
a,b = extractTrain(conversations,66477)


print("\nWriting training file...")
with open(cmd_train, 'w', encoding='utf-8') as outputfile_1:
    writer = csv.writer(outputfile_1, delimiter=delimiter, lineterminator='\n')
    for c_train_seq in a:
        writer.writerow(c_train_seq)

        
print("\nWriting test file...")
with open(cmd_test, 'w', encoding='utf-8') as outputfile_2:
    writer = csv.writer(outputfile_2, delimiter=delimiter, lineterminator='\n')
    for c_test_seq in b:
        writer.writerow(c_test_seq)

    
# Print a sample of lines
print("\nSample lines from training file:")
printLines(cmd_train)

print("\nSample lines from test file:")
printLines(cmd_test)


# In[ ]:


# Ways have been tried: Chop data in training and test set    
# ~~~~~~~~~~~~~~~~~~
#
# While not True
#while not True:
#        conversation["lines"][-1]["text"] = conversation["lines"][-1]["text"]+"\n<EOC>"
        
#        for i in range(len(conversation["lines"])):  
#            conversationLine = conversation["lines"][i]["text"].strip()
#           # Filter wrong samples (if one of the lists is empty)
#            if conversationLine:
#                c_test_seq.append([conversationLine])
# For not in 
   #for conversation in conversations:
   #     for conversation not in c_train:
   #         conversation["lines"][-1]["text"] = conversation["lines"][-1]["text"]+"\n<EOC>"
   #         
   #         for i in range(len(conversation["lines"])):  
   #         conversationLine = conversation["lines"][i]["text"].strip()
   #        # Filter wrong samples (if one of the lists is empty)
   #             if conversationLine:
   #                 c_test_seq.append([conversationLine])

# List Subtraction
#c_test_seq = [item for item in extractConversationSequence(conversations) if item not in c_train_seq]


# Another dependent function
##def extractTest(conversations, c_train_seq):
 #   c_test_seq = extractConversationSequence(conversations) - c_train_seq
 #   return c_test_seq
                
#def extractConversationSequence(conversations):
#    c_seqs = []
#    for conversation in conversations:
#        conversation["lines"][-1]["text"] = conversation["lines"][-1]["text"]+'<EOC>'
        
        # Iterate over all the lines of the conversation
#        for i in range(len(conversation["lines"])):  
#            conversationLine = conversation["lines"][i]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
#            if conversationLine:
#                c_seqs.append([conversationLine])
#    return c_seqs
                        


# In[ ]:


# Other Way: Chop data in training and test set
# ~~~~~~~~~~~~~~~~~~
#
# This method chops w.r.t. lines
#import nltk
#from random import shuffle
#def shuffle_split(infilename, outfilename1, outfilename2, num_train):
    
#    with open(infilename, 'r') as f:
#        c_seq = ['\n-----\n'.join('<EOC>'.tokenize(f))]
#        traingdata = random.sample(c_seq,num_train)
#        testdata = [item for item in c_seq if item not in traingdata]

#    with open(outfilename1, 'w') as f:
#        f.writelines(lines(traingdata))
#        
#    with open(outfilename2, 'w') as f:
#        f.writelines(lines(testdata))        
#shuffle_split(datafile, cmd_train, cmd_test, 66477)


# In[ ]:


# Load and trim data
# ~~~~~~~~~~~~~~~~~~
#
# Our next order of business is to create a vocabulary and load
# query/response sentence pairs into memory.
#
# Note that we are dealing with sequences of **words**, which do not have
# an implicit mapping to a discrete numerical space. Thus, we must create
# one by mapping each unique word that we encounter in our dataset to an
# index value.
#
# For this we define a ``Voc`` class, which keeps a mapping from words to
# indexes, a reverse mapping of indexes to words, a count of each word and
# a total word count. The class provides methods for adding a word to the
# vocabulary (``addWord``), adding all words in a sentence
# (``addSentence``) and trimming infrequently seen words (``trim``). More
# on trimming later.
#

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)


######################################################################
# Now we can assemble our vocabulary and query/response sentence pairs.
# Before we are ready to use this data, we must perform some
# preprocessing.
#
# First, we must convert the Unicode strings to ASCII using
# ``unicodeToAscii``. Next, we should convert all letters to lowercase and
# trim all non-letter characters except for basic punctuation
# (``normalizeString``). Finally, to aid in training convergence, we will
# filter out sentences with length greater than the ``MAX_LENGTH``
# threshold (``filterPairs``).
#
4444444444444444444444444444444
MAX_LENGTH = 10  # Maximum sentence length to consider

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# Read query/response pairs and return a voc object
def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs


# Load/Assemble voc and pairs
save_dir = os.path.join("data", "save")
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
# Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)


# In[ ]:


# Another tactic that is beneficial to achieving faster convergence during
# training is trimming rarely used words out of our vocabulary. Decreasing
# the feature space will also soften the difficulty of the function that
# the model must learn to approximate. We will do this as a two-step
# process:
#
# 1) Trim words used under ``MIN_COUNT`` threshold using the ``voc.trim``
#    function.
#
# 2) Filter out pairs with trimmed words.
#

MIN_COUNT = 3    # Minimum word count threshold for trimming

def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


# Trim voc and pairs
pairs = trimRareWords(voc, pairs, MIN_COUNT)

