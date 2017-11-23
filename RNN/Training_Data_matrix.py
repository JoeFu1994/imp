import csv 
import itertools
import operator 
import numpy as np
import nltk
import sys
import os

from datetime import datetime


vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"


#Read data and append start token and end token
print "Reading CSV file..."

with open('../Flickr8k.lemma.token.csv', 'rb') as f:
    reader = csv.reader(f, skipinitialspace = True)
    reader.next()
    #Split into sentences 
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    sentences = ["%s" % x for x in sentences]
    separated_sentences = []
    for x in sentences: 
        char_count = 0
        for char in x:
            if char == '\t':
                separated_sentences.append(x[char_count+1:])
            else: char_count = char_count+1
    #print separated_sentences


    #Append start token and end token 
    separated_sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in separated_sentences]
print "Parsed %d sentences." % (len(separated_sentences))


#with open('../Flickr8k.lemma.token.csv', 'rb') as f:
#    reader = csv.reader(f, skipinitialspace = True)
#    reader.next()
#    Split into sentences 
#    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    #Append start token and end token 
#    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
#print "Parsed %d sentences." % (len(sentences))


#Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in separated_sentences]

#Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())

#Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
temp = vocab[0]
vocab[0] = vocab[1]
vocab[1] = vocab[2]
vocab[2] = temp
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])


print "Using vocab size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

#Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

print "\nExample sentence: '%s'" % separated_sentences[0]
print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]

#Create the training data 
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

