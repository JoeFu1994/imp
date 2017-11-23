import csv 
import itertools
import nltk

sentence_start_token = "START"
sentence_end_token = "END"
print ("Reading CSV file...")

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
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in separated_sentences]
print "Parsed %d sentences." % (len(sentences))
print "E.g. sentences: %s" % (sentences[0:5])