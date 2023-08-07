'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np

# define your epsilon for laplace smoothing here

def baseline(train, test):
        '''
        Implementation for the baseline tagger.
        input:  training data (list of sentences, with tags on the words)
                test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
        output: list of sentences, each sentence is a list of (word,tag) pairs.
                E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
        '''
        list = []
        counts = {}
        maxTag = []
        tagCounts = {}
        for sentence in train:
                for word in sentence:
                        if word[0] not in counts: 
                                counts[word[0]] = {}
                        if word[1] not in counts[word[0]]: 
                                counts[word[0]][word[1]] = 0
                        counts[word[0]][word[1]] += 1
                        tagCounts[word[1]] = tagCounts.get(word[1], 0) + 1

        maxTag = max(tagCounts, key=tagCounts.get)
        for sentence in test:
                temp = []
                for word in sentence:
                        if word not in counts:
                                temp.append((word, maxTag))     
                        else:
                                temp.append((word, max(counts[word], key=counts[word].get)))
                list.append(temp)

        return list


def viterbi(train, test):
        '''
        Implementation for the viterbi tagger.
        input:  training data (list of sentences, with tags on the words)
                test data (list of sentences, no tags on the words)
        output: list of sentences with tags on the words
                E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]

        '''
        list = []
        tagCounts = {} # initial: HOW OFTEN DOES TAG T OCCUR
        tagPairCounts = {} # transition: HOW OFTEN DOES TAG T FOLLOW TAG T-1
        wordTagCounts = {} # emission: HOW OFTEN DOES TAG T YIELD WORD W  
        tagWordProb = {} # probability of word given tag
        tagPairProb = {}
        tagProb = {}
        alpha = 0.0001

        for sentence in train:
                prevTag = "START"
                for word in sentence:
                        currWord = word[0]
                        tag = word[1]
                        tagCounts[tag] = tagCounts.get(tag, 0) + 1
                        if prevTag not in tagPairCounts:
                                tagPairCounts[prevTag] = {}
                        if tag != "START" and tag != "END":
                                tagPairCounts[prevTag][tag] = tagPairCounts[prevTag].get(tag, 0) + 1
                        if tag not in wordTagCounts:
                                wordTagCounts[tag] = {}
                        wordTagCounts[tag][currWord] = wordTagCounts[tag].get(currWord, 0) + 1
                        prevTag = word[1]
                tagPairCounts["END"] = {}

        for tag in tagCounts:
                tagProb[tag] = math.log((tagCounts[tag] + alpha) / (len(train) + alpha * (len(tagCounts) + 1)))
        for tag in tagCounts:
                tagPairProb[tag] = {}
                for tag2 in tagPairCounts[tag]:
                        if tag2 != "START":
                                tagPairProb[tag][tag2] = math.log((tagPairCounts[tag][tag2] + alpha) / (tagCounts[tag] + alpha * (len(tagPairCounts[tag])+1)))
                tagPairProb[tag]["UNKNOWN"] = math.log(alpha / (tagCounts[tag] + alpha * (len(tagPairCounts[tag])+1)))
        for tag in tagCounts:
                tagWordProb[tag] = {}
                for word in wordTagCounts[tag]:
                        tagWordProb[tag][word] = math.log((wordTagCounts[tag][word] + alpha) / (tagCounts[tag] + alpha * (len(wordTagCounts[tag]) + 1)))
                tagWordProb[tag]["UNKNOWN"] = math.log(alpha / (tagCounts[tag] + alpha * (len(wordTagCounts[tag]) + 1)))

        labels = []
        for sentence in test:
                startprob = {}
                for tag in tagCounts:
                        startprob[tag] = ("",-math.inf)
                startprob["START"] = ("",0)
                trellis = [startprob]
                count = 0
                for i in range(1,len(sentence)):
                        tagprob = {}
                        word = sentence[i]
                        prevTag = ""
                        for tag in tagCounts:
                                maxprob = -math.inf
                                for prev_tag in tagCounts:
                                        prob = trellis[i-1][prev_tag][1]
                                        if word not in tagWordProb[tag]:
                                                prob += tagWordProb[tag]["UNKNOWN"]  
                                        else:
                                                prob += tagWordProb[tag][word]
                                        if tag not in tagPairProb[prev_tag]:
                                                prob += tagPairProb[prev_tag]["UNKNOWN"]
                                               
                                        else:
                                                prob  += tagPairProb[prev_tag][tag]
                                        if prob > maxprob:
                                                maxprob = prob
                                                prevTag = prev_tag
                                tagprob[tag] = (prevTag, maxprob)
                        trellis.append(tagprob)
                path = []
                prevTag = "END"
                for i in range(len(sentence)-1, -1, -1):
                        path.append((sentence[i], prevTag))
                        prevTag = trellis[i][prevTag][0]
                path.reverse()
                labels.append(path)
        return labels


        



        
        

def viterbi_ec(train, test):
    '''
    Implementation for the improved viterbi tagger.
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    raise NotImplementedError("You need to write this part!")



