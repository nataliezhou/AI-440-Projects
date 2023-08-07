'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
from collections import Counter
import copy

stopwords = set(["a","about","above","after","again","against","all","am","an","and","any","are","aren","'t","as","at","be","because","been","before","being","below","between","both","but","by","can","cannot","could","couldn","did","didn","do","does","doesn","doing","don","down","during","each","few","for","from","further","had","hadn","has","hasn","have","haven","having","he","he","'d","he","'ll","he","'s","her","here","here","hers","herself","him","himself","his","how","how","i","'m","'ve","if","in","into","is","isn","it","its","itself","let","'s","me","more","most","mustn","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan","she","she","'d","she","ll","she","should","shouldn","so","some","such","than","that","that","the","their","theirs","them","themselves","then","there","there","these","they","they","they","they","'re","they","this","those","through","to","too","under","until","up","very","was","wasn","we","we","we","we","we","'ve","were","weren","what","what","when","when","where","where","which","while","who","who","whom","why","why","with","won","would","wouldn","you","your","yours","yourself","yourselves"])

def create_frequency_table(train):
    '''
    Parameters:
    train (dict of list of lists) 
        - train[y][i][k] = k'th token of i'th text of class y

    Output:
    frequency (dict of Counters) 
        - frequency[y][x] = number of tokens of word x in texts of class y
    '''
    #raise RuntimeError("You need to write this part!")
    freq = {}
    for y in train:
        freq[y] = {}   
        for i in train[y]:
            for k in i:
                freq[y].setdefault(k, 0)
                freq[y][k] = freq[y][k] + 1
            
    return freq

def remove_stopwords(frequency):
    '''
    Parameters:
    frequency (dict of Counters) 
        - frequency[y][x] = number of tokens of word x in texts of class y

    Output:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in texts of class y,
          but only if x is not a stopword.
    '''
    #raise RuntimeError("You need to write this part!")
    nonstop = copy.deepcopy(frequency)
    for y in nonstop:
        for x in nonstop[y]:  # counts for words for each class x 
            if x in stopwords: # a stopword
                nonstop[y][x] = 0
    return nonstop

def laplace_smoothing(nonstop, smoothness):
    '''
    Parameters:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of x in y, if x not a stopword
    smoothness (float)
        - smoothness = Laplace smoothing hyperparameter

    Output:
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
        - likelihood[y]['OOV'] = likelihood of an out-of-vocabulary word given y

    Be careful that your vocabulary only counts words that occurred at least once
    in the training data for class y.
    '''
    #raise RuntimeError("You need to write this part!")
    likelihood = {}
    for y in nonstop:
        likelihood[y] = {}
        for x in nonstop[y]: # each word NOT in stopwords
            likelihood[y][x] = (nonstop[y][x] + smoothness) / (sum(nonstop[y].values()) + smoothness * (len(nonstop[y])+1))
        likelihood[y]['OOV'] = smoothness / (sum(nonstop[y].values()) + smoothness * (len(nonstop[y])+1))
    return likelihood

def naive_bayes(texts, likelihood, prior):
    '''
    Parameters:
    texts (list of lists) -
        - texts[i][k] = k'th token of i'th text
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
    prior (float)
        - prior = the prior probability of the class called "pos"

    Output:
    hypotheses (list)
        - hypotheses[i] = class label for the i'th text
    '''
    #raise RuntimeError("You need to write this part!")
    hypotheses = [] 
    for i in texts: # each text 
        negP = np.log(1-prior) 
        posP = np.log(prior) 
        for k in i: # token (word) in each text 
            if k not in stopwords:
                if k in likelihood['pos']:
                    posP += np.log(likelihood['pos'][k])
                else:
                    posP += np.log(likelihood['pos']['OOV'])
                if k in likelihood['neg']:
                    negP += np.log(likelihood['neg'][k])
                else:
                    negP += np.log(likelihood['neg']['OOV'])
        posP *= sum(likelihood['pos'].values())
        negP *= sum(likelihood['neg'].values())
        if posP > negP:
            hypotheses.append('pos')
        else:
            hypotheses.append('neg')
    return hypotheses

def optimize_hyperparameters(texts, labels, nonstop, priors, smoothnesses):
    '''
    Parameters:
    texts (list of lists) - dev set texts
        - texts[i][k] = k'th token of i'th text
    labels (list) - dev set labels
        - labels[i] = class label of i'th text
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in class y, x not stopword
    priors (list)
        - a list of different possible values of the prior
    smoothnesses (list)
        - a list of different possible values of the smoothness

    Output:
    accuracies (numpy array, shape = len(priors) x len(smoothnesses))
        - accuracies[m,n] = dev set accuracy achieved using the
          m'th candidate prior and the n'th candidate smoothness
    '''
    #raise RuntimeError("You need to write this part!")
    accuracies = np.zeros((len(priors), len(smoothnesses)))
    for i in range(len(priors)):
        for j in range(len(smoothnesses)):
            likelihood = laplace_smoothing(nonstop, smoothnesses[j])
            hypotheses = naive_bayes(texts, likelihood, priors[i])
            accuracies[i,j] = len([m for m,n in zip(hypotheses, labels) if m == n])/len(labels)
    return accuracies
                          
