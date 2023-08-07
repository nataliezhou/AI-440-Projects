'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np

def joint_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word to count
    word1 (str) - the second word to count

    Output:
    Pjoint (numpy array) - Pjoint[m,n] = P(X1=m,X2=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    '''
    #raise RuntimeError('You need to write this part!')
    # find height
    counts0 = []
    counts1 = []
    for text in texts:
      counts0.append(text.count(word0))
      counts1.append(text.count(word1))

    height = max(counts0) + 1
    width = max(counts1) + 1
    numTexts = len(texts)

    jointDist = [ [0]*width for i in range(height)]
    for i in range(height):
      for j in range(width):
        list0 = [x for x in range(len(counts0)) if counts0[x] == i] # returns the indices of texts that have count matching the row number
        count = 0
        for index in list0:
          if counts1[index] == j:
            count+=1
        jointDist[i][j] = count

    jointDist = np.array(jointDist) / numTexts
    return jointDist

def marginal_distribution_of_word_counts(Pjoint, index):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    index (0 or 1) - which variable to retain (marginalize the other) 

    Output:
    Pmarginal (numpy array) - Pmarginal[x] = P(X=x), where
      if index==0, then X is X0
      if index==1, then X is X1
    '''
    #raise RuntimeError('You need to write this part!')
    Pmarginal = []
    if index==0:
      Pmarginal = np.array(list(map(sum, Pjoint)))
    else:
      Pmarginal = np.array([ sum(x) for x in zip(*Pjoint) ])
    return Pmarginal
    
def conditional_distribution_of_word_counts(Pjoint, Pmarginal):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
    Pmarginal (numpy array) - Pmarginal[m] = P(X0=m)

    Outputs: 
    Pcond (numpy array) - Pcond[m,n] = P(X1=n|X0=m)
    '''
    #raise RuntimeError('You need to write this part!')
    Pcond = np.divide(Pjoint.T, Pmarginal).T
    return Pcond

def mean_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)
    
    Outputs:
    mu (float) - the mean of X
    '''
    #raise RuntimeError('You need to write this part!')
    mu = 0
    for i in range(len(P)):
      mu += P[i] * i
    return mu

def variance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)
    
    Outputs:
    var (float) - the variance of X
    '''
    #raise RuntimeError('You need to write this part!')
    mu = mean_from_distribution(P)
    var = 0
    for i in range(len(P)):
      var += (i - mu) ** 2 * P[i]
    return var

def covariance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[m,n] = P(X0=m,X1=n)
    
    Outputs:
    covar (float) - the covariance of X0 and X1
    '''
    #raise RuntimeError('You need to write this part!')
    X = marginal_distribution_of_word_counts(P, 0)
    Y = marginal_distribution_of_word_counts(P, 1)
    muX = mean_from_distribution(X)
    muY = mean_from_distribution(Y)

    muXY = 0
    for x in range(len(P)):
      for y in range(len(P[0])):
        muXY += P[x][y] * x * y

    return muXY - muX*muY

def expectation_of_a_function(P, f):
    '''
    Parameters:
    P (numpy array) - joint distribution, P[m,n] = P(X0=m,X1=n)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       must be a real number for all values of (x0,x1)
       such that P(X0=x0,X1=x1) is nonzero.

    Output:
    expected (float) - the expected value, E[f(X0,X1)]
    '''
    #raise RuntimeError('You need to write this part!')
    expected = 0
    for x in range(len(P)):
      for y in range(len(P[0])):
        expected += f(x,y) * P[x][y]
    return expected
    
