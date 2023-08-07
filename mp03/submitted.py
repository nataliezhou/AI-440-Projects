'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

def k_nearest_neighbors(image, train_images, train_labels, k):
    '''
    Parameters:
    image - one image
    train_images - a list of N images
    train_labels - a list of N labels corresponding to the N images
    k - the number of neighbors to return

    Output:
    neighbors - 1-D array of k images, the k nearest neighbors of image
    labels - 1-D array of k labels corresponding to the k images
    '''
    dists = []
    for i in range(len(train_images)):
        dist = np.sum((image-train_images[i])**2)
        dists.append((i,dist))
    
    dists.sort(key = lambda x: x[1])
    idxs = dists[0:k]
    
    top_idxs = list(zip(*idxs))[0]
    neighbors = np.array([train_images[x] for x in top_idxs])
    labels = np.array([train_labels[x] for x in top_idxs])
    return neighbors, labels

   #raise RuntimeError('You need to write this part!')


def classify_devset(dev_images, train_images, train_labels, k):
    '''
    Parameters:
    dev_images (list) -M images
    train_images (list) -N images
    train_labels (list) -N labels corresponding to the N images
    k (int) - the number of neighbors to use for each dev image

    Output:
    hypotheses (list) -one majority-vote labels for each of the M dev images
    scores (list) -number of nearest neighbors that voted for the majority class of each dev image
    '''

    hypotheses = []
    scores = []
    for i in range(len(dev_images)):
        neighbors, labels = k_nearest_neighbors(dev_images[i], train_images, train_labels, k)
        count = np.bincount(labels)
        hypotheses.append(np.argmax(count))
        scores.append(count[hypotheses[i]])
    return hypotheses, scores

   # raise RuntimeError('You need to write this part!')


def confusion_matrix(hypotheses, references):
    '''
    Parameters:
    hypotheses (list) - a list of M labels output by the classifier
    references (list) - a list of the M correct labels

    Output:
    confusions (list of lists, or 2d array) - confusions[m][n] is 
    the number of times reference class m was classified as
    hypothesis class n.
    accuracy (float) - the computed accuracy
    f1(float) - the computed f1 score from the matrix
    '''

    confusions = np.zeros((2,2))
    for x in range(len(set(references))):
        for y in range(len(set(hypotheses))):
            confusions[x][y] = sum([1 for i in range(len(hypotheses)) if hypotheses[i] == y and references[i] == x])
    tp = confusions[1][1]
    tn = confusions[0][0]
    fp = confusions[0][1]
    fn = confusions[1][0]
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    f1 = 2 / (1/recall + 1/precision)
    return confusions, accuracy, f1
    #raise RuntimeError('You need to write this part!')
