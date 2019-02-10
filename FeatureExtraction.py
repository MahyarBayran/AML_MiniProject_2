import numpy as np
from math import log2
import csv

def getX(features, trainLines):

    X_counts = np.zeros( (len(trainLines), len(features)) )
    X_occurs = np.zeros( (len(trainLines), len(features)) )

    # for unigram features
    for i in range(0, len(trainLines)):
        for j in range(0,len(features)):
            X_counts[i][j] = countTerm( features[j], trainLines[i])
            if X_counts[i][j] > 0:
                X_occurs[i][j] = 1

    return X_counts, X_occurs


def countTerm (term, document):
    count = 0
    for t in document:
        if term == t:
            count += 1
    return count

def FEMain(trainLines, trainY, numFeatures):
    IGain = {} # store information gain for choosing each feature
    
    ################################
    ########## Unigrams ############
    ################################

    # for calculation of distribution of counts for features
    countXj = {}
    countXjYy = {}
    i = 0
    t_size = len(trainLines)

    for line in trainLines:
        found_words_in_line = []
        
        for w in line:
            if w not in found_words_in_line:
                found_words_in_line.append(w)
                c = countTerm (w, line)
            
                if (w, c) not in countXj.keys():
                    countXj[(w, c)] =  1
                    if trainY[i] == 1: 
                        countXjYy[(w, c, 1)] = 1
                        countXjYy[(w, c, 0)] = 0
                    else:
                        countXjYy[(w, c, 0)] = 1
                        countXjYy[(w, c, 1)] = 0

                else:
                    countXj[(w, c)] = countXj[(w, c)] + 1
                    if trainY[i] == 1:
                        countXjYy[(w, c, 1)] = countXjYy[(w, c, 1)] + 1
                    else:
                        countXjYy[(w, c, 0)] = countXjYy[(w, c, 0)] + 1
        i += 1

    # Calculate Mutual Information (assuming the training data is even)
    for w, c in countXj.keys():
        Pxj = countXj[(w, c)] / t_size
        for y in [0,1]:
            PxjYy = countXjYy[(w, c, y)]  / (t_size/2)
            PYy = 0.5
            if (Pxj > 0) & (PxjYy > 0):
                if w in IGain.keys():
                    IGain[w] = IGain[w] + PYy * PxjYy * log2 ( PxjYy /Pxj )
                else:
                    IGain[w] = PYy * PxjYy * log2 ( PxjYy /Pxj )


    ###### Any other features in mind? put it here for I calculation
    '''
    
    '''
    
    # Sort features by their Mutual Information
    sortedI = [(k, IGain[k]) for k in sorted(IGain, key=IGain.get, reverse=True)]

    # Write the sortedI to csv
    with open('sortedI.csv', 'w', errors='replace') as f:
        for key, value in sortedI:
            f.write("%s,%s\n"%(key, value))
    
    # Pick the top numFeatures informative features 
    features = []
    index = 0
    for key, value in sortedI:
        if index < numFeatures:
            features.append(key)
        index +=1
            
    return features