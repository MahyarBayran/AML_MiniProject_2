import os
import random
import string
import numpy as np
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


def splitData(posData, negData, start, end):
    # start and end are for validation data selection part
    val_size = end - start
    tr_size = len(posData) - val_size
    
    valLines = posData[start:end] + negData[start:end]
    trainLines = posData[0:start] + posData[end:-1] + negData[0:start] + negData[end:-1]
    
    trainY = np.concatenate( (np.ones( (1, tr_size) ), np.zeros( (1, tr_size)) ) , axis=1 )[0]
    valY = np.concatenate( (np.ones( (1, val_size) ), np.zeros( (1, val_size)) ) , axis=1 )[0]
    
    return trainLines, trainY, valLines, valY

def filerstopWords( tokens ):
    # input: tokens of a just one file
    sw = set(stopwords.words('english'))
    filtered = [t for t in tokens if not t in sw]
    
    return filtered

def prepTokens(tokens):
    # input: tokens of a just one file
    out_tokens = []
    
    # Lemmatization
    wnLemmatizer = WordNetLemmatizer()
    for token in tokens:
        out_tokens.append( wnLemmatizer.lemmatize(token) )
    
    # Remove stopwords
    out_tokens = filerstopWords( out_tokens )

    return out_tokens

def cleanLine(line):
    line = line.replace( '<br /><br />', '')
    # Clear Punctuations + Numbers
    line = line.translate( line.maketrans('','',string.punctuation + "0123456789") ) 
    # anything else to do?
            
    return line

def readAndPrep(path):
    files = os.listdir(path)
    lines = []
    for fname in files:#[0:100]:
        with open(path+fname, 'r', errors='replace') as f:
            # each file has only one line
            line = cleanLine( f.readline() ) 
            tokens = prepTokens( word_tokenize(line) )   
            lines.append ( tokens )

    return lines

def LPPMain(start, end):
    # start end are for validation set boundaries
    posLines = readAndPrep( 'train/pos/' )
    negLines = readAndPrep( 'train/neg/' )
    #testLines = readAndPrep( 'test/' )

    trainLines, trainY, valLines, valY = splitData( posLines, negLines, start, end)
    
    return trainLines, trainY, valLines, valY #, testLines