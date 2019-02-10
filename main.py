
from LoadAndPreProcess import LPPMain
from FeatureExtraction import FEMain
from FeatureExtraction import getX
import csv

# Parameters
start = 10001
end = 12500
numFeatures = 100
readFromFile = False

# Read and Preprocess Data
trainLines, trainY, valLines, valY = LPPMain(start, end)
print("Data preprocessed! Starting Feature Extraction ...")

# Get features
features = []
if not readFromFile:
    features = FEMain(trainLines, trainY, numFeatures)
else: # doesn't work! Don't know why
    with open('sortedI.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            features.append( row[0] )
        
print ( features )

# Build the inputs
X_counts_tr, X_occurs_tr = getX (features, trainLines)
X_counts_val, X_occurs_val = getX (features, valLines)
