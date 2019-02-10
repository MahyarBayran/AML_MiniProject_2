
from LoadAndPreProcess import LPPMain
from FeatureExtraction import FEMain
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
features = set([])
if not readFromFile:
    features = FEMain(trainLines, trainY, numFeatures)
else:
    with open('sortedI.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            print( type(row[0]))
            features.add( row[0] )
        print(features)
        

print ( features )
