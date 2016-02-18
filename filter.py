import numpy as np
from readData import *

def createMask(data):
    mask = np.zeros((10, data.shape[1]-1))

    for i in range(10):
        lines = [line[1:] for line in data if line[0] == i]
        #print("%s - %s" % (i, len(lines)))
        if lines:
            mask[i] = np.average(lines, axis=0)
    return mask

def absDist(list1, list2):
    # this function calculates the sum of absolute distances
    # of each list value
    dist = sum(abs(list1 - list2))
    return dist

# Read data from given csv file
data = readCsvData("train.csv")

# Create number mask for numbers 0 - 9
mask = createMask(data)

# Read test data
testdata = readCsvData("test.csv")

# Compare test data with mask
# Loop for each line representing one number
dist = np.zeros(10)
sol = np.zeros((len(testdata),2))
for i in range(len(testdata)):
    
    # Compare line with each mask
    for j in range(10):
        dist[j] = absDist(mask[j],testdata[i])
        
    # Find index where dist is minimal
    sol[i][0] = i+1
    sol[i][1] = np.argmin(dist)
    
    
# Write number in solution file
np.savetxt("solution.csv", sol, delimiter=",", fmt = '%d',
           header = '"ImageId", "Label"')