#main imports
import sys
import numpy as np

#local imports
from readData import *


class filter:
    """ this class is used for character recognition """

    # input-files
    trainFile = "train.csv"
    trainShort = "train_short.csv"
    testFile = "test.csv"
    testShort = "test_short.csv"
    outFile = "solution.csv"

    # TODO: move compare and save logic into their own functions
    def __init__(self, test=False):
        """ initialize and run the recognition """

        # use shorter versions for a quicker test
        if test:
            self.trainFile = self.trainShort
            self.testFile = self.testShort

        # Read data from given csv file
        data = readCsvData(self.trainFile)

        # Create number mask for numbers 0 - 9
        mask = self.createMask(data)

        # Read test data
        testdata = readCsvData(self.testFile)

        # Compare test data with mask
        # Loop for each line representing one number
        dist = np.zeros(10)
        sol = np.zeros((len(testdata),2))
        for i in range(len(testdata)):

            # Compare line with each mask
            for j in range(10):
                dist[j] = self.absDist(mask[j],testdata[i])

            # Find index where dist is minimal
            sol[i][0] = i+1
            sol[i][1] = np.argmin(dist)


        # Write number in solution file
        np.savetxt(self.outFile, sol, delimiter=",", fmt = '%d, "%d"',
                   header = '"ImageId", "Label"', comments = '')


    def createMask(self, data):
        """ create a mask from the given data """

        mask = np.zeros((10, data.shape[1]-1))

        for i in range(10):
            lines = [line[1:] for line in data if line[0] == i]
            #print("%s - %s" % (i, len(lines)))
            if lines:
                mask[i] = np.average(lines, axis=0)
        return mask


    def absDist(self, list1, list2):
        """
        this function calculates the sum of absolute distances
        of each list value
        """

        dist = sum(abs(list1 - list2))
        return dist


if __name__ == "__main__":
    # when the test-flag is given, the much smaller test-files are used
    # (100 rows instead of 28000)
    # TODO: use a library to parse arguments
    if (('-t' in sys.argv[1:] ) | ('--test' in sys.argv[1:])):
        filter(test=True)
    else:
        filter()
