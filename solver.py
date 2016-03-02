#main imports
import numpy as np

class Solver:
    """ base class to recognize digits """

    def __init__(self, trainingData, testData):
        """
        Initialization
        """
        raise NotImplementedError


    def solve(self):
        """
        run the solving algorythm

        returns a numpy array with the digits
        """
        raise NotImplementedError


class LinearSolver(Solver):
    """ a linear solver """

    def __init__(self, trainingData = None, testData = None):
        """
        Initialization: create the mask and set public variables
        """

        self.trainingData = trainingData
        self.testData = testData

        # Create number mask for numbers 0 - 9
        self.mask = self.createMask(trainingData)


    def solve(self, testData = None):
        """
        run the solving algorythm

        returns a numpy array with the found digits
        """

        if not testData:
            testData = self.testData

        # Compare test data with mask
        # Loop for each line representing one number
        dist = np.zeros(10)
        sol = np.zeros((len(testData),2))
        for i in range(len(testData)):

            # Compare line with each mask
            for j in range(10):
                dist[j] = self.absDist(self.mask[j],testData[i])

            # Find index where dist is minimal
            sol[i][0] = i+1
            sol[i][1] = np.argmin(dist)

        self.sol = sol
        return sol


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
