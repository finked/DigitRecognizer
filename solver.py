#main imports
import numpy as np
from joblib import Parallel, delayed

from time import time

class Solver:
    """
    base class to recognize digits

    a solver takes one set of known images to train (trainingData)
    and can then work on another set of images (testData) to get a solution

    the __init__() function should take the data as optional parameters
    and if present start the training and the solving-process
    """

    def __init__(self, trainingData = None, testData = None, *args, **kwargs):
        """
        Initialization

        if trainingData is present, start the training
        """
        raise NotImplementedError


    def solve(self):
        """
        run the solving algorythm

        returns a numpy array with the digits
        """
        raise NotImplementedError

    
    def timeit(self, funcName):
        """
        this function returns the time it takes to run another function

        I do this internally instead of the python package timeit because 
        like this we can preload the data and just run the functions repeatedly
        on this allready loaded data
        """

        #get local function to call
        func = getattr(self, funcName)

        before = time()
        func()
        after = time()
        return after - before


class MaskSolver(Solver):
    """
    base class of a solver with a mask to check the test set against
    """

    def createMask(self, data):
        """ create a mask from the given data """

        mask = np.zeros((10, data.shape[1]-1))

        for i in range(10):
            lines = [line[1:] for line in data if line[0] == i]
            if lines:
                mask[i] = np.average(lines, axis=0)

        self.mask = mask
        return mask


class LinearSolver(MaskSolver):
    """
    a linear solver

    this solver creates a mask from the training set and compares each image of
    the testset with the mask
    """

    def __init__(self, trainingData = None, testData = None, *args, **kwargs):
        """
        Initialization: create the mask and set public variables
        """

        self.trainingData = trainingData
        self.testData = testData

        # Create number mask for numbers 0 - 9
        if trainingData is not None:
            self.createMask(trainingData)


    def solve(self, testData = None):
        """
        run the solving algorythm

        returns a numpy array with the found digits
        """

        if testData is None:
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


    def solveVectorized(self, testData = None):
        """
        run the solving algorythm

        the computation of the absolute Distance is vectorized.
        returns a numpy array with the found digits
        """

        if testData is None:
            testData = self.testData

        # Compare test data with mask
        # Loop for each line representing one number
        dist = np.zeros(10)
        sol = np.zeros((len(testData),2))
        for i in range(len(testData)):

            sol[i][0] = i+1
            sol[i][1] = np.argmin(np.sum(np.abs(self.mask - testData[i]),
                                         axis = 1))

        self.sol = sol
        return sol

    
    def solveParallel(self, testData = None):
        """
        run the solving algorythm parallel
        """

        if testData is not None:
            self.testData = testData

        self.sol = np.zeros((len(self.testData),2))
        #start parallel loop
        #Parallel(backend="threading")(
                #delayed(self.solveStep)(i) for i in range(len(self.testData)))
        Parallel(n_jobs=4)(
                delayed(self.solveStep)(i) for i in range(len(self.testData)))
        return self.sol
        

    def solveStep(self, i):
        """
        solve one image
        """

        dist = np.zeros(10)

        # Compare line with each mask
        for j in range(10):
            dist[j] = self.absDist(self.mask[j],self.testData[i])

        # Find index where dist is minimal
        self.sol[i][0] = i+1
        self.sol[i][1] = np.argmin(dist)


    def absDist(self, list1, list2):
        """
        this function calculates the sum of absolute distances
        of each list value
        """

        dist = sum(abs(list1 - list2))
        return dist
