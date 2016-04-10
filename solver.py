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


    def loadData(self, trainingData = None, testData = None):
        """
        load the Data and start needed preparations

        if the solver needs to prepare something after loading the data,
        overwrite this function to do so
        (for example: create a mask from the training-data)
        """

        if trainingData is not None:
            self.trainingData = trainingData
        if testData is not None:
            self.testData = testData


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

    def loadData(self, trainingData = None, testData = None):
        """
        load the Data and start needed preparations
        """

        if trainingData is not None:
            self.trainingData = trainingData
        if testData is not None:
            self.testData = testData

        # Create number mask for numbers 0 - 9
        if trainingData is not None:
            self.createMask(trainingData)


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


class KNearestSolver(Solver):
    """
    A KNearestSolver for digit recognition
    
    Solver that compares each test file with each training file and
    finds the k nearest ones. The solution is given by the maximal occurence
    of one number in the k nearest numbers.
    """

    def __init__(self, trainingData = None, testData = None, *args, **kwargs):
        """
        Initialization: set public variables
        """

        self.trainingData = trainingData
        self.testData = testData

    def solve(self, testData = None):
        """
        run the solving algorythm

        returns a numpy array with the found digits
        """

        dist = np.zeros(10)
        sol = np.zeros((len(self.testData),2))
        
        k = 10;
        
        if testData is None:
            testData = self.testData
        
        for i in range(len(testData)):
            # create index
            sol[i][0] = i+1
            # calculate k closest values
            near = np.argpartition(np.sum(np.abs(self.trainingData[:,1:] - testData[i]),
                                         axis = 1), k)
            #print(near[:k])
                                      
            # find class of k closest values and put it into solution
            sol[i][1] = np.argmax(np.bincount(self.trainingData[near[:k],0]));
                                      
            # heapq.nsmallest(5, a)[-1]
            # np.partition(a, 4)[4]

        self.sol = sol
        return sol
        
class NeuralNetwork(Solver):
    """
    A neural network for digit recognition
    
    ...
    """
    def __init__(self, trainingData = None, testData = None, *args, **kwargs):
        """
        Initialization: set public variables
        """
    
        self.trainingData = trainingData
        self.testData = testData



    def loadData(self, trainingData = None, testData = None):
        """
        load the Data and start needed preparations

        if the solver needs to prepare something after loading the data,
        overwrite this function to do so
        (for example: create a mask from the training-data)
        """

        if trainingData is not None:
            self.trainingData = trainingData
        if testData is not None:
            self.testData = testData

        if trainingData is not None:
            splitter = (int)(0.7 * len(trainingData))
            self.validationData = trainingData[splitter:]
            self.trainingData = trainingData[:splitter]


    def solve(self, testData = None):
        """
        run the solving algorythm

        returns a numpy array with the found digits
        """

        import network

        net = network.Network([784, 30, 10])

        # transform data to correct format
        training_inputs = self.trainingData[:,1:]
        training_results = [self.vectorized_result(y) for y in self.trainingData[:,0]]
        self.trainingData = list(zip(training_inputs, training_results))

        # no transformation needed
        validation_inputs = self.validationData[:,1:]
        validation_results = self.validationData[:,0]
        self.validationData = list(zip(validation_inputs, validation_results))

        net.SGD(self.trainingData, 30, 10, 3.0, test_data = self.validationData)

        # sol = np.argmax(net.feedforward(self.testData))
        # return sol

    def vectorized_result(self, j):
        """Return a 10-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere.  This is used to convert a digit
        (0...9) into a corresponding desired output from the neural
        network."""
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e
