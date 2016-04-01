#main imports
import sys
import numpy as np
from time import time

#local imports
from readData import readCsvData
import solver


class DigitRecognizer:
    """ this class is used for Digit recognition """

    # input-files
    trainFile = "train.csv"
    trainShort = "train_short.csv"
    testFile = "test.csv"
    testShort = "test_short.csv"
    outFile = "solution.csv"

    # solver classes
    solvers = {"LinearSolver": solver.LinearSolver, "KNearestSolver": solver.KNearestSolver}

    def __init__(self, test=False, solver = "KNearestSolver", *args, **kwargs):
        """ initialize and run the recognition """

        # use shorter versions for a quicker test
        if test:
            self.trainFile = self.trainShort
            self.testFile = self.testShort

        # try to load the solver from the dictionary of solvers
        try:
            self.solver = self.solvers[solver]
        except:
            raise Exception("couldn't load the solver")

        # these args are used by the solver when initialized in loadData
        self.solverArgs = args
        self.solverKwargs = kwargs


    def loadData(self):
        """
        read the trainingset and testset from the csv files
        and prepare the solver
        """

        # Read data from given csv file
        data = readCsvData(self.trainFile)

        # Read test data
        testdata = readCsvData(self.testFile)

        # prepare the solver with the data
        self.solver = self.solver(data,
                                  testdata,
                                  self.solverArgs,
                                  self.solverKwargs)


    def run(self):
        """
        read the data, run the solver and write the output file
        """
        
        self.loadData()

        # call the solver to recognize the digit
        self.sol = self.solver.solve()

        self.saveSolution()


    def saveSolution(self):
        # Write number in solution file
        np.savetxt(self.outFile, self.sol, delimiter=",", fmt = '%d, "%d"',
                   header = '"ImageId", "Label"', comments = '')


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


def Main():
    # when the test-flag is given, the much smaller test-files are used
    # (100 rows instead of 28000)
    # TODO: use a library to parse arguments
    if (('-t' in sys.argv[1:] ) | ('--test' in sys.argv[1:])):
        DR = DigitRecognizer(test=True)
    else:
        DR = DigitRecognizer()

    DR.run()

if __name__ == "__main__":
    Main()
