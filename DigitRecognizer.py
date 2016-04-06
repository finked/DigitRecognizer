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

    def __init__(self,
                 test=False,
                 testTest=False,
                 testTraining=False,
                 solver = "KNearestSolver",
                 *args,
                 **kwargs):
        """ initialize and run the recognition """

        # use shorter versions of both sets for a quicker test
        if test:
            self.trainFile = self.trainShort
            self.testFile = self.testShort

        # use shorter version of trainingset
        if testTraining:
            self.trainFile = self.trainShort

        # use shorter version of testset
        if testTest:
            self.testFile = self.testShort

        # try to load the solver from the dictionary of solvers
        try:
            self.solver = self.solvers[solver](None, None, args, kwargs)
        except:
            raise Exception("couldn't load the solver")


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
        self.solver.loadData(data, testdata)


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


    def crossValidate(self, splitAt = 0.7):
        """
        this function takes the training set where the right answer is known,
        splits it 70/30, takes the bigger part as the normal training set and
        the smaller part as test set. afterwards it compares the answer from
        the solver with the known right answer
        """

        #load the training set from the csv file
        data = readCsvData(self.trainFile)

        #define where to split the dataset. Rounds down
        splitter = (int)(splitAt * len(data))

        trainData = data[:splitter]
        validationData = data[splitter:]

        # run the solving algorythm
        # use the validation set without the solution in the first row
        self.solver.loadData(trainData, validationData[:,1:])
        self.sol = self.solver.solve()
        self.validationData = validationData

        # check the solution
        rightSol = validationData[:,0] == self.sol[:,1]
        rightPercent = np.bincount(rightSol)[1] / len(rightSol)
        return rightPercent, rightSol


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
    argTest = (('-t' in sys.argv[1:] ) | ('--test' in sys.argv[1:]))
    # when the test-Training-flag is given, the smaller trainingset-file is used
    argTestTraining = (('--test-Training' in sys.argv[1:])|('--test-training' in sys.argv[1:]))
    # when the test-Test-flag is given, the smaller testset-file is used
    argTestTest = (('--test-Test' in sys.argv[1:])|('--test-test' in sys.argv[1:]))

    DR = DigitRecognizer(test=argTest, testTraining=argTestTraining, testTest=argTestTest)
    DR.run()

if __name__ == "__main__":
    Main()
