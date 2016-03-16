#main imports
import sys
import numpy as np

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

    # TODO: move compare and save logic into their own functions
    def __init__(self, test=False, solver = solver.LinearSolver):
        """ initialize and run the recognition """

        # use shorter versions for a quicker test
        if test:
            self.trainFile = self.trainShort
            self.testFile = self.testShort

        self.solver = solver


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
        self.solver = self.solver(data, testdata)


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
