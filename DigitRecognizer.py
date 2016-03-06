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

        # Read data from given csv file
        data = readCsvData(self.trainFile)

        # Read test data
        testdata = readCsvData(self.testFile)

        # call the solver to recognize the digit
        self.solver = solver(data, testdata)
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
        DigitRecognizer(test=True)
    else:
        DigitRecognizer()

if __name__ == "__main__":
    Main()
