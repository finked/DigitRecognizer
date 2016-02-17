from readData import *

data = readCsvData("train_short.csv")

#mask = array[10]

for i in range(10):
    lines = [line for line in data if line[0] == i]
    print("%s - %s" % (i, len(lines)))
