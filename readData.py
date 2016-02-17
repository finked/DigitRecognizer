from numpy import genfromtxt

def readCsvData(filename):
	#"This function loads data from .csv files"
	
	my_data = genfromtxt(filename, delimiter=',', skip_header=1)
	return my_data