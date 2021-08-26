

#  Drive Code
if __name__ == "__main__":
    #  Load input data from QuickSort.txt.
    input_data = []

    f = open("QuickSort.txt")
    line = f.readline()
    while line:
        input_data.append( int(line) )
        line = f.readline()
    f.close()