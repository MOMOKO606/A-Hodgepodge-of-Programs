"""
This script is for the Homework2 - Programming Assignment 2 - Question 1 to 3
in the "StanfordOnline CSX0003 Algorithms: Design and Analysis, Part 1" on edX.

Your task is to compute the total number of comparisons used to sort the given input file by QuickSort.
As you know, the number of comparisons depends on which elements are chosen as pivots,
so we'll ask you to explore three different pivoting rules,
1.Use the first element of the array as the pivot element.
2.Use the final element of the array as the pivot element.
3.Use the median of first, middle and last elements of the array as the pivot element.
"""

"""
Change A by the pivot_type.
"""
def choose_pivot( A, p, r, pivot_type ):
    #  The default partition chooses the first element as pivot.
    if pivot_type == "first":
        pass
    #  Choosing the last element as pivot.
    elif pivot_type == "last":
        A[p], A[r] = A[r], A[p]
    #  Choosing the median of first, middle and last elements as pivot.
    else:
        mid = (p + r) // 2
        tmp = [A[p], A[mid], A[r]]
        tmp.sort()
        #  Checking the median.
        if tmp[1] == A[mid]:
            A[p], A[mid] = A[mid], A[p]
        elif tmp[1] == A[r]:
            A[p], A[r] = A[r], A[p]
    return A


"""
The exquisite & classic function used in Quicksort.
Input @para: list A, the start & end index of A.
Output @para: the index of the pivot, the list satisfies A[p, ..., j - 2] <= A[j - 1] == pivot < A[j, ..., r].
"""
def partition( A, low, high ):

    #  pivot can be any element in A[low, ..., high], it's your choice.
    pivot = A[low]
    #  Initialize index j
    j = low + 1

    for i in range(low + 1, high + 1):
        #  A[...j] <= pivot
        if A[i] <= pivot:
            A[i], A[j] = A[j], A[i]
            j = j + 1

    #  A[...j] <= pivot, A[j + 1] == pivot, A[j + 2, ...] > pivot.
    A[low], A[j - 1] = A[j - 1], A[low]

    #  Return the index of pivot in A after partition.
    return j - 1


"""
The classic Quicksort.
Input @para: list A, the start & end index of A.
Output @para: the sorted A[low, ..., high].
"""
def quicksort( A, low, high, pivot_type ):

    #  Initialize the number of comparisons.
    count = high - low

    #  Base case: one or 0 element.
    #  Just return A and the number of comparisons = 0.
    if low >= high:
        return A, 0

    #  Change input list by exchanging A[p] and pivot.
    A = choose_pivot( A, low, high, pivot_type )

    #  Divide A into 2 parts.
    mid = partition( A, low, high )

    #  Recursively sort the 2 parts.
    A, left_count = quicksort( A, low, mid - 1, pivot_type )
    A, right_count = quicksort( A, mid + 1, high, pivot_type )

    #  Update the count by adding the numbers of comparisons in left and right parts.
    count += left_count + right_count

    return A, count


#  Drive Code
if __name__ == "__main__":
    # Test data.
    # input_data = [2, 8, 7, 1, 3, 5, 6, 4]
    # input_data = [13, 19, 9, 5, 12, 8, 7, 4, 21, 2, 6, 11]

    #  Load input test data from QuickSort.txt.
    input_data = []
    f = open("QuickSort.txt")
    line = f.readline()
    while line:
        input_data.append( int(line) )
        line = f.readline()
    f.close()

    #  Compute and print the output.
    pivots = [ "first", "last", "middle" ]
    for pivot in pivots:
        output, count = quicksort(input_data[:], 0, len(input_data) - 1, pivot)
        print("The number of comparisons when choosing the", pivot, "element as pivot: ", count)





