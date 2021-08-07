"""
given a sorted array A of n distinct integers which can be positive, negative, or zero.
You want to decide whether or not there is an index i such that A[i] = i.

Idea: Divide-and-conquer
Input @para: list A and its start and end indices.
Outpyt: the index which A[i] = i or None.
"""
def get_index( A, low, high ):

    #  Base case, no such element in A.
    if low > high:
        return None

    mid = ( low + high ) // 2
    #  Base case, find the index directly.
    if A[mid] == mid:
        return mid
    #  The potential index might be in the right part only.
    elif A[mid] < mid:
        return get_index( A, mid + 1, high)
    #  The potential index might be in the left part only.
    else:
        return get_index( A, low, mid - 1)




if __name__ == "__main__":

    #  Drive code.
    A1 = [-9, -6, -3, 0, 1, 3, 4, 7]
    A2 = [-5, 0, 2, 5, 9, 10, 11]

    #  Find the index i which A[i] = i .
    print("The index of list A1 is", get_index( A1, 0, len(A1) ), '\n')
    print("The index of list A2 is", get_index( A2, 0, len(A2) ), '\n')