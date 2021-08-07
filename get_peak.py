"""
given a unimodal array of n distinct elements,
meaning that its entries are in increasing order up until its maximum element,
after which its elements are in decreasing order.
Give an algorithm to compute the maximum element that runs in O(log n) time.

Idea: Divide-and-Conquer.
Split the array into 2 parts, compare the elements in the middle to determine the trend in the middle.
Input @para: list A and its start and end indices.
Output: the value on the peak.
"""
def get_peak( A, low, high ):
    #  Base case, only one element left.
    if low == high:
        return A[low]

    mid = ( low + high ) // 2
    if A[mid] > A[mid + 1]:
        return get_peak(A, low, mid)
    else:
        return get_peak(A, mid + 1, high)


if __name__ == "__main__":

    #  Drive code.
    A1 = [1, 2, 6, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    A2 = [1, 2, 3, 4, 5, 9, 5, 4, 3, 2, 2, 1]

    #  Find the peak of list A1.
    print("The peak of list A1 is", get_peak( A1, 0, len(A1) ), '\n')
    #  Find the peak of list A2.
    print("The peak of list A2 is", get_peak( A2, 0, len(A2) ), '\n')
