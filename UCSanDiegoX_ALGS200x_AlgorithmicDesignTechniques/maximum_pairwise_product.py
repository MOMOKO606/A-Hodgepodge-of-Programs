# Uses python3
# Programming Challenge 1-2: Maximum Pairwise Product

"""
Find the maximum product of two distinct numbers in a sequence of non-negative integers.
Find the largest and the second largest element in A, then return the product of them.

Input @para: list A and the length of A
Output @para: the maximum product of 2 elements in A.
"""
def max_pair_product( A, n ):

    #  Get the largest element and store it in A[0].
    k = 0
    largest = A[k]
    for i in range( 1, n ):
        if A[i] >= largest:
            k = i
            largest = A[i]
    A[k], A[0] = A[0], A[k]

    #  Get the second largest element and store it in A[1].
    k = 1
    sec_largest = A[k]
    for i in range( 2, n ):
        if A[i] >= sec_largest:
            k = i
            sec_largest = A[i]
    A[k], A[1] = A[1], A[k]

    return A[0] * A[1]


if __name__ == "__main__":
    #  Create the input parameters.
    n = int(input())
    a = [int(x) for x in input().split()]
    #  Sentinel.
    assert (len(a) == n)

    print( max_pair_product( a, n ))

