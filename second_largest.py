import math


"""
Get the left child index of index i in a heap.
Input @para: index i.
Output i's left index.
"""
def left( i ):
    #  i << 1 equals to i * 2
    #  << is bit operation which means move x digit(s) to the left.
    #  i << 1 will first transfer i to binary format, then move one digit to the left.
    # "move one digit to the left" means "i * 2".
    # "move n digits to the right" means "i * 2 ^ n", e.g. 1 << 10 is 1024.
    return (i << 1) + 1


"""
Get the right child index of index i in a heap.
Input @para: index i.
Output i's right index.
"""
def right( i ):
    #  i << 1 equals to i * 2
    #  << is bit operation which means move x digit(s) to the left.
    #  i << 1 will first transfer i to binary format, then move one digit to the left.
    # "move one digit to the left" means "i * 2".
    # "move n digits to the right" means "i * 2 ^ n", e.g. 1 << 10 is 1024.
    return i + 1 << 1


"""
Find the second largest element in array A within n + lgn - 2 comparisons.
Building a heap-like data structure, compare the elements in pairs and find the largest element, O(n).
Tracking the path of the comparisons of the largest element.
The second largest element must be on one of the (lgn - 1) nodes that oppose to the path.
Therefore, there is  n + lgn - 2 comparisons maximum when n is a power of 2.

Input @para: list A.
Output @pare: the second largest element in list A.
"""
def second_largest( A ):
    #  Get the length of A.
    n = len( A )

    #  Set Sentinel.
    #  & means 位运算"与"，&对比的两个位都是1则为1，否则为0。
    #  2的几次幂0除外，用二进制表示第一位全是1，后面全是0。
    #  跟2的几次幂对应的数相反，比他小一个数全是第一位为0后面全为1。
    #  所以，如果n是a power of 2，n & n-1 equals 0.
    if n & (n - 1) != 0:
        return "n must be a power of 2!"

    #  Initialize the heap-like data structure.
    tmp = A[:]  #  The bottom level of the heap.
    #  The heap-like data structure.
    heapA = A[:]

    #  Build the heap-like data structure from list A.
    for j in range( int(math.log2(n)) ):
        #  The upper level of the heap.
        upper = []
        for i in range( 0, n, 2 ):
            upper.append(max(tmp[i], tmp[i + 1]))

        #  Update parameters.
        n = int( n / 2 )
        tmp = upper

        heapA = upper + heapA

    #  Tracking the path and find the second largest element.
    #  The potential second largest elements are stored in tmp.
    largest = heapA[0]
    heap_size = len(heapA)
    i = 0
    tmp =[]

    while i < heap_size // 2:
        l = left(i)
        r = right(i)
        if heapA[l] != largest:
            tmp.append(heapA[l])  # the potential second largest element.
            i = r           # Tracking down.
        else:
            tmp.append(heapA[r])  # the potential second largest element.
            i = l  # Tracking down.

    #  Brute force compare every two elements on the path stored in tmp.
    #  The largest element in tmp is the second largest element in A.
    largest = tmp[0]
    for i in range( 1, len(tmp), 1 ):
        if tmp[i] > largest:
            largest = tmp[i]

    return largest



# Drive code
if __name__ == "__main__":

    A1 = [1, 7, 2, 11, 9, 5, 4, 10]
    A2 = [11, 10, 2, 9, 5, 4, 1, 7]

    #  Test list A1.
    print("The second largest element in A1 is ", second_largest(A1), '\n')
    #  Test list A2.
    print("The second largest element in A2 is ", second_largest(A2), '\n')
