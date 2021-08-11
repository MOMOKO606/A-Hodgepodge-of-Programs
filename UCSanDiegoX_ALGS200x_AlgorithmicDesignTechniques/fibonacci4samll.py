# Uses python3
# Programming Challenge 2-1: Fibonacci Number, Compute a Small Fibonacci Number.

"""
The iterative algorithm for finding the 𝑛th Fibonacci number 𝐹𝑛.
Input @para:  an integer 𝑛.
Output @para: the 𝑛th Fibonacci number 𝐹𝑛.
"""
def fib_iter( n ):
    #  Initialize the Fibonacci list.
    F=[ 0,1 ]

    #  Iteratively add the list to the nth element.
    for i in range( 2, n + 1):
        F.append( F[ i - 1 ] + F[i - 2])
    #  Return the 𝑛th Fibonacci number 𝐹𝑛.
    return F[n]


if __name__ == "__main__":
    n = int(input())
    print( fib_iter(n) )