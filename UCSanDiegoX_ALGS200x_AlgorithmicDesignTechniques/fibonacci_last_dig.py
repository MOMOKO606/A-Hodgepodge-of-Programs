# Uses python3
# Programming Challenge 2-2: Last Digit of a Large Fibonacci Number.

"""
The iterative algorithm for finding the last digit of 𝑛th Fibonacci number 𝐹𝑛.
Input @para:  an integer 𝑛.
Output @para: the last digit of 𝑛th Fibonacci number 𝐹𝑛.
"""
def fib_last_dig( n ):
    #  Initialize the Fibonacci list.
    F=[ 0,1 ]

    #  Iteratively add the list to the nth element.
    for i in range( 2, n + 1):
        F.append( (F[ i - 1 ] + F[i - 2]) % 10 )
    #  Return the 𝑛th Fibonacci number 𝐹𝑛.
    return F[n]

#  Drive code.
if __name__ == "__main__":
    n = int(input())
    print( fib_last_dig(n) )