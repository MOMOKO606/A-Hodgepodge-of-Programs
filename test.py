# Uses python3
def fib_iter( n ):
    F=[ 0,1 ]
    for i in range( 2, n + 1):
        F.append( F[ i - 1 ] + F[i - 2])

    return F[n]


if __name__ == "__main__":
    n = int(input())
    print( fib_iter(n) )

