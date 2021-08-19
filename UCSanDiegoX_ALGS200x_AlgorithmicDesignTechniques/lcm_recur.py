# Uses python3
# Programming Challenge 2-4: Least Common Multiple.
import sys

"""
The recursive Greatest Common Divisor algorithm.
Input @para: integers a and b.
Output @para: the greatest common divisor of a and b.
"""
def gcd_recur( a, b ):

    #  Check out which is the bigger one and which is the smaller one.
    if a >= b:
        smaller = b
        bigger = a
    else:
        smaller = a
        bigger = b

    #  Base case,
    #  when the bigger number is a multiply of the smaller number, gcd is the smaller one.
    if smaller == 0:
        return bigger

    #  The recursive step
    #  Assume d is the gcd of a & b, then,
    #  bigger / d = ( k * smaller + reminder ) / d
    #             =   k * smaller / d + reminder / d
    #  therefore, gcd( bigger, smaller ) = gcd( smaller, reminder ).
    return gcd_recur( bigger % smaller, smaller )


def lcm_recur( a, b ):
    gcd = gcd_recur( a, b )
    return a * b // gcd


if __name__ == "__main__":
    input = sys.stdin.read()
    a, b = map(int, input.split())
    print(lcm_recur(a, b))