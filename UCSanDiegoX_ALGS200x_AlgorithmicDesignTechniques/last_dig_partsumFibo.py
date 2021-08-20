# Uses python3
# Programming Challenge 2-7: Compute the Last Digit of a Partial Sum of Fibonacci Numbers.

import sys


"""
Compute and return the Pisano period corresponded to m.
Input: integer 𝑚, 2 ≤ 𝑚 ≤ 105.
Output: m's Pisano period.
"""
def pisano_period( m ):
    #  Initialize the fibonacci sequence and the Pisano period.
    fibo = [0, 1]
    pisano = [0, 1]

    #  Remember the previous element in Pisano period.
    previous = 1

    #  Computer the Pisano period.
    #  key: 0, 1 are the first two elements of all period.
    for i in range(2, m * m + 1):
        fi = fibo[i - 2] + fibo[i - 1]
        pi = fi % m

        #  Sentinel, break the loop and return.
        if pi == 1 and previous == 0:
            return pisano[:i - 1]

        #  Update the fibonacci sequence and the Pisano period.
        #  Update the previous element.
        previous = pi
        fibo.append(fi)
        pisano.append(pi)


"""
Compute and return the length of the Pisano period corresponded to m.
Input @para: integer 𝑚, 2 ≤ 𝑚 ≤ 105.
Output: the length of the m's Pisano period.
"""
def length_pisano_period( m ):

    f_previous = 0
    f_current = 1

    p_previous = 0
    p_current = 1

    #  Check the Pisano period.
    #  key: 0, 1 are the first two elements of all period.
    for i in range( 2, m * m + 1):

        f_previous, f_current = f_current, f_current + f_previous

        p_previous = p_current
        p_current = f_current % m

        #  Sentinel, break the loop and return the length.
        if p_current == 1 and p_previous == 0:
            return i - 1

"""
Find the last digit of a partial sum of Fibonacci numbers: 𝐹𝑚 + 𝐹𝑚+1 + · · · + 𝐹𝑛.
Input @para: two non-negative integers fro𝑚_ and to. 
Output: the last digit of 𝐹fro𝑚_ + 𝐹fro𝑚_+1 + · · · + 𝐹to.

Key: 
(Fi + Fi+1 + ,..., + Fj) mod 10
= Fi mod 10 + Fi+1 mod 10 + ,..., + Fj mod 10
= sum of the last digit in Pisano period
"""
def fibonacci_partial_sum(from_, to):

    #  Get the length and sequence of the Pisano_period for 10.
    l = length_pisano_period( 10 )
    p_sequence = pisano_period( 10 )

    #  Get the indices.
    r1 = from_ % l
    k1 = (from_ - r1) / l

    r2 = to % l
    k2 = (to - r2) / l

    # k2 - k1 means how many times we need to go through the Pisano_period.
    if k2 - k1 == 0:
        total = sum(p_sequence[r1: r2 + 1]) % 10
    else:
        total = (sum(p_sequence[:]) % 10 * (k2 - k1)) % 10 + sum(p_sequence[r1:]) % 10 + sum(p_sequence[:r2+1]) % 10

    return int(total)


#  Drive code.
if __name__ == '__main__':
    input = sys.stdin.read()
    from_, to = map(int, input.split())
    print( fibonacci_partial_sum(from_, to) )