# Uses python3
# Programming Challenge 2-5: Compute a Huge Fibonacci Number Modulo m.
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
Given two integers 𝑛 and 𝑚, output 𝐹𝑛 mod 𝑚 (that is, the remainder of 𝐹𝑛 when divided by 𝑚).
Using extra memory of list.
Input @para: two integers 𝑛 and 𝑚, 1 ≤ n ≤ 1018, 2 ≤ 𝑚 ≤ 105.
Output: 𝐹𝑛 mod 𝑚.
"""
def fibonacci_mod_m(n, m):
    #  Get the correspond Pisano period.
    pisano_sequence = pisano_period( m )
    #  get the length of mth Pisano period.
    factor = len(pisano_sequence)
    #  Fn mod m = reminder mod m = reminder th element in Pisano period.
    return pisano_sequence[n % factor]


#  Drive code.
if __name__ == "__main__":
    input = sys.stdin.read()
    n, m = map(int, input.split())
    print(fibonacci_mod_m(n, m))