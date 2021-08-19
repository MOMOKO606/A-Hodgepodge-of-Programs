# Uses python3
import sys


"""
Compute and return the length of the Pisano period corresponded to m.
Input @para: two integers 𝑛 and 𝑚, 1 ≤ n ≤ 1018, 2 ≤ 𝑚 ≤ 105.
Output: the length of the m's Pisano period.
"""
def length_pisano_period( n, m ):

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
Given two integers 𝑛 and 𝑚, output 𝐹𝑛 mod 𝑚 (that is, the remainder of 𝐹𝑛 when divided by 𝑚).
Using several variables.
Input @para: two integers 𝑛 and 𝑚, 1 ≤ n ≤ 1018, 2 ≤ 𝑚 ≤ 105.
Output: 𝐹𝑛 mod 𝑚.
"""
def fibonacci_modm(n, m):
    #  Get the correspond Pisano period.
    factor = length_pisano_period(n, m)

    #  Fn mod m = reminder mod m = reminder th element in Pisano period.
    reminder = n % factor

    f_previous = 0
    f_current = 1

    if reminder == 0:
        return 0

    if reminder == 1:
        return 1

    for i in range( 2, reminder + 1):
        f_previous, f_current = f_current, f_current + f_previous

    return f_current % m



#  Drive code.
if __name__ == "__main__":
    input = sys.stdin.read()
    n, m = map(int, input.split())
    print(fibonacci_modm(n, m))