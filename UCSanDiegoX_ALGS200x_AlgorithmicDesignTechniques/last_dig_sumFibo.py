# Uses python3
# Programming Challenge 2-6: Compute the Last Digit of a Sum of Fibonacci Numbers.


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
Given two integers 𝑛 and 𝑚, output 𝐹𝑛 mod 𝑚 (that is, the remainder of 𝐹𝑛 when divided by 𝑚).
Using several variables.
Input @para: two integers 𝑛 and 𝑚, 1 ≤ n ≤ 1018, 2 ≤ 𝑚 ≤ 105.
Output: 𝐹𝑛 mod 𝑚.

Key: for each m, 𝐹𝑛 mod 𝑚 generates a periodical sequence.
we need to determine the periodical sequence or at least know its length first.
"""
def fibonacci_mod_m(n, m):
    #  Get the correspond Pisano period.
    factor = length_pisano_period( m )

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



"""
Given an integer 𝑛, find the last digit of the sum 𝐹0 +𝐹1 +···+𝐹𝑛.
Input:  Integer 𝑛.
Output: the last digit of 𝐹0 + 𝐹1 + · · · + 𝐹𝑛.

Note: O(n) is way too slow, which means we cannot use loop to check every Fibonacci number.

Key1:
------------------------------------------
    Index 0, 1, 2,  3, 4,  5,  6,  7,  8,...
------------------------------------------
    Fi    0, 1, 1,  2, 3, |5,  8,  13, 21,...
------------------------------------------
    Sum   0, 1, 2, |4, 7,  12, 20, 33, 54,...
------------------------------------------
Sum[j] = Sum[Fi[0] + Fi[1] + ..., + Fi[j]] = Fi[ j + 2 ] - 1, when j > 2.

Key2:
The last digit = Sum[j] % 10 = Fi[ j + 2 ] % 10 - 1 % 10 = Fi[ j + 2 ] % 10 - 1
               = Fi[ j + 2 ] mod 10 -1
"""
def fib_sum_last_dig( n ):

    #  Base case, n <= 2
    if n == 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 2

    ans = fibonacci_mod_m(n + 2, 10)
    #  The special case, when the last digit is 0, 0 - 1 should be digit 9.
    if ans == 0:
        return 9
    #  return Fi[ n + 2 ] mod 10 -1 when the last digit isn't 0.
    else:
        return ans - 1


#  Drive code.
if __name__ == "__main__":
    n = int(input())
    print( fib_sum_last_dig( n ) )