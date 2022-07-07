class Solution:
    #  1137.N-th Tribonacci Number(easy)
    def tribonacci(self, n: int) -> int:
        if n < 2: return n
        if n == 2: return 1
        f0, f1, f2 = 0, 1, 1
        for _ in range(n - 2):
            f0, f1, f2 = f1, f2, f0 + f1 + f2
        return f2

    #  125.Valid Palindrome(easy)
    def isPalindrome(self, s: str) -> bool:
        #  Turn into lowercase.
        s = s.lower()
        #  Only leave numbers and letters.
        s = [s[i] for i in range(len(s)) if s[i].isalnum()]
        #  Compare.
        return s[::-1] == s


if __name__ == "__main__":
    S = Solution()
    #  1137 (easy)
    print(S.tribonacci(25))
    #  125 (easy)
    print(S.isPalindrome(" "))
    print(S.isPalindrome("race a car"))
    print(S.isPalindrome("A man, a plan, a canal: Panama"))


