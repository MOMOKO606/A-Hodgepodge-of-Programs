class Solution:
    #  1137 (easy)
    def tribonacci(self, n: int) -> int:
        if n < 2: return n
        if n == 2: return 1
        f0, f1, f2 = 0, 1, 1
        for _ in range(n - 2):
            f0, f1, f2 = f1, f2, f0 + f1 + f2
        return f2

if __name__ == "__main__":
    S = Solution()
    #  1137 (easy)
    print(S.tribonacci(25))


