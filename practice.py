import math
from functools import cache
from typing import List


class Solution:
    #  1137. N-th Tribonacci Number(easy)  一
    def tribonacci(self, n: int) -> int:
        if n < 2: return n
        if n == 2: return 1
        f0, f1, f2 = 0, 1, 1
        for _ in range(n - 2):
            f0, f1, f2 = f1, f2, f0 + f1 + f2
        return f2

    #  125. Valid Palindrome(easy)  一
    def isPalindrome(self, s: str) -> bool:
        s = s.lower()
        n = len(s)
        i, j = 0, n - 1
        while True:
            while i < n and not s[i].isalnum():
                i += 1
            while j >= 0 and not s[j].isalnum():
                j -= 1
            if i < n and j >= 0 and s[i] == s[j]:
                i += 1
                j -= 1
            elif i > j:
                return True
            else:
                return False

    #  283. Move zeros(easy)  一
    def moveZeroes(self, nums: List[int]) -> None:
        i = 0
        for j in range(len(nums)):
            if nums[j] != 0:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1

    #  11. Container With Most Water (Medium) 一
    def maxArea(self, height: List[int]) -> int:
        i, j = 0, len(height) - 1
        largest = -math.inf
        while i < j:
            largest = max(largest, (j - i) * min(height[i], height[j]))
            if height[i] <= height[j]:
                i += 1
            else:
                j -= 1
        return largest

    #  70. Climbing Stairs (easy)  一
    def climbStairs(self, n: int) -> int:
        if n < 3: return n
        f1, f2 = 1, 2
        for i in range(n - 2):
            f2, f1 = f1 + f2, f2
        return f2

    #  1. Two Sum (easy)
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums) - 1):
            remain = target - nums[i]
            for j in range(i + 1, len(nums)):
                if nums[j] == remain:
                    return [nums[i], nums[j]]



if __name__ == "__main__":
    S = Solution()

    #  1137 (easy)
    print(S.tribonacci(25))

    #  125 (easy)
    print(S.isPalindrome(" "))
    print(S.isPalindrome(".,"))
    print(S.isPalindrome("race a car"))
    print(S.isPalindrome("A man, a plan, a canal: Panama"))

    #  283 (easy)
    nums01 = [0, 1, 0, 3, 12]
    nums02 = [0]
    S.moveZeroes(nums01)
    S.moveZeroes(nums02)
    print(nums01)
    print(nums02)

    #  11 （medium）
    print(S.maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7]))
    print(S.maxArea([1, 1]))

    #  70 (easy)
    print(S.climbStairs(0))
    print(S.climbStairs(2))
    print(S.climbStairs(3))

    #  1 (easy)
    print("---------------------------")
    print(S.twoSum([2, 7, 11, 15], 9))
    print(S.twoSum([3, 2, 4], 6))
    print(S.twoSum([3, 3], 6))
