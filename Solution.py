from typing import Optional, List
import bisect
import math

#  Auxiliary functions and classes.
#  Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


#  Transfer an array to a linked list.
def array2Linkedlist( nums:List[int] ) -> Optional[ListNode]:
    head = cur = ListNode(nums[0])
    for i in range(1, len(nums)):
        cur.next = ListNode(nums[i])
        cur = cur.next
    return head


#  Transfer a linked list to an array.
def linkedlist2Array( head: Optional[ListNode] ) -> List[int]:
    cur = head
    res = []
    while cur:
        res.append(cur.val)
        cur = cur.next
    return res


#  Transfer a linked list to represent a number
#  The number in each node represents a digit of a number from left to right.
def linkedlist2num( head: Optional[ListNode] ) -> int:
    if head is None:
        return 0
    return head.val + 10 * linkedlist2num( head.next )


#  Transfer a number to a linked list in reverse order.
#  The number in each node represents a digit of a number in reverse order.
def num2list_rever( num:[int] ) -> List[int]:
    if num == 0: return [0]
    res = []
    while num:
        res.append( num % 10 )
        num //= 10
    return res


class Solution:
    """
    01. Two sum(Easy)

    Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
    You may assume that each input would have exactly one solution, and you may not use the same element twice.
    You can return the answer in any order.

    Example:
    Input: nums = [2,7,11,15], target = 9
    Output: [0,1]
    Output: Because nums[0] + nums[1] == 9, we return [0, 1].
    """
    def twoSum01(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums)):
            x = target - nums[i]
            for j in range(i + 1, len(nums)):
                if nums[j] == x:
                    return i, j


    def twoSum02(self, nums: List[int], target: int) -> List[int]:
        hashmap = {}
        for i in range(len(nums)):
            hashmap[nums[i]] = i
        for i in range(len(nums)):
            x = target - nums[i]
            if x in hashmap:
                return hashmap[x], i


    def twoSum03(self, nums: List[int], target:int) -> List[int]:
        hashmap = {}
        for i in range(len(nums)):
            x = target - nums[i]
            if x in hashmap:
                return hashmap[x], i
            hashmap[nums[i]] = i


    """
    02. Add two numbers(Medium)
    
    You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, 
    and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.
    You may assume the two numbers do not contain any leading zero, except the number 0 itself.
    
    Input: l1 = [2,4,3], l2 = [5,6,4]
    Output: [7,0,8]
    Explanation: 342 + 465 = 807.
    """
    def addTwoNumbers01(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        return array2Linkedlist(num2list_rever( linkedlist2num( l1 ) + linkedlist2num( l2 ) ))


    def addTwoNumbers02(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:

        ans = cur = ListNode(0)
        carry = 0

        while l1 or l2 or carry:

            if l1:
                carry += l1.val
                l1 = l1.next

            if l2:
                carry += l2.val
                l2 = l2.next


            cur.next = ListNode( carry % 10)
            cur = cur.next

            carry //= 10

        return ans.next

    """
    03.Longest Substring Without Repeating Characters(Medium)
    
    Given a string s, find the length of the longest substring without repeating characters.

    Input: s = "pwwkew"
    Output: 3
    Explanation: The answer is "wke", with the length of 3.
    Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
    """
    def lengthOfLongestSubstring(self, s: str) -> int:
        #  Initialization
        #  usedchar是一个dict（即哈希表），用来存储已出现过的字符的；
        #  left is the start index of the sliding window, -1是考虑到边界情况的初值;
        #  max_count tracks the result.
        usedchar, left, max_count = {}, -1, 0

        #  Go through the string one time.
        for right in range(len(s)):
            #  s[right] is already in the dict.
            #  特别注意要考虑到当字符已存在dict时，还需考虑是否是当前substring的重复字符。
            if s[right] in usedchar and left < usedchar [s[right]]:
                left = usedchar[s[right]]
            #  s[right] is not in the dict.
            else:
                #  Update the result.
                if right - left > max_count:
                    max_count = right - left
            #  无论如何，都要更新usedchar中的字符（如果没有就添加，如果有则因为sliding window要更新它的index）。
            usedchar[s[right]] = right

        return max_count


    """
    300.Medium
    
    Given an integer array nums, return the length of the longest strictly increasing subsequence.

    A subsequence is a sequence that can be derived from an array by deleting some or no elements 
    without changing the order of the remaining elements. For example, [3,6,2,7] is a subsequence 
    of the array [0,3,1,6,2,2,7].
    
    Input: nums = [10,9,2,5,3,7,101,18]
    Output: 4
    Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.
    """
    #  Solution1: the recursive lengthOfLIS with memo.
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        for i in range(n):
            self.lis(i, nums, dp)
        return max( dp )


    def lis(self, i: int, nums: List[int], dp: List[int]) -> int:
        if dp[i]:
            return dp[i]
        if i == 0:
            dp[i] = 1
            return 1

        max_len = 1
        for j in range(i):
            if nums[j] < nums[i]:
                max_len = max(max_len, self.lis(j, nums, dp) + 1  )
        dp[i] = max_len
        return max_len


    #  Solution2: the iterative dynamic programming algorithm of lengthOfLIS.
    def lengthOfLIS_iter( self, nums: List[int] ) -> int:
        n = len(nums)
        dp = [1] * n
        for i in range(n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)


    #  Solution3: the greedy lengthOfLIS.
    def lengthOfLIS_greedy( self, nums: List[int] ) -> int:
        ans = []
        for i in range(len(nums)):
            index = bisect.bisect_left(ans, nums[i])
            if index == len(ans) :
                ans.append(nums[i])
            else:
                ans[index] = nums[i]
        return len(ans)


    """
    322.Coin change(Medium)
    
    You are given an integer array coins representing coins of different denominations and an integer amount representing 
    a total amount of money.

    Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any 
    combination of the coins, return -1.

    You may assume that you have an infinite number of each kind of coin.
    
    Input: coins = [1,2,5], amount = 11
    Output: 3
    Explanation: 11 = 5 + 5 + 1
    """

    #  Solution1: the naive recursive algorithm.
    def coinChange_recur(self, coins: List[int], amount: int) -> int:

        def coinChange_Aux( coins: List[int], amount: int):
            if amount < 0:
                return float("inf")
            if amount == 0:
                return 0

            smallest = float("inf")
            for i in range(len(coins)):
                smallest = min( smallest, coinChange_Aux( coins, amount - coins[i]))

            return smallest + 1

        ans = coinChange_Aux(coins, amount)

        if ans == float("inf"):
            return -1
        else:
            return ans


    #  Solution2: the naive recursive algorithm with memo.
    def coinChange_recur_memo(self, coins: List[int], amount: int) -> int:

        memo = [float("nan")] * (amount + 1)
        memo[0] = 0

        def coinChange_Aux( coins: List[int], amount: int, memo):

            if amount < 0:
                return float("inf")

            if not math.isnan(memo[amount]):
                return memo[amount]


            smallest = float("inf")
            for i in range(len(coins)):
                smallest = min( smallest, coinChange_Aux( coins, amount - coins[i], memo) )

            memo[amount] = smallest + 1
            return smallest + 1

        ans = coinChange_Aux(coins, amount, memo)

        if ans == float("inf"):
            return -1
        else:
            return ans


    #  Solution3: the iterative dynamic programming algorithm.
    def coinChange_iter(self, coins: List[int], amount: int) -> int:
        memo = [amount + 1] * (amount + 1)
        memo[0] = 0

        for i in range(1, amount + 1):
            for j in range(len(coins)):
                if i - coins[j] >= 0:
                    memo[i] = min(memo[i], memo[i -coins[j]] + 1)

        return -1 if memo[amount] == amount + 1 else memo[amount]


    """
    62.Unique Paths(Medium)

    A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

    The robot can only move either down or right at any point in time. 
    The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

    How many possible unique paths are there?

    Input: m = 3, n = 2
    Output: 3
    Explanation:
    From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
    1. Right -> Down -> Down
    2. Down -> Down -> Right
    3. Down -> Right -> Down
    """

    #  Solution1: the naive recursive algorithm.
    def uniquePaths_recur(self, m: int, n: int) -> int:
        if m == 0 or n == 0:
            return 0
        if m == 1 or  n == 1:
            return 1
        return self.uniquePaths_recur(m - 1, n) + self.uniquePaths_recur(m, n - 1)


    #  Solution2: the recursive algorithm with memo.
    def uniquePaths_recur_memo(self, m: int, n: int) -> int:

        def uniquePath_Aux( m: int, n: int, memo: List[int] ) -> int:
            if memo[m][n] > 0:
                return memo[m][n]
            if m == 0 or n == 0:
                memo[m][n] = 0
                return 0
            if m == 1 or n == 1:
                memo[m][n] = 1
                return 1
            return uniquePath_Aux(m - 1, n, memo) + uniquePath_Aux(m, n - 1, memo)

        memo = [[-1 for j in range(n + 2)] for i in range(m + 2)]
        return uniquePath_Aux(m, n, memo)


    #  Solution3: the iterative dynamic programming algorithm.
    def uniquePaths(self, m: int, n: int) -> int:

        memo = [[0 for i in range(n + 1)] for j in range(m + 1)]
        memo[1][1] = 1

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if i != 1 or j != 1:
                    memo[i][j] = memo[i - 1][j] + memo[i][j - 1]
        return memo[m][n]


    """
    55.Jump Game(Medium)

    You are given an integer array nums. You are initially positioned at the array's first index, 
    and each element in the array represents your maximum jump length at that position.

    Return true if you can reach the last index, or false otherwise.

    Input: nums = [2,3,1,1,4]
    Output: true
    Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
    
    Input: nums = [3,2,1,0,4]
    Output: false
    Explanation: You will always arrive at index 3 no matter what. Its maximum jump length is 0, 
    which makes it impossible to reach the last index.
    """

    #  Solution1: the naive recursive algorithm.
    def canJump_recur(self, nums: List[int]) -> bool:

        def canJump_Aux(nums: List[int], i: int) -> bool:
            if i == 0:
                return True

            for j in range(i):
                if i - j <= nums[j]:
                    if canJump_Aux(nums, j):
                        return True
            return False

        return canJump_Aux( nums, len(nums) - 1)


    #  Solution2: the recursive algorithm with memo.
    def canJump_recur_memo(self, nums: List[int]) -> bool:

        def canJump_Aux(nums: List[int], i: int, memo: List[bool]) -> bool:

            if type(memo[i]) is bool:
                return memo[i]

            if i == 0:
                memo[i] = True
                return True

            for j in range(i):
                if i - j <= nums[j]:
                    if canJump_Aux(nums, j, memo):
                        memo[i] = True
                        return True

            memo[i] = False
            return False

        memo = [0] * len(nums)
        return canJump_Aux(nums, len(nums) - 1 , memo)


    #  Solution3: the iterative dynamic programming algorithm.
    def canJump_iter(self, nums: List[int]) -> bool:
        n = len(nums)
        ans = [False] * n
        ans[0] = True
        for i in range(1, len(nums)):
            for j in range(i):
                if i - j <= nums[j] and ans[j] is True:
                    ans[i] = True
                    break
        return ans[n - 1]


    #  Solution4: the greedy algorithm.
    def canJump_greedy(self, nums: List[int]) -> bool:
        reach = 0
        n = len(nums)
        for i in range(n):
            if i <= reach:
                reach = max(reach, nums[i] + i)
            if reach >= n - 1:
                return True
        return False


    """
    125.Valid Palindrome(Easy)

    A phrase is a palindrome if, 
    after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, 
    it reads the same forward and backward. Alphanumeric characters include letters and numbers.

    Given a string s, return true if it is a palindrome, or false otherwise.

    Example 1:
    Input: s = "A man, a plan, a canal: Panama"
    Output: true
    Explanation: "amanaplanacanalpanama" is a palindrome.
    
    Example 2:
    Input: s = "race a car"
    Output: false
    Explanation: "raceacar" is not a palindrome.
    
    Example 3:
    Input: s = " "
    Output: true
    """
    #  Solution 1.
    def isPalindrome01(self, s: str) -> bool:
        #  Put s into lowercase.
        s = s.lower()

        #  Delete all the chars that are not alpha nor numbers.
        s = [s[i] for i in range(len(s)) if s[i].isalnum()]

        #  Compare by slicing.
        if s[::-1] == s:
            return True
        return False


    #  Solution 2.
    def isPalindrome02(self, s: str) -> bool:
        s = s.lower()

        i = 0
        j = len(s) - 1

        while i < j:
            if not s[i].isalnum():
                i += 1
                continue

            if not s[j].isalnum():
                j -= 1
                continue

            if s[i] != s[j]:
                return False
            else:
                i += 1
                j -= 1

        return True


    """
    283. Move Zeros(Easy)
    
    Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements.

    Note that you must do this in-place without making a copy of the array.
    
    Example
    Input: nums = [0,1,0,3,12]
    Output: [1,3,12,0,0]
    """
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        j = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[j] = nums[i]
                j += 1

        for i in range(j, len(nums)):
            nums[i] = 0














#  Drive code.
if __name__ == "__main__":
    #  Create an instance
    S = Solution()

    #  Data: lists.
    l0 = [2,7,11,15]
    l1_list = [9,9,9,9,9,9,9]
    l2_list = [9,9,9,9]
    string01 = "abcabcbb"
    string02 = "bbbbb"
    string03 = "pwwkew"
    string04 = "dvdf"
    string05 = ""
    string06 = " "
    string07 = "tmmzuxt"

    string08 = "A man, a plan, a canal: Panama"
    string09 = "race a car"

    l3 = [1, 5, 2, 4, 3]
    l4 = [3, 2, 4, 1, 7, 6, 10]
    l5 = [10,9,2,5,3,7,101,18]
    l6 = [2, 3, 1, 1, 4]
    l7 = [3, 2, 1, 0, 4]
    l8 = [0, 2, 3]
    l9 = [0,1,0,3,12]

    coins = [1,2,5]
    amount = 11

    #  Data: Linked lists.
    l1 = array2Linkedlist(l1_list)
    l2 = array2Linkedlist(l2_list)

    print(S.twoSum03(l0, 9))   #  Leetcode, 01
    print(linkedlist2Array(S.addTwoNumbers02(l1, l2)))  #  Leetcode, 02
    print(S.lengthOfLongestSubstring( string01 ))  #  Leetcode, 03
    print(S.lengthOfLIS(l3))  # Leetcode, 300
    print(S.lengthOfLIS_greedy(l5))  # Leetcode, 300
    print(S.coinChange_iter([2], 3))  # Leetcode, 322
    print(S.uniquePaths(7, 3))  # Leetcode, 62
    print(S.canJump_greedy([1]))  # Leetcode, 55
    print(S.isPalindrome01(string08))  # Leetcode, 125
    print(S.isPalindrome02(".,"))  # Leetcode, 125
    print("--------------------------------")
    S.moveZeroes(l8)
    print(l8)  # Leetcode 283