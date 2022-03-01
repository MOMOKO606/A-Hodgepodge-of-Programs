from typing import List, Optional
import bisect, math, string
from functools import cache


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def array2Linkedlist(nums: List[int]) -> Optional[ListNode]:
    dummy = cur = ListNode()
    for num in nums:
        cur.next = ListNode(num)
        cur = cur.next
    return dummy.next


def linkedlist2Array(head: Optional[ListNode]) -> List[int]:
    cur = head
    ans = []
    while cur:
        ans.append(cur.val)
        cur = cur.next
    return ans


def printLinkedlist(head: Optional[ListNode]) -> None:
    print(linkedlist2Array(head))


class Solution:

    #  Leetcode 01
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        map = {}
        for i, num in enumerate(nums):
            key = target - num
            if key in map:
                return [map[key], i]
            map[num] = i

    #  Leetcode 02
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = cur = ListNode()
        carry = 0
        while l1 or l2 or carry:

            if l1:
                carry += l1.val
                l1 = l1.next
            if l2:
                carry += l2.val
                l2 = l2.next

            cur.next = ListNode(carry % 10)
            cur = cur.next
            carry //= 10
        return dummy.next

    # def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    #
    #     def ll2ReverseNum(head: Optional[ListNode]) -> int:
    #         cur = head
    #         ans = 0
    #         factor = 1 / 10
    #         while cur:
    #             factor *= 10
    #             ans += cur.val * factor
    #             cur = cur.next
    #         return ans
    #
    #     def num2Reversell( num: int ) -> Optional[ListNode]:
    #         if num == 0: return ListNode()
    #         cur = dummy = ListNode()
    #         while num:
    #             cur.next = ListNode( num % 10)
    #             cur = cur.next
    #             num //= 10
    #         return dummy.next
    #
    #
    #     num1 = ll2ReverseNum(l1)
    #     num2 = ll2ReverseNum(l2)
    #     return num2Reversell(num1 + num2)

    #  Leetcode 03
    def lengthOfLongestSubstring(self, s: str) -> int:
        l, max_count, usedchar = -1, 0, {}
        for r in range(len(s)):
            if s[r] in usedchar and usedchar[s[r]] > l:
                l = usedchar[s[r]]
            else:
                max_count = max(max_count, r - l)
            usedchar[s[r]] = r
        return max_count

    #  Leetcode 11
    def maxArea(self, height: List[int]) -> int:
        area, i, j = 0, 0, len(height) - 1
        while i < j:
            area = max(area, min(height[i], height[j]) * (j - i))
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
        return area

    #  Leetcode 15
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        ans = []
        for i in range(n):
            #  Avoid duplicates
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            l = i + 1
            r = n - 1
            while l < r:
                res = nums[i] + nums[l] + nums[r]
                if res == 0:
                    ans.append([nums[i], nums[l], nums[r]])
                    l += 1
                    r -= 1
                    while l < n and nums[l] == nums[l - 1]:
                        l += 1
                    while r > -1 and nums[r] == nums[r + 1]:
                        r -= 1
                elif res < 0:
                    l += 1
                else:
                    r -= 1
        return ans

    #  Leetcode 21
    def mergeTwoLists(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        if not l1 or not l2:
            return l1 or l2
        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2

    # def mergeTwoLists(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    #     dummy = cur = ListNode()
    #     while l1 and l2:
    #         if l1.val < l2.val:
    #             cur.next = l1
    #             l1 = l1.next
    #         else:
    #             cur.next = l2
    #             l2 = l2.next
    #         cur = cur.next
    #
    #     cur.next = l1 or l2
    #     return dummy.next

    #  Leetcode 26
    def removeDuplicates(self, nums: List[int]) -> int:
        j = -1
        for i in range(len(nums)):
            if nums[i] != nums[i - 1] or j < 0:
                j += 1
                nums[j] = nums[i]
        return j + 1

    #  Leetcode 55
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        reach = 0
        for j in range(n):
            if j > reach:
                return False
            if reach >= n - 1:
                return True
            reach = max(reach, j + nums[j])

    # def canJump(self, nums: List[int]) -> bool:
    #     n = len(nums)
    #     ans = [False] * n
    #     ans[0] = True
    #     for i in range(1, n):
    #         for j in range(i):
    #             if i - j <= nums[j] and ans[j] is True:
    #                 ans[i] = True
    #                 break
    #     return ans[n - 1]

    # def canJump(self, nums:List[int]) -> bool:
    #
    #     def _canJump(nums: List[int], memo: List[int]) -> bool:
    #         n = len(nums)
    #         #  Base case:
    #         if type(memo[n - 1]) is bool:
    #             return memo[n - 1]
    #
    #         for i in range(n - 1):
    #             if nums[i] + i >= n - 1:
    #                 if _canJump(nums[:i + 1], memo):
    #                     memo[n - 1] = True
    #                     return True
    #         memo[n - 1] = True
    #         return False
    #
    #     memo = [0] * len(nums)
    #     memo[0] = True
    #     return _canJump( nums, memo)

    #  Leetcode 45
    def jump(self, nums: List[int]) -> int:
        reach, nextReach, steps = 0, 0, 0
        for i in range(len(nums)):
            nextReach = max(nextReach, i + nums[i])
            if nextReach >= len(nums) - 1: return steps + 1
            if i == reach:
                steps += 1
                reach = nextReach

    #  Leetcode 62
    #  The iterative dp solution.
    def uniquePaths(self, m: int, n: int) -> int:
        prev = [1] * n
        curr = [0] * n
        curr[0] = 1
        for _ in range(m - 1):
            for j in range(1, n):
                curr[j] = curr[j - 1] + prev[j]
            prev = curr
        return curr[-1]

    #  The recursive solution with a self-made memo.
    # def uniquePaths(self, m: int, n:int) -> int:
    #
    #     def _uniquePaths( m: int, n: int, memo:List[List[int]]) -> int:
    #         if memo[m - 1][n - 1] > 0:
    #             return memo[m - 1][n - 1]
    #
    #         if m == 1 or n == 1:
    #             memo[m - 1][n - 1] = 1
    #             return 1
    #
    #         return _uniquePaths( m - 1, n, memo) + _uniquePaths( m, n - 1, memo)
    #
    #     memo = [[0 for j in range(n)] for i in range(m)]
    #     return _uniquePaths(m, n, memo)

    #  The recursive solution with a cache.
    # def uniquePaths(self, m: int, n: int) -> int:
    #     @cache
    #     def _uniquePaths( i, j ):
    #         if i == 0 and j == 0: return 1
    #         if i < 0 or j < 0: return 0
    #         return _uniquePaths( i - 1, j ) + _uniquePaths( i, j - 1 )
    #     return _uniquePaths( m - 1, n - 1 )

    #  Leetcode 66
    def plusOne(self, digits: List[int]) -> List[int]:

        if len(digits) < 1:
            return [1]

        if digits[-1] + 1 < 10:
            digits[-1] += 1
            return digits

        return self.plusOne(digits[:-1]) + [0]

    # def plusOne(self, digits: List[int]) -> List[int]:
    #     for j in reversed(range(len(digits))):
    #         if digits[j] + 1 < 10:
    #             digits[j] += 1
    #             return digits
    #         digits[j] = 0
    #     return [1] + digits

    #  Leetcode 70
    #  The iterative solution.
    def climbStairs(self, n: int) -> int:
        if n < 3: return n
        f1, f2, f3 = 1, 2, 3
        for _ in range(n - 2):
            f3 = f1 + f2
            f1, f2 = f2, f3
        return f3

    #  The recursive solution with a self-made memo.
    # def climbStairs(self, n: int) -> int:
    #
    #     def _climbStairs( n: int, memo: List[int]) -> int:
    #         if memo[n]:
    #             return memo[n]
    #         if n == 1:
    #             memo[n] = 1
    #             return 1
    #         if n == 2:
    #             memo[n] = 2
    #             return 2
    #         return _climbStairs( n - 1, memo ) + _climbStairs( n - 2, memo )
    #
    #     memo= [0] * (n + 1)
    #     return _climbStairs(n, memo)

    # #  The recursive solution with a cache.
    # @cache
    # def climbStairs(self, n: int) -> int:
    #     if n < 3: return n
    #     return self.climbStairs( n - 1 ) + self.climbStairs( n - 2 )

    #  Leetcode 80
    def removeDuplicates02(self, nums: List[int]) -> int:
        k = -1
        for j in range(len(nums)):
            if j < 2 or nums[j] > nums[k - 1]:
                k += 1
                nums[k] = nums[j]
        return k + 1

    #  Leetcode 82
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = dummy = ListNode()
        cur = dummy.next = head
        while cur:
            while cur.next and cur.val == cur.next.val:
                cur = cur.next
            if prev.next != cur:
                prev.next = cur.next
            else:
                prev = cur
            cur = cur.next
        return dummy.next

    #  Leetcode 83
    def deleteDuplicates03(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur = head
        while cur:
            while cur.next and cur.val == cur.next.val:
                cur.next = cur.next.next
            cur = cur.next
        return head

    #  Leetcode 88
    def merge(self, nums01: List[int], nums02: List[int], m: int, n: int) -> None:
        while m and n:
            if nums01[m - 1] > nums02[n - 1]:
                nums01[m + n - 1] = nums01[m - 1]
                m -= 1
            else:
                nums01[m + n - 1] = nums02[n - 1]
                n -= 1
        nums01[:n] = nums02[:n]

    #  Leetcode 125
    def isPalindrome(self, s: str) -> bool:
        s = s.lower()
        i, j = 0, len(s) - 1
        while i < j:
            if not s[i].isalpha():
                i += 1
                continue
            if not s[j].isalpha():
                j -= 1
                continue
            if s[i] != s[j]:
                return False
            i += 1
            j -= 1
        return True

    #  Leetcode 141
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        p1 = p2 = head
        while p1 and p2:
            p1 = p1.next
            p2 = p2.next.next
            if p1 == p2:
                return True
        return False

    #  Leetcode 142
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        p1 = p2 = head
        #  We can use p1 and p2 and p2.next as the while condition
        #  Also, we just need to make sure the faster pointer is legal
        #  which means the while condition can be simplified as while p2 and p2.next.
        #  Tricky!
        while p2 and p2.next:
            p1 = p1.next
            p2 = p2.next.next
            if p1 == p2:
                break
        else:
            return None
        p1 = head
        while p1 != p2:
            p1 = p1.next
            p2 = p2.next
        return p1

    #  Leetcode 189
    def rotate(self, nums: List[int], k: int) -> None:

        def _reverselist(arr: List[int]) -> List[int]:
            i = 0
            j = len(arr) - 1
            while i < j:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
                j -= 1
            return arr

        k = k % len(nums)
        nums = _reverselist(nums)
        nums[:k] = _reverselist(nums[:k])
        nums[k:] = _reverselist(nums[k:])

    #  Joseph-like method.
    # def rotate(self, nums: List[int], k: int) -> None:
    #     count = start = 0
    #     n = len(nums)
    #     while count < n:
    #         cur = start
    #         cur_val = nums[cur]
    #         while True:
    #             cur = (cur + k) % n
    #             tmp = nums[cur]
    #             nums[cur] = cur_val
    #             cur_val = tmp
    #             count += 1
    #             if cur == start:
    #                 break
    #         start += 1

    #  Leetcode 206
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, cur = None, head
        while cur:
            next = cur.next
            cur.next = prev
            prev = cur
            cur = next
        return prev

    # def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    #
    #     def _reverseList( head: Optional[ListNode]):
    #         #  Base case
    #         if not head or not head.next:
    #             return head, head
    #
    #         new_head, tail = _reverseList( head.next )
    #         tail.next = head
    #         head.next = None
    #         return new_head, head
    #
    #     new_head, _ = _reverseList(head)
    #     return new_head

    #  Leetcode 300
    def lengthOfLIS(self, nums: List[int]) -> int:
        lis = []
        for i in range(len(nums)):
            index = bisect.bisect_left(lis, nums[i])
            if index > len(lis) - 1:
                lis.append(nums[i])
            else:
                lis[index] = nums[i]
        return len(lis)

    #  Recursive version with memo.
    # def lengthOfLIS(self, nums: List[int]) -> int:
    #
    #     def _lengthOfLIS( nums: List[int], memo: List[int],k: int ) -> int:
    #         if memo[k]:
    #             return memo[k]
    #         if k == 0:
    #             memo[k] = 1
    #             return 1
    #         memo[k] = 1
    #         for j in range(k):
    #             if nums[j] < nums[k]:
    #                 memo[k] = max( memo[k], _lengthOfLIS(nums, memo, j) + 1)
    #         return memo[k]
    #
    #     n = len(nums)
    #     memo = [0] * n
    #     ans = 0
    #     for i in reversed(range(n)):
    #         ans = max(ans, _lengthOfLIS( nums, memo, i ))
    #     return ans

    #  The dynamic programming version.
    # def lengthOfLIS(self, nums: List[int] ) -> int:
    #     n = len( nums )
    #     memo = [1] * n
    #     for i in range(1, n):
    #         for j in range(i):
    #             if nums[j] < nums[i]:
    #                 memo[i] = max( memo[i], memo[j] + 1)
    #     return max(memo)

    #  Leetcode 322
    #  Greedy algorithm doesn't work on coinChange.
    #  Example:
    #  Coins = [2, 3, 6, 7] and Amount = 12,
    #  Greedy takes [2, 3, 7] and the optimal choice is [6, 6].
    # def coinChange(self, nums: List[int], amount: int) -> int:
    #     ans = [amount + 1] * (amount + 1)
    #     ans[0] = 0
    #     for i in range(len(ans)):
    #         for num in nums:
    #             if i - num >= 0:
    #                 ans[i] = min(ans[i], ans[i - num] + 1)
    #     return ans[amount] if ans[amount] < amount + 1 else -1

    # #  The recursive solution with a memo.
    # def coinChange(self, coins: List[int], amount: int) -> int:
    #     @cache
    #     def _coinChange( n ):
    #         if n < 0: return math.inf
    #         if n == 0: return 0
    #         ans = math.inf
    #         for coin in coins:
    #             ans = min(ans, _coinChange(n - coin) + 1)
    #         return ans
    #     ans = _coinChange(amount)
    #     return ans if ans != math.inf else -1

    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [amount + 1] * (amount + 1)
        dp[0] = 0
        for i in range(len(dp)):
            for coin in coins:
                if i - coin >= 0:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        return dp[-1]

    #  Leetcode 874
    def robotSim(self, commands: List[int], obstacles: List[List[int]]) -> int:
        direction = {"up": [0, 1, "left", "right"],
                     "down": [0, -1, "right", "left"],
                     "left": [-1, 0, "down", "up"],
                     "right": [1, 0, "up", "down"]}
        curDir, x, y, ans = "up", 0, 0, 0
        obstacles = set(map(tuple, obstacles))
        for command in commands:
            #  Turn left
            if command == -2:
                curDir = direction[curDir][2]
            #  Turn right
            elif command == -1:
                curDir = direction[curDir][3]
            else:
                for i in range(command):
                    if (x + direction[curDir][0], y + direction[curDir][1]) in obstacles:
                        break
                    x += direction[curDir][0]
                    y += direction[curDir][1]
                    ans = max(ans, x * x + y * y)
        return ans

    #  Leetcode 46
    #  The straightforward recursive solution.
    def permute(self, nums: List[int]) -> List[List[int]]:
        #  Base case
        if not nums: return [[]]
        return [[nums[i]] + item for i in range(len(nums)) for item in self.permute(nums[:i] + nums[i + 1:])]

    # #  The backtrack solution.
    # def permute(self, nums: List[int]) -> List[List[int]]:
    #     def _permute( nums, pos ):
    #         #  Base case.
    #         if pos == n:
    #             ans.append( seq[:] )
    #             return
    #         for i in range(len(nums)):
    #             seq.append( nums[i] )
    #             _permute( nums[:i] + nums[i + 1:], pos + 1)
    #             seq.pop()
    #     n = len(nums)
    #     ans, seq = [], []
    #     _permute( nums, 0 )
    #     return ans

    #  Leetcode 47
    #  The straightforward recursive solution.
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        #  Base case.
        if not nums: return [[]]
        ans, visited = [], set()
        for i in range(len(nums)):
            if nums[i] in visited: continue
            visited.add(nums[i])
            for seq in self.permuteUnique(nums[:i] + nums[i + 1:]):
                ans += [[nums[i]] + seq]
        return ans

    # #  The backtrace solution.
    # def permuteUnique(self, nums: List[int]) -> List[List[int]]:
    #     def _permuteUnique( nums, pos ):
    #         #  Base case.
    #         if pos == n:
    #             ans.append( seq[:] )
    #             return
    #         visited = set()
    #         for i in range(len(nums)):
    #             if nums[i] in visited: continue
    #             visited.add(nums[i])
    #             seq.append( nums[i] )
    #             _permuteUnique( nums[:i] + nums[i + 1:], pos + 1)
    #             seq.pop()
    #
    #     ans, seq, n = [], [], len(nums)
    #     _permuteUnique(nums, 0)
    #     return ans

    #  Leetcode 77
    #  The straightforward solution.
    def combine(self, n: int, k: int) -> List[List[int]]:
        #  Base case.
        if k == 0: return [[]]
        return [[i] + seq for i in reversed(range(1, n + 1)) for seq in self.combine(i - 1, k - 1)]
        #  The expanded version.
        # ans = []
        # for i in reversed(range(1, n + 1)):
        #     for seq in self.combine( i - 1, k - 1 ):
        #         ans += [ [i] + seq ]
        # return ans

    # #  The backtrack solution.
    # def combine(self, n: int, k: int) -> List[List[int]]:
    #     def _combine( n, k ):
    #         #  Base case.
    #         if k == 0:
    #             ans.append( seq[:] )
    #             return
    #         for i in reversed( range(1, n + 1) ):
    #             seq.append( i )
    #             _combine( i - 1, k - 1 )
    #             seq.pop()
    #     ans, seq = [], []
    #     _combine(n , k)
    #     return ans

    #  Leetcode 78
    def subsets(self, nums: List[int]) -> List[List[int]]:
        sets = [[]]
        for num in nums:
            sets += [[num] + s for s in sets]
        return sets

    #  Leetcode 72
    # #  The straightforward recursive solution with a memo.
    # @cache
    # def minDistance(self, word1: str, word2: str) -> int:
    #     if not word1 and not word2:
    #         return 0
    #     if not word1 or not word2:
    #         return len(word1 or word2)
    #     if word1[0] == word2[0]:
    #         return self.minDistance( word1[1:], word2[1:] )
    #     insert = self.minDistance( word1, word2[1:])
    #     delete = self.minDistance( word1[1:], word2)
    #     replace = self.minDistance( word1[1:], word2[1:] )
    #     return min(insert, delete, replace) + 1

    # #  The recursive solution using indices and a memo.
    # def minDistance(self, word1: str, word2: str) -> int:
    #     @cache
    #     def _minDistance( i, j ):
    #         if i < 0 and j < 0: return 0
    #         if i < 0 and j >= 0: return j + 1
    #         if i >= 0 and j < 0: return i + 1
    #         if word1[i] == word2[j]:
    #             return _minDistance( i - 1, j - 1 )
    #         insert = _minDistance( i, j - 1 )
    #         replace = _minDistance( i - 1, j - 1 )
    #         delete = _minDistance( i - 1, j)
    #         return min(insert, replace, delete) + 1
    #     return _minDistance( len(word1) - 1, len(word2) - 1)

    #  The iterative solution.
    def minDistance(self, word1: str, word2: str) -> int:
        n1, n2 = len(word1) + 1, len(word2) + 1
        dp = [[0] * n2 for _ in range(n1)]
        for j in range(1, n2):
            dp[0][j] = j
        for i in range(1, n1):
            dp[i][0] = i
        for i in range(1, n1):
            for j in range(1, n2):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
        return dp[-1][-1]

    #  Leetcode 200
    def numIslands(self, grid: List[List[str]]) -> int:
        def dfs(i, j):
            if not (0 <= i < rows and 0 <= j < cols and grid[i][j] == "1"): return
            grid[i][j] = "0"
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                dfs(i + dy, j + dx)

        rows, cols = len(grid), len(grid[0])
        ans = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == "1":
                    ans += 1
                    dfs(i, j)
        return ans

    #  Leetcode 22
    #  The recursive solution with a memo.
    def generateParenthesis(self, n: int) -> List[str]:
        @cache
        def _generateParenthesis(i, j, parenthesis):
            #  Base case.
            if i == 0 and j == 0:
                ans.append(parenthesis)
            if i > 0:
                _generateParenthesis(i - 1, j, parenthesis + "(")
            if i < j:
                _generateParenthesis(i, j - 1, parenthesis + ")")

        ans = []
        _generateParenthesis(n, n, "")
        return ans

    #  Leetcode 51
    def solveNQueens(self, n: int) -> List[List[str]]:
        def _solveNQueens(i, seq):
            if i > n - 1:
                ans.append(seq)
                return
            for j in range(n):
                if j in cols or i - j in diag or i + j in backDiag:
                    continue
                cols.append(j)
                diag.add(i - j)
                backDiag.add(i + j)
                _solveNQueens(i + 1, seq + [j])
                cols.pop()
                diag.remove(i - j)
                backDiag.remove(i + j)

        cols, diag, backDiag = [], set(), set()
        ans = []
        _solveNQueens(0, [])
        return [["." * i + "Q" + "." * (n - i - 1) for i in seq] for seq in ans]

    #  Leetcode 433
    #  The BFS solution.
    def minMutation(self, start: str, end: str, bank: List[str]) -> int:
        n, level = len(start), 0
        queue, bankSet = [start], set(bank)
        if start in bankSet: bankSet.remove( start )
        while queue:
            nextQueue = []
            level += 1
            for seq in queue:
                for i in range(n):
                    for char in ['A', 'C', 'G', 'T']:
                        muta = seq[:i] + char + seq[i + 1:]
                        if muta in bankSet:
                            if muta == end: return level
                            bankSet.remove( muta )
                            nextQueue.append( muta )
            queue = nextQueue
        return -1

    # #  Two-ended BFS solution.
    # def minMutation(self, start: str, end: str, bank: List[str]) -> int:
    #     startSet, endSet, bankSet = {start}, {end}, set(bank)
    #     if end not in bankSet: return -1
    #     if start in bankSet:  bankSet.remove( start )
    #     if end in bankSet:  bankSet.remove( end )
    #     n, level = len(start), 0
    #     while startSet:
    #         nextSet = set()
    #         level += 1
    #         for seq in startSet:
    #             for i in range(n):
    #                 for char in ['A', 'C', 'G', 'T']:
    #                     mutation = seq[: i] + char + seq[i + 1:]
    #                     if mutation in endSet:
    #                         return level
    #                     if mutation in bankSet:
    #                         nextSet.add(mutation)
    #                         bankSet.remove(mutation)
    #         startSet = nextSet
    #         if len(startSet) > len(endSet):
    #             startSet, endSet = endSet, startSet
    #     return - 1

    #  Leetcode 127
    # #  The BFS solution.
    # def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
    #     queue, wordList = [beginWord], set(wordList)
    #     level = 1
    #     while queue:
    #         newQueue = []
    #         level += 1
    #         for seq in queue:
    #             for word in [seq[:i] + char + seq[i+1:]  for i in range(len(seq)) for char in string.ascii_lowercase]:
    #                 if word in wordList:
    #                     if word == endWord:
    #                         return level
    #                     wordList.remove( word )
    #                     newQueue.append( word )
    #         queue = newQueue
    #     return 0

    #  The two-ended BFS solution.
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in wordList: return 0
        front, end, wordList = {beginWord}, {endWord}, set(wordList)
        wordList.remove( endWord )
        level, n = 1, len(beginWord)
        while front:
            newFront = set()
            level += 1
            for seq in front:
                for word in [seq[:i] + char + seq[i + 1:] for i in range(n) for char in string.ascii_lowercase]:
                    if word in end: return level
                    if word in wordList:
                        wordList.remove( word )
                        newFront.add( word )
            front = newFront
            if len(front) > len(end):
                front, end = end, front
        return 0



#  Drive code.
if __name__ == "__main__":
    S = Solution()

    #  Leetcode 01
    print(S.twoSum([2, 7, 11, 15], 9))
    print(S.twoSum([3, 2, 4], 6))
    print(S.twoSum([3, 3], 6))

    #  Leetcode 02
    l1 = array2Linkedlist([2, 4, 3])
    l2 = array2Linkedlist([5, 6, 4])
    printLinkedlist(S.addTwoNumbers(l1, l2))

    #  Leetcode 03
    print(S.lengthOfLongestSubstring(""))

    #  Leetcode 11
    print(S.maxArea([1, 2, 1]))

    #  Leetcode 15
    print(S.threeSum([0]))

    #  Leetcode 21
    l1 = array2Linkedlist([1, 2, 4])
    l2 = array2Linkedlist([1, 3, 4])
    printLinkedlist(S.mergeTwoLists(l1, l2))

    #  Leetcode 26
    print(S.removeDuplicates([1, 1, 1, 2, 2, 3]))

    #  Leetcode 55
    print(S.canJump([2, 3, 1, 1, 4]))

    #  Leetcode 45
    print(S.jump([2, 3, 1, 1, 4]))
    print(S.jump([2, 3, 0, 1, 4]))

    #  Leetcode 62
    print(S.uniquePaths(3, 7))
    print(S.uniquePaths(3, 2))

    #  Leetcode 66
    print(S.plusOne([1, 2, 3, 9, 9]))

    #  Leetcode 70
    print(S.climbStairs(3))

    #  Leetcode 80
    print(S.removeDuplicates02([0, 0, 1, 1, 1, 1, 2, 3, 3]))

    #  Leetcode 82
    l1 = array2Linkedlist([1, 2, 3, 3, 4, 4, 5])
    l2 = array2Linkedlist([1, 1, 1, 2, 3])
    printLinkedlist(S.deleteDuplicates(l1))
    printLinkedlist(S.deleteDuplicates(l2))

    #  Leetcode 83
    l1 = array2Linkedlist([1, 1, 2])
    l2 = array2Linkedlist([1, 1, 2, 3, 3])
    printLinkedlist(S.deleteDuplicates03(l1))
    printLinkedlist(S.deleteDuplicates03(l2))

    #  Leetcode 88
    nums01, nums02 = [1, 2, 3, 0, 0, 0], [2, 5, 6]
    S.merge(nums01, nums02, 3, 3)
    print(nums01)
    nums01, nums02 = [0], [1]
    S.merge(nums01, nums02, 0, 1)
    print(nums01)

    #  Leetcode 125
    print(S.isPalindrome("A man, a plan, a canal: Panama"))
    print(S.isPalindrome("race a car"))
    print(S.isPalindrome(""))

    #  Leetcode 141 passed.
    #  Leetcode 142 passed.

    #  Leetcode 189
    nums01 = [1, 2, 3, 4, 5, 6, 7]
    nums02 = [-1, -100, 3, 99]
    S.rotate(nums01, 3)
    print(nums01)
    S.rotate(nums02, 2)
    print(nums02)

    #  Leetcode 206
    l1 = array2Linkedlist([1, 2, 3, 4, 5])
    l2 = array2Linkedlist([1, 2])
    l3 = array2Linkedlist([])
    printLinkedlist(S.reverseList(l1))
    printLinkedlist(S.reverseList(l2))
    printLinkedlist(S.reverseList(l3))

    #  Leetcode 300
    print(S.lengthOfLIS([10, 9, 2, 5, 3, 7, 101, 18]))
    print(S.lengthOfLIS([0, 1, 0, 3, 2, 3]))
    print(S.lengthOfLIS([7, 7, 7, 7, 7, 7, 7]))

    #  Leetcode 322
    print(S.coinChange([1, 2, 5], 11))
    print(S.coinChange([2], 3))
    print(S.coinChange([1], 0))
    print(S.coinChange([2, 5, 10, 1], 27))
    print(S.coinChange([186, 419, 83, 408], 6249))

    #  Leetcode 874
    print(S.robotSim([4, -1, 3], []))
    print(S.robotSim([4, -1, 4, -2, 4], [[2, 4]]))
    print(S.robotSim([6, -1, -1, 6], []))

    #  Leetcode 46
    print(S.permute([1, 2, 3]))
    print(S.permute([0, 1]))
    print(S.permute([1]))

    #  Leetcode 47
    print(S.permuteUnique([1, 1, 2]))
    print(S.permuteUnique([1, 2, 3]))

    # Leetcode 77
    print(S.combine(4, 2))
    print(S.combine(1, 1))

    # Leetcode 78
    print(S.subsets([1, 2, 3]))
    print(S.subsets([0]))

    #  Leetcode 72
    print(S.minDistance("hr", "r"))
    print(S.minDistance("horse", "ros"))
    print(S.minDistance("intention", "execution"))

    #  Leetcode 200
    print(S.numIslands([
        ["1", "1", "1", "1", "0"],
        ["1", "1", "0", "1", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "0", "0", "0"]
    ]))
    print(S.numIslands([
        ["1", "1", "0", "0", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "1", "0", "0"],
        ["0", "0", "0", "1", "1"]
    ]))

    #  Leetcode 22
    print(S.generateParenthesis(3))
    print(S.generateParenthesis(1))

    #  Leetcode 51
    print(S.solveNQueens(4))
    print(S.solveNQueens(1))

    #  Leetcode 433
    print(S.minMutation("AACCGGTT", "AACCGGTA", ["AACCGGTA"]))
    print(S.minMutation("AACCGGTT","AACCGGTA", []))
    print(S.minMutation("AACCGGTT", "AAACGGTA", ["AACCGGTA","AACCGCTA","AAACGGTA"]))
    print(S.minMutation("AAAAACCC", "AACCCCCC", ["AAAACCCC", "AAACCCCC", "AACCCCCC"]))
    print(S.minMutation("AAAACCCC","CCCCCCCC",["AAAACCCA","AAACCCCA","AACCCCCA","AACCCCCC","ACCCCCCC","CCCCCCCC","AAACCCCC","AACCCCCC"]))

    print("---------------------------------")
    #  Leetcode 127
    print(S.ladderLength("hit", "cog", ["hot","dot","dog","lot","log","cog"]))
    print(S.ladderLength("hit", "cog", ["hot","dot","dog","lot","log"]))
