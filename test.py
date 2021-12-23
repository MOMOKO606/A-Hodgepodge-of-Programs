from typing import List, Optional
import bisect


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
        ans = [False] * n
        ans[0] = True
        for i in range(1, n):
            for j in range(i):
                if i - j <= nums[j] and ans[j] is True:
                    ans[i] = True
                    break
        return ans[n - 1]

    # def canJump(self, nums: List[int]) -> bool:
    #     n = len(nums)
    #     reach = 0
    #     for j in range(n):
    #         if j > reach:
    #             return False
    #         if reach >= n - 1:
    #             return True
    #         reach = max(reach, j + nums[j])

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

    #  Leetcode 62
    def uniquePaths(self, m: int, n: int) -> int:
        ans = [[1 for j in range(n)] for i in range(m)]
        for i in range(1, m):
            for j in range(1, n):
                ans[i][j] = ans[i - 1][j] + ans[i][j - 1]
        return ans[m - 1][n - 1]

    #  Recursive version with a memo.
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

    #  Leetcode 70.
    def climbStairs(self, n: int) -> int:
        f1 = 1
        f2 = 2
        if n == 1: return f1
        if n == 2: return f2
        for i in range(3, n + 1):
            f3 = f1 + f2
            f1 = f2
            f2 = f3
        return f3

    #  The recursive algorithm with a memo.
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
    def coinChange(self, nums: List[int], amount: int) -> int:
        ans = [amount + 1] * (amount + 1)
        ans[0] = 0
        for i in range( len(ans) ):
            for num in nums:
                if i - num >= 0:
                    ans[i] = min(ans[i], ans[ i - num ] + 1)
        return ans[amount] if ans[amount] < amount + 1 else -1


    # def coinChange(self, nums: List[int], amount: int) -> int:
    #
    #     def _coinChange( nums: List[int], memo: List[float], amount: int) -> int:
    #         if memo[amount] < float("inf"):
    #             return int(memo[amount])
    #         if not amount:
    #             memo[amount] = 0
    #             return 0
    #         least_coins = float("inf")
    #         for num in nums:
    #             if amount - num >= 0:
    #                 least_coins = min(least_coins, _coinChange(nums, memo, amount - num) + 1)
    #         return least_coins
    #
    #     memo = [float("inf")] * (amount + 1)
    #     memo[0] = 0
    #     for num in nums:
    #         if num < len(nums):
    #             memo[num] = 1
    #     ans = _coinChange( nums, memo, amount )
    #     return ans if ans < float("inf")  else -1


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

    #  Leetcode 62
    print(S.uniquePaths(3, 7))

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
    print("---------------------------------")
    print(S.coinChange([1,2,5], 11))
    print(S.coinChange([2], 3))
    print(S.coinChange([1], 0))
    print(S.coinChange([2,5,10,1], 27))
    print(S.coinChange([186,419,83,408],6249))


