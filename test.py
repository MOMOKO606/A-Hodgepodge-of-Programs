from typing import List, Optional


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
                nums01[ m + n - 1] = nums01[m - 1]
                m -= 1
            else:
                nums01[m + n - 1] = nums02[n - 1]
        nums01[:n] = nums02[:n]



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

    #  leetcode 88
    print("---------------------------------")
    nums01,nums02 = [1,2,3,0,0,0], [2,5,6]
    S.merge(nums01, nums02, 3, 3)
    print(nums01)

    nums01, nums02 = [0], [1]
    S.merge(nums01, nums02, 0, 1)
    print(nums01)