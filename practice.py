import math
from functools import cache
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
    ans, cur = [], head
    while cur:
        ans.append(cur.val)
        cur = cur.next
    return ans


class Solution:
    #  1137. N-th Tribonacci Number(easy)
    def tribonacci(self, n: int) -> int:
        if n < 2:
            return n
        elif n == 2:
            return 1
        f0, f1, f2 = 0, 1, 2
        for _ in range(n - 2):
            f0, f1, f2 = f1, f2, f0 + f1 + f2
        return f2

    #  125. Valid Palindrome(easy)
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

    #  283. Move zeros(easy)
    def moveZeroes(self, nums: List[int]) -> None:
        j = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                if i != j:
                    nums[i], nums[j] = nums[j], nums[i]
                j += 1

    #  11. Container With Most Water (Medium)
    def maxArea(self, height: List[int]) -> int:
        i, j, largest = 0, len(height) - 1, -math.inf,
        while i < j:
            largest = max(largest, min(height[i], height[j]) * (j - i))
            if height[i] <= height[j]:
                i += 1
            else:
                j -= 1
        return largest

    #  70. Climbing Stairs (easy)
    def climbStairs(self, n: int) -> int:
        if n < 3: return n
        f1, f2 = 1, 2
        for _ in range(n - 2):
            f1, f2 = f2, f1 + f2
        return f2

    # 1. Two Sum (easy)
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        nums_dict = dict()
        for i in range(len(nums)):
            remain = target - nums[i]
            if remain in nums_dict:
                return [i, nums_dict[remain]]
            nums_dict[nums[i]] = i

    #  15. 3Sum (medium)
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        ans, n = [], len(nums)
        for k in range(n - 2):
            if k > 0 and nums[k] == nums[k - 1]:
                continue
            i, j = k + 1, n - 1
            while i < j:
                key = nums[k] + nums[i] + nums[j]
                if key == 0:
                    ans.append([nums[k], nums[i], nums[j]])
                    i += 1
                    while i < n and nums[i] == nums[i - 1]:
                        i += 1
                    j -= 1
                    while j > k and nums[j] == nums[j + 1]:
                        j -= 1
                elif key < 0:
                    i += 1
                else:
                    j -= 1
        return ans

    #  206. Reverse Linked List (easy)
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, cur = None, head
        while cur:
            next = cur.next

            cur.next = prev

            prev = cur
            cur = next
        return prev

    #  141. Linked List Cycle (easy)
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False

    #  66. Plus One (easy)
    def plusOne(self, digits: List[int]) -> List[int]:
        j = len(digits) - 1
        while digits[j] + 1 == 10:
            digits[j] = 0
            j -= 1
            if j < 0:
                return [1] + digits
        digits[j] += 1
        return digits





    #  21.Merge Two Sorted Lists (easy)
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = cur = ListNode()
        while list1 and list2:
            if list1.val <= list2.val:
                cur.next = list1
                list1 = list1.next
            else:
                cur.next = list2
                list2 = list2.next
            cur = cur.next
        cur.next = list1 or list2
        return dummy.next

    #  26 (Easy)
    def removeDuplicates(self, nums: List[int]) -> int:
        i = 0
        for j in range(1, len(nums)):
            if nums[j] != nums[i]:
                i += 1
                nums[i] = nums[j]
        return i + 1

    #  88 (easy)
    def merge(self, nums1: List[int], m, nums2: List[int], n):
        while m and n:
            if nums1[m - 1] <= nums2[n - 1]:
                nums1[m + n - 1] = nums2[n - 1]
                n -= 1
            else:
                nums1[m + n - 1] = nums1[m - 1]
                m -= 1
        nums1[:n] = nums2[:n]


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
    nums02 = [2, 1]
    # S.moveZeroes(nums01)
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
    print(S.twoSum([2, 7, 11, 15], 9))
    print(S.twoSum([3, 2, 4], 6))
    print(S.twoSum([3, 3], 6))

    #  15 (medium)

    print(S.threeSum([-1, 0, 1, 2, -1, -4]))
    print(S.threeSum([0, 1, 1]))
    print(S.threeSum([0, 0, 0]))

    #  206 (easy)
    print(linkedlist2Array(S.reverseList(array2Linkedlist([1, 2, 3, 4, 5]))))

    print("--------------------------------------")
    #  66 (easy)
    print(S.plusOne([1, 2, 3]))
    print(S.plusOne([4, 3, 2, 1]))
    print(S.plusOne([9]))
    print("--------------------------------------")

    #  21 (easy)
    print(linkedlist2Array(S.mergeTwoLists(array2Linkedlist([1, 2, 4]), array2Linkedlist([1, 3, 4]))))
    print(linkedlist2Array(S.mergeTwoLists(array2Linkedlist([]), array2Linkedlist([]))))
    print(linkedlist2Array(S.mergeTwoLists(array2Linkedlist([]), array2Linkedlist([0]))))
    print(linkedlist2Array(S.mergeTwoLists(array2Linkedlist([1, 2, 3]), array2Linkedlist([5, 6, 7]))))

    #  26 (easy)
    print(S.removeDuplicates([1, 1, 2]))
    print(S.removeDuplicates([0, 0, 1, 1, 1, 2, 2, 3, 3, 4]))

    #  88 (easy)
    nums01 = [1, 2, 3, 0, 0, 0]
    nums02 = [2, 5, 6]
    S.merge(nums01, 3, nums02, 3)
    print(nums01)
