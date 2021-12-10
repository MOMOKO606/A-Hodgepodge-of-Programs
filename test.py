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

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        map = {}
        for i, num in enumerate(nums):
            key = target - num
            if key in map:
                return [map[key], i]
            map[num] = i

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


    def lengthOfLongestSubstring(self, s: str) -> int:
        l, max_count, usedchar = -1, 0, {}
        for r in range(len(s)):
            if s[r] in usedchar and usedchar[s[r]] > l:
                l = usedchar[s[r]]
            else:
                max_count = max(max_count, r - l)
            usedchar[s[r]] = r
        return max_count


    def maxArea(self, height: List[int]) -> int:
        area, i, j = 0, 0, len(height) - 1
        while i < j:
            area = max( area, min(height[i], height[j]) * (j - i))
            if height[i] < height[j]:
                i += 1
            else: j -= 1
        return  area


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







#  Drive code.
if __name__ == "__main__":
    S = Solution()

    #  Leetcode 01
    print(S.twoSum([2, 7, 11, 15], 9))
    print(S.twoSum([3, 2, 4], 6))
    print(S.twoSum([3, 3], 6))

    #  Leetcode 02
    l1 = array2Linkedlist([2,4,3])
    l2 = array2Linkedlist([5,6,4])
    printLinkedlist(S.addTwoNumbers(l1, l2))

    #  Leetcode 03
    print(S.lengthOfLongestSubstring(""))

    #  Leetcode 11
    print(S.maxArea([1,2,1]))

    #  Leetcode 15
    print(S.threeSum([0]))
