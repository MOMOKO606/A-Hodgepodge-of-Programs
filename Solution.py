from typing import List, Optional


#  Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    """
    Easy.01

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


    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        pass



def list2Linkedlist( nums: List[int]) -> Optional[ListNode]:
    #  Generate linked list with the head called head.
    head = moving_point = ListNode( nums[0] )
    for i in range(1, len(nums)):
        moving_point.next = ListNode(nums[i])
        moving_point = moving_point.next
    return head


def printLinkedlist( head: Optional[ListNode]):
    while head:
        print(head.val)
        head = head.next


if __name__ == "__main__":
    #  Create an instance
    S = Solution()

    #  Data: lists.
    l0 = [2,7,11,15]
    l1_list = [9,9,9,9,9,9,9]
    l2_list = [9,9,9,9]

    #  Data: Linked lists.
    l1 = list2Linkedlist( l1_list )
    l2 = list2Linkedlist( l2_list )

    print(S.twoSum03(l0, 9))   #  Leetcode, easy01
    printLinkedlist( S.addTwoNumbers(l1, l2) )



