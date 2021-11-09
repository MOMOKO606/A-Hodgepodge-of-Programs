from typing import Optional, List

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

    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:

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










if __name__ == "__main__":
    S = Solution()

    l1_list = [9,9,9,9,9,9,9]
    l2_list = [9,9,9,9]

    l1 = array2Linkedlist( l1_list )
    l2 = array2Linkedlist( l2_list )



    print(linkedlist2Array(S.addTwoNumbers( l1, l2 )))


