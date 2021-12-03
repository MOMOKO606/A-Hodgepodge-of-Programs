from typing import Optional, List

class ListNode:
    def __init__(self, var = 0, next = None):
        self.var = var
        self.next = next


def array2Linkedlist( nums: List[int] ) -> Optional[ListNode]:

    if not nums:
        return None

    head = cur = ListNode(nums[0])
    for j in range(1,len(nums)):
        cur.next = ListNode(nums[j])
        cur = cur.next

    return head


def linkedlist2Array( head: Optional[ListNode] ) -> List[int]:
    ans = []
    while head:
        ans.append(head.var)
        head = head.next
    return ans


def printLinkedlist( head: Optional[ListNode] ) -> None:
    print(linkedlist2Array(head))





class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:

        def reverseList_aux( head ):
            if head is None or head.next is None:
                return head, head

            new_head, new_tail = reverseList_aux( head.next )

            new_tail.next = head
            head.next = None

            return new_head, head

        head, tail = reverseList_aux( head )
        return head



if __name__ == "__main__":
    l00 = []
    l01 = [2,4,5,12,56,3,-3]
    l02 = [1,2,3,4,5]
    l03 = [1,2]

    ll00 = array2Linkedlist(l00)
    ll01 = array2Linkedlist(l01)
    ll02 = array2Linkedlist(l02)

    S = Solution()
    head = S.reverseList(ll00)
    printLinkedlist(head)







