from typing import Optional, List

class ListNode:
    def __init__(self, var = 0, next = None):
        self.var = var
        self.next = next


def array2Linkedlist( nums:List[int] ) -> Optional[ListNode]:
    if not nums:
        return None

    head = cur = ListNode(nums[0])
    for i in range(1, len(nums)):
        cur.next = ListNode(nums[i])
        cur = cur.next
    return head


def linkedlist2Array( head: Optional[ListNode] ) -> List[int]:
    cur = head
    ans = []
    while cur:
        ans.append(cur.var)
        cur = cur.next
    return ans


def printLinkedlist( head: Optional[ListNode] ) -> None:
    print(linkedlist2Array(head))


class Solution:

    def reverseList_aux(self, head: Optional[ListNode]) -> Optional[ListNode]:
        #  Base case
        if head is None or head.next is None:
            return head, head

        new_head, new_tail = self.reverseList_aux( head.next )
        new_tail.next = head
        head.next = None

        return new_head, head


    def reverseList(self, head:Optional[ListNode]) -> Optional[ListNode]:
        new_head, _ = self.reverseList_aux( head )
        return new_head




if __name__ == "__main__":
    l00 = []
    l01 = [2,4,5,12,56,3,-3]
    l02 = [1,2,3,4,5]
    l03 = [1,2]

    ll00 = array2Linkedlist(l00)
    ll01 = array2Linkedlist(l01)
    ll02 = array2Linkedlist(l02)

    printLinkedlist( ll00 )
    printLinkedlist(ll01)
    printLinkedlist(ll02)

    S = Solution()
    printLinkedlist(S.reverseList(ll01))

