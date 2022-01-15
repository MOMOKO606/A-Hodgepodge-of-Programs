from typing import Optional, List
import bisect, math, collections, itertools


class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        return 'TreeNode({})'.format(self.val)


#  Tool made by StefanPochmann
#  Transfer [1,2,3,null,null,4,null,null,5] to a root.
#  For example deserialize('[1,2,3,null,null,4,null,null,5]')
#  https://leetcode.com/problems/recover-binary-search-tree/discuss/32539/Tree-Deserializer-and-Visualizer-for-Python
def deserialize(string):
    if string == '{}':
        return None
    nodes = [None if val == 'null' else TreeNode(int(val))
             for val in string.strip('[]{}').split(',')]
    kids = nodes[::-1]
    root = kids.pop()
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return root


def drawtree(root):
    def height(root):
        return 1 + max(height(root.left), height(root.right)) if root else -1

    def jumpto(x, y):
        t.penup()
        t.goto(x, y)
        t.pendown()

    def draw(node, x, y, dx):
        if node:
            t.goto(x, y)
            jumpto(x, y - 20)
            t.write(node.val, align='center', font=('Arial', 12, 'normal'))
            draw(node.left, x - dx, y - 60, dx / 2)
            jumpto(x, y - 20)
            draw(node.right, x + dx, y - 60, dx / 2)

    import turtle
    t = turtle.Turtle()
    t.speed(0);
    turtle.delay(0)
    h = height(root)
    jumpto(0, 30 * h)
    draw(root, 0, 30 * h, 40 * h)
    t.hideturtle()
    turtle.mainloop()


class DLLNode:
    def __init__(self, val=0):
        self.val = val
        self.prev = self.next = None


#  Auxiliary functions and classes.
#  Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


#  Transfer an array to a linked list.
def array2Linkedlist(nums: List[int]) -> Optional[ListNode]:
    dummy = cur = ListNode()
    for num in nums:
        cur.next = ListNode(num)
        cur = cur.next
    return dummy.next


#  Transfer a linked list to an array.
def linkedlist2Array(head: Optional[ListNode]) -> List[int]:
    cur = head
    res = []
    while cur:
        res.append(cur.val)
        cur = cur.next
    return res


def printLinkedlist(head: Optional[ListNode]) -> None:
    print(linkedlist2Array(head))


#  Transfer a linked list to represent a number
#  The number in each node represents a digit of a number from left to right.
def linkedlist2num(head: Optional[ListNode]) -> int:
    if head is None:
        return 0
    return head.val + 10 * linkedlist2num(head.next)


#  Transfer a number to a linked list in reverse order.
#  The number in each node represents a digit of a number in reverse order.
def num2list_rever(num: [int]) -> List[int]:
    if num == 0: return [0]
    res = []
    while num:
        res.append(num % 10)
        num //= 10
    return res


"""
449. Serialize and Deserialize BST (Medium) 
297. Serialize and Deserialize Binary Tree (Hard)
Design an algorithm to serialize and deserialize a binary search tree. 
There is no restriction on how your serialization/deserialization algorithm should work. 
You need to ensure that a binary search tree can be serialized to a string, 
and this string can be deserialized to the original tree structure.

The encoded string should be as compact as possible.
"""


class Codec:
    #  The serialize algorithm without using "^".
    def serialize(self, root: Optional[TreeNode]) -> str:
        def _serialize(root: Optional[TreeNode]) -> List[int]:
            if not root: return []
            return [root.val] + _serialize(root.left) + _serialize(root.right)

        return " ".join(map(str, _serialize(root)))

    #  The deserialize algorithm without using "^".
    def deserialize(self, data: str) -> Optional[TreeNode]:

        def _deserialize(data, leftLimit: float, rigjtLimit: float) -> Optional[TreeNode]:
            if data and leftLimit < data[0] < rigjtLimit:
                val = data.popleft()
                root = TreeNode(val)
                root.left = _deserialize(data, leftLimit, val)
                root.right = _deserialize(data, val, rigjtLimit)
                return root

        data = collections.deque(map(int, data.split()))
        return _deserialize(data, -math.inf, math.inf)

    # #  The serialize algorithm using "^".
    # def serialize(self, root: Optional[TreeNode]) -> str:
    #     """Encodes a tree to a single string.
    #     """
    #     if not root: return "^"
    #     #  We must add " " to distinguish  "1" "2" from "12".
    #     return str(root.val) + " " +  self.serialize( root.left ) + " " + self.serialize(root.right)
    #
    #
    # #  The deserialize algorithm using "^".
    # def deserialize(self, data: str) -> Optional[TreeNode]:
    #     """Decodes your encoded data to tree.
    #     """
    #     def _deserialize( data ) -> Optional[TreeNode]:
    #         val = data.popleft()
    #         if val == "^": return None
    #         root = TreeNode(int(val))
    #         root.left = _deserialize( data )
    #         root.right = _deserialize( data )
    #         return root
    #
    #     data = collections.deque( data.split(" "))
    #     return _deserialize( data )


"""
641. Design Circular Deque(Medium)
Design your implementation of the circular double-ended queue (deque).

Implement the MyCircularDeque class:
MyCircularDeque(int k) Initializes the deque with a maximum size of k.
boolean insertFront() Adds an item at the front of Deque. Returns true if the operation is successful, or false otherwise.
boolean insertLast() Adds an item at the rear of Deque. Returns true if the operation is successful, or false otherwise.
boolean deleteFront() Deletes an item from the front of Deque. Returns true if the operation is successful, or false otherwise.
boolean deleteLast() Deletes an item from the rear of Deque. Returns true if the operation is successful, or false otherwise.
int getFront() Returns the front item from the Deque. Returns -1 if the deque is empty.
int getRear() Returns the last item from Deque. Returns -1 if the deque is empty.
boolean isEmpty() Returns true if the deque is empty, or false otherwise.
boolean isFull() Returns true if the deque is full, or false otherwise.
"""


#  Method 1. circular deque using list.
class MyCircularDeque:

    def __init__(self, k: int):
        self.capacity = k
        self.size = 0
        self.front, self.rear = k - 1, 0
        self.deque = [0] * k

    def insertFront(self, value: int) -> bool:
        if self.isFull(): return False
        self.deque[self.front] = value
        self.front = (self.front - 1) % self.capacity
        self.size += 1
        return True

    def insertLast(self, value: int) -> bool:
        if self.isFull(): return False
        self.deque[self.rear] = value
        self.rear = (self.rear + 1) % self.capacity
        self.size += 1
        return True

    def deleteFront(self) -> bool:
        if self.isEmpty(): return False
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return True

    def deleteLast(self) -> bool:
        if self.isEmpty(): return False
        self.rear = (self.rear - 1) % self.capacity
        self.size -= 1
        return True

    def getFront(self) -> int:
        if self.isEmpty(): return -1
        return self.deque[(self.front + 1) % self.capacity]

    def getRear(self) -> int:
        if self.isEmpty(): return -1
        return self.deque[(self.rear - 1) % self.capacity]

    def isEmpty(self) -> bool:
        return self.size == 0

    def isFull(self) -> bool:
        return self.size == self.capacity


#  Method 2. circular deque using double linked list (= deque when use linked list) .
# class MyCircularDeque:
#
#     def __init__(self, k: int):
#         self.size, self.capacity = 0, k
#         self.front =  DLLNode()
#         self.rear = DLLNode()
#         self.front.next = self.rear
#         self.rear.prev = self.front
#
#     def insertFront(self, value: int) -> bool:
#         if self.isFull(): return False
#         self.front.val = value
#         if not self.front.prev:
#             new = DLLNode()
#             new.next = self.front
#             self.front.prev = new
#         self.front = self.front.prev
#         self.size += 1
#         return True
#
#     def insertLast(self, value: int) -> bool:
#         if self.isFull(): return False
#         self.rear.val = value
#         if not self.rear.next:
#             new = DLLNode()
#             new.prev = self.rear
#             self.rear.next = new
#         self.rear = self.rear.next
#         self.size += 1
#         return True
#
#     def deleteFront(self) -> bool:
#         if self.isEmpty(): return False
#         self.front = self.front.next
#         self.size -= 1
#         return True
#
#     def deleteLast(self) -> bool:
#         if self.isEmpty(): return False
#         self.rear = self.rear.prev
#         self.size -= 1
#         return True
#
#     def getFront(self) -> int:
#         if self.isEmpty(): return -1
#         return self.front.next.val
#
#     def getRear(self) -> int:
#         if self.isEmpty(): return -1
#         return self.rear.prev.val
#
#     def isEmpty(self) -> bool:
#         return self.size == 0
#
#     def isFull(self) -> bool:
#         return self.size == self.capacity


"""
155. Min Stack(Easy)
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.
Implement the MinStack class:

MinStack() initializes the stack object.
void push(int val) pushes the element val onto the stack.
void pop() removes the element on the top of the stack.
int top() gets the top element of the stack.
int getMin() retrieves the minimum element in the stack.

Example:
Input
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

Output
[null,null,null,null,-3,null,0,-2]
"""


class MinStack:
    def __init__(self):
        self.stack = []
        self.minstack = []

    def push(self, val: int) -> None:

        mintop = self.getMin()

        if mintop == [] or val < mintop:
            self.minstack.append(val)
        else:
            self.minstack.append(mintop)
        self.stack.append(val)

    def pop(self) -> None:
        self.stack.pop()
        self.minstack.pop()

    def top(self) -> int:
        if self.stack:
            return self.stack[-1]
        return []

    def getMin(self) -> int:
        if self.minstack:
            return self.minstack[-1]
        return []


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

    def twoSum03(self, nums: List[int], target: int) -> List[int]:
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
        return array2Linkedlist(num2list_rever(linkedlist2num(l1) + linkedlist2num(l2)))

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

            cur.next = ListNode(carry % 10)
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
            if s[right] in usedchar and left < usedchar[s[right]]:
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
        return max(dp)

    def lis(self, i: int, nums: List[int], dp: List[int]) -> int:
        if dp[i]:
            return dp[i]
        if i == 0:
            dp[i] = 1
            return 1

        max_len = 1
        for j in range(i):
            if nums[j] < nums[i]:
                max_len = max(max_len, self.lis(j, nums, dp) + 1)
        dp[i] = max_len
        return max_len

    #  Solution2: the iterative dynamic programming algorithm of lengthOfLIS.
    def lengthOfLIS_iter(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * n
        for i in range(n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

    #  Solution3: the greedy lengthOfLIS.
    def lengthOfLIS_greedy(self, nums: List[int]) -> int:
        ans = []
        for i in range(len(nums)):
            index = bisect.bisect_left(ans, nums[i])
            if index == len(ans):
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

        def coinChange_Aux(coins: List[int], amount: int):
            if amount < 0:
                return float("inf")
            if amount == 0:
                return 0

            smallest = float("inf")
            for i in range(len(coins)):
                smallest = min(smallest, coinChange_Aux(coins, amount - coins[i]))

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

        def coinChange_Aux(coins: List[int], amount: int, memo):

            if amount < 0:
                return float("inf")

            if not math.isnan(memo[amount]):
                return memo[amount]

            smallest = float("inf")
            for i in range(len(coins)):
                smallest = min(smallest, coinChange_Aux(coins, amount - coins[i], memo))

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
                    memo[i] = min(memo[i], memo[i - coins[j]] + 1)

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
        if m == 1 or n == 1:
            return 1
        return self.uniquePaths_recur(m - 1, n) + self.uniquePaths_recur(m, n - 1)

    #  Solution2: the recursive algorithm with memo.
    def uniquePaths_recur_memo(self, m: int, n: int) -> int:

        def uniquePath_Aux(m: int, n: int, memo: List[int]) -> int:
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

        return canJump_Aux(nums, len(nums) - 1)

    # def canJump_recur(self, nums:List[int]) -> bool:
    #     n = len(nums)
    #     #  Base case:
    #     if n == 1:
    #         return True
    #
    #     for i in range(n - 1):
    #         if self.canJump_recur( nums[:i + 1] ):
    #             if nums[i] + i >= n - 1:
    #                 return True
    #     return False

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
        return canJump_Aux(nums, len(nums) - 1, memo)

    # def canJump_recur_memo(self, nums:List[int]) -> bool:
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

    # def canJump_greedy(self, nums: List[int]) -> bool:
    #     n = len(nums)
    #     reach = 0
    #     for j in range(n):
    #         if j > reach:
    #             return False
    #         if reach >= n - 1:
    #             return True
    #         reach = max(reach, j + nums[j])

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

    #  Solution1 with two loops.
    def moveZeroes01(self, nums: List[int]) -> None:
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

    #  Solution2 with one loop.
    def moveZeroes02(self, nums: list[int]) -> None:
        j = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[j] = nums[i]
                if j != i:
                    nums[i] = 0
                j += 1

    #  Solution3 with one loop.
    def moveZeroes03(self, nums: List[int]) -> None:
        j = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[i], nums[j] = nums[j], nums[i]
                j += 1

    """
    11. Container with most water(Medium)
    
    Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai). 
    n vertical lines are drawn such that the two endpoints of the line i is at (i, ai) and (i, 0). 
    Find two lines, which, together with the x-axis forms a container, such that the container contains the most water.
    
    Example
    Input: height = [1,8,6,2,5,4,8,3,7]
    Output: 49
    Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. 
    In this case, the max area of water (blue section) the container can contain is 49.
    """

    #  Solution1 brute force.
    def maxArea_naive(self, height: List[int]) -> int:
        ans = 0
        for i in range(len(height) - 1):
            for j in range(i + 1, len(height)):
                ans = max(ans, self.getArea(height, i, j))
        return ans

    def getArea(self, height, i, j):
        return (j - i) * min(height[i], height[j])

    #  Solution2 one loop with two pointers.
    def maxArea(self, height: List[int]) -> int:
        i = 0
        j = len(height) - 1
        max_area = 0
        while i < j:
            max_area = max(max_area, (j - i) * min(height[i], height[j]))

            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
        return max_area

    """
    70. Climbing Stairs(Easy)
    
    You are climbing a staircase. It takes n steps to reach the top.

    Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
    
    Example
    Input: n = 2
    Output: 2
    Explanation: There are two ways to climb to the top.
    1. 1 step + 1 step
    2. 2 steps
    """

    #  Solution01: the naive recursive algorithm.
    def climbStairs_naive(self, n: int) -> int:
        if n == 1:
            return 1
        if n == 2:
            return 2
        return self.climbStairs_naive(n - 1) + self.climbStairs_naive(n - 2)

    #  Solution02: the recursive algorithm with memo.
    def climbStairs_memo(self, n: int) -> int:

        def climbStairs_aux(n: int, memo: List[int]) -> int:
            if memo[n]:
                return memo[n]
            if n == 1:
                memo[n] = 1
                return 1
            if n == 2:
                memo[n] = 2
                return 2
            return climbStairs_aux(n - 1, memo) + climbStairs_aux(n - 2, memo)

        memo = [0] * (n + 1)
        return climbStairs_aux(n, memo)

    #  Solution03: the dynamic algoritm.
    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1
        if n == 2:
            return 2
        ans = [0] * (n + 1)
        ans[1] = 1
        ans[2] = 2
        for i in range(3, n + 1):
            ans[i] = ans[i - 1] + ans[i - 2]
        return ans[n]

    #  Solution04: optimized algorithm.
    def climbStairs_opt(self, n: int) -> int:
        if n <= 2: return n
        f1, f2 = 1, 2
        for i in range(3, n + 1):
            f3 = f1 + f2
            f1 = f2
            f2 = f3
        return f3

    """
    15. threeSum(Medium)

    Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that 
    i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

    Example
    Input: nums = [-1,0,1,2,-1,-4]
    Output: [[-1,-1,2],[-1,0,1]]
    """

    #  Solution01: brute force - O(n^3).
    def threeSum_naive(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        ans = []
        for i in range(n - 2):
            key = 0 - nums[i]
            for j in range(i + 1, n - 1):
                pivot = key - nums[j]
                for k in range(j + 1, n):
                    if nums[k] == pivot:
                        l = sorted([nums[i], nums[j], nums[k]])
                        if l not in ans:
                            ans.append(l)
        return ans

    #  Solution02: Hash table - O(n^2)
    def threeSum_hash(self, nums: List[int]) -> List[int]:

        def twoSum(nums, i, key):
            hashmap = {}
            result = []
            for j in range(len(nums)):
                #  Avoid pivot i.
                if j == i:
                    continue

                wanted = key - nums[j]
                if wanted in hashmap:
                    result.append([nums[hashmap[wanted]], nums[j]])
                hashmap[nums[j]] = j

            return result

        ans = []
        for i in range(len(nums)):
            tmp = twoSum(nums, i, -nums[i])
            for l in tmp:
                if l is not None:
                    l.append(nums[i])
                    l.sort()
                    if l not in ans:
                        ans.append(l)
        return ans

    #  Solution03: sort and use 2 pointers - O(nlgn).
    def threeSum(self, nums: List[int]) -> List[int]:

        #  Sort the input array.
        nums.sort()
        ans, n = [], len(nums)

        for i in range(n):
            #  Sentinel, avoid duplicate element checking.
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            l, r = i + 1, n - 1
            while l < r:
                total = nums[l] + nums[r] + nums[i]
                if total == 0:
                    ans.append([nums[l], nums[r], nums[i]])
                    l += 1
                    r -= 1
                    #  Sentinels, avoid duplicate answer.
                    while l < n and nums[l] == nums[l - 1]:
                        l += 1
                    while r >= 0 and nums[r] == nums[r + 1]:
                        r -= 1
                elif total < 0:
                    l += 1
                else:
                    r -= 1
        return ans

    """
    206. Reverse Linked List(Easy)
    Given the head of a singly linked list, reverse the list, and return the reversed list.
    
    Example:
    Input: head = [1,2,3,4,5]
    Output: [5,4,3,2,1]
    """

    #  Solution01: recursive algorithm.
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:

        def reverseList_aux(head):
            if head is None or head.next is None:
                return head, head

            new_head, new_tail = reverseList_aux(head.next)

            new_tail.next = head
            head.next = None

            return new_head, head

        head, _ = reverseList_aux(head)
        return head

    #  Solution02: iterative algorithm.
    def reverseList_iter(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, cur = None, head

        while cur:
            #  Pythonic style:
            #  cur.next, prev, cur = prev, cur, cur.next
            next = cur.next
            cur.next = prev

            prev = cur
            cur = next

        return prev

    """
    141. Linked List Cycle(Easy)
    
    Given head, the head of a linked list, determine if the linked list has a cycle in it.
    There is a cycle in a linked list if there is some node in the list that can be reached again by continuously 
    following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. 
    Note that pos is not passed as a parameter.

    Return true if there is a cycle in the linked list. Otherwise, return false.
    
    Example
    Input: head = [3,2,0,-4], pos = 1
    Output: true
    Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).
    """

    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = fast = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if slow == fast:
                return True
        return False

    """
    142. c(Medium)

    Given the head of a linked list, return the node where the cycle begins. If there is no cycle, return null.
    There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. 
    Internally, pos is used to denote the index of the node that tail's next pointer is connected to (0-indexed). 
    It is -1 if there is no cycle. Note that pos is not passed as a parameter.
    Do not modify the linked list.
    
    Example:
    Input: head = [3,2,0,-4], pos = 1
    Output: tail connects to node index 1
    Explanation: There is a cycle in the linked list, where tail connects to the second node.
    """

    def detectCycle(self, head: ListNode) -> ListNode:
        #  Stage1. Find the node that fast and slow meet each other or there is no cycle.
        slow = fast = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next

            #  fast and slow first meet.
            if fast == slow:
                p1 = fast
                p2 = head
                while p1 != p2:
                    p1 = p1.next
                    p2 = p2.next
                return p1
        return None

    # The while-else version

    # def detectCycle(self, head: ListNode) -> ListNode:
    #     #  Stage1. Find the node that fast and slow meet each other or there is no cycle.
    #     slow = fast = head
    #     while fast and fast.next:
    #         fast = fast.next.next
    #         slow = slow.next
    #         #  fast and slow first meet, go to the second stage.
    #         if fast == slow:
    #             p1 = fast
    #             p2 = head
    #             break
    #     else:
    #         #  No cycle.
    #         return None
    #     #  Stage2. find the start node of the cycle.
    #     while head != slow:
    #         head = head.next
    #         slow = slow.next
    #     return head

    """
    26. Remove Duplicates from Sorted Array(Easy)
    
    Given an integer array nums sorted in non-decreasing order, 
    remove the duplicates in-place such that each unique element appears only once. 
    The relative order of the elements should be kept the same.
    
    Example
    Input: nums = [1,1,2]
    Output: 2, nums = [1,2,_]
    Explanation: Your function should return k = 2, with the first two elements of nums being 1 and 2 respectively.
    It does not matter what you leave beyond the returned k (hence they are underscores).
    """

    def removeDuplicates(self, nums: List[int]) -> int:
        #  j represents the index where nums[...j] are distinct elements.
        #  nums[...j] is the loop invariant.
        #  for each loop we maintain nums[...j] is true, and return the length of nums[...j].
        j = -1
        for i in range(len(nums)):
            if nums[i] != nums[j] or j < 0:
                j += 1
                nums[j] = nums[i]
        return j + 1

    """
    80. Remove Duplicates from Sorted Array II(Medium)
    
    Given an integer array nums sorted in non-decreasing order, 
    remove some duplicates in-place such that each unique element appears at most twice. 
    The relative order of the elements should be kept the same.
    
    Example1:
    Input: nums = [1,1,1,2,2,3]
    Output: 5, nums = [1,1,2,2,3,_]
    Explanation: Your function should return k = 5, with the first five elements of nums being 1, 1, 2, 2 and 3 respectively.
    
    Example2:
    Input: nums = [0,0,1,1,1,1,2,3,3]
    Output: 7, nums = [0,0,1,1,2,3,3,_,_]
    Explanation: Your function should return k = 7, with the first seven elements of nums being 0, 0, 1, 1, 2, 3 and 3 respectively.
    """

    #  Solution01: easier to read, but more lines.
    def removeDuplicates02(self, nums: List[int]) -> int:
        j = -1
        #  flag == 1 indicates we can get more the same elements.
        #  flag == 0 indicates no more duplicate elements.
        flag = 1
        for i in range(len(nums)):
            if j < 0 or nums[i] != nums[j]:
                j += 1
                flag = 1
                nums[j] = nums[i]
            elif flag:  # nums[i] == nums[j]
                j += 1
                nums[j] = nums[i]
                flag = 0
        return j + 1

    #  Solution02: fewer lines.
    def removeDuplicates02_beta(self, nums: List[int]) -> int:
        k = -1
        for i, num in enumerate(nums):
            if i < 2 or num != nums[k] or num != nums[k - 1]:
                k += 1
                nums[k] = num
        return k + 1

    #  Solution03: more tricky constrain.
    def removeDuplicates02_theta(self, nums: List[int]) -> int:
        k = -1
        for i in range(len(nums)):
            if i < 2 or nums[i] > nums[k - 1]:
                k += 1
                nums[k] = nums[i]
        return k + 1

    """
    83. Remove Duplicates from Sorted List(Easy)

    Given the head of a sorted linked list, delete all duplicates such that each element appears only once. 
    Return the linked list sorted as well.

    Example1:
    Input: head = [1,1,2,3,3]
    Output: [1,2,3]
    """

    #  Solution01: the recursive version.
    def deleteDuplicates_recur(self, head: Optional[ListNode]) -> Optional[ListNode]:

        if not head:
            return
        cur = head
        #  Find the first node that is not the same as head.
        while cur.next:
            if cur.val == cur.next.val:
                cur = cur.next
            else:
                #  Find the next different node.
                next = cur.next
                break
        else:
            #  There is no next different node.
            next = None
        #  Link head to next.
        head.next = next
        #  Do the same process from next recursively.
        self.deleteDuplicates_recur(next)
        return head

    #  Solution02: the iterative version using while-else.
    #  idea: link the previous distinct node to he next distinct node.
    def deleteDuplicates_iter(self, head: Optional[ListNode]) -> Optional[ListNode]:

        distinct = head

        while distinct:

            cur = distinct

            while cur.next:

                if cur.val == cur.next.val:
                    cur = cur.next
                else:
                    next = cur.next
                    break
            else:
                next = None

            distinct.next = next
            distinct = distinct.next

        return head

    #  Solution03: the iterative version with one while loop.
    #  idea: 1. if the next node is the same as the previous one, we link to the one after next node.
    #        2. if the next node is distinct, we move the pointer to the new one.
    def deleteDuplicates_iter02(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur = head
        while cur:
            if cur.next and cur.val == cur.next.val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        return head

    #  Solution04: the iterative version with one while loop.
    def deleteDuplicates_iter03(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur = head
        while cur:
            while cur.next and cur.val == cur.next.val:
                cur.next = cur.next.next
            cur = cur.next
        return head

    """
    82. Remove Duplicates from Sorted List II(Medium)
    
    Given the head of a sorted linked list, delete all nodes that have duplicate numbers, 
    leaving only distinct numbers from the original list. Return the linked list sorted as well.

    Example:
    Input: head = [1,2,3,3,4,4,5]
    Output: [1,2,5]
    """

    #  Solution01
    def deleteDuplicates02(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        cur = head
        #  flag = 1 means this node and nodes with the same value should be delete.
        #  flag = 0 means we keep the node.
        flag = 0

        while cur:
            #  Search nodes with the same value.
            if cur.next and cur.val == cur.next.val:
                cur.next = cur.next.next
                flag = 1
                continue

            #  When flag == 1
            if flag:
                #  Connect prev -> cur.next
                if prev:
                    prev.next = cur.next
                #  If prev doesn't exit, cur.next is the new head.
                else:
                    head = cur.next
            #  When flag == 0, we move prev
            else:
                prev = cur

            #  Move cur and reset flag.
            cur = cur.next
            flag = 0

        return head

    #  Solution02: short version with dummy head.
    def deleteDuplicates02_short(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        dummy.next = head
        prev, cur = dummy, head
        while cur:
            while cur.next and cur.val == cur.next.val:
                cur = cur.next
            if prev.next != cur:
                prev.next = cur.next
            else:
                prev = cur
            cur = cur.next
        return dummy.next

    """
    189. Rotate Array(Medium)
    
    Given an array, rotate the array to the right by k steps, where k is non-negative.
    
    Example1:
    Input: [1,2,3,4,5,6,7], k = 3 
    Output: [5,6,7,1,2,3,4]
    Explanation:
    rotate 1 steps to the right: [7,1,2,3,4,5,6]
    rotate 2 steps to the right: [6,7,1,2,3,4,5]
    rotate 3 steps to the right: [5,6,7,1,2,3,4]
    
    Example2:
    Input: nums = [-1,-100,3,99], k = 2
    Output: [3,99,-1,-100]
    Explanation: 
    rotate 1 steps to the right: [99,-1,-100,3]
    rotate 2 steps to the right: [3,99,-1,-100]
    """

    #  Solution01: 约瑟夫问题。
    def rotate01(self, nums: List[int], k: int) -> None:
        #  Sentinel for empty input.
        if not nums:
            return
        # Pointer start & cur are used to avoid the infinite loop.
        # When start == cur for the second time,
        # either we've checked all elements or we encountered an infinite loop
        # in which case we have to move on to pick another element.
        n = len(nums)
        count = start = 0
        while count < n:
            cur = start
            prev = nums[start]
            #  Keep replacing elements after rotated.
            #  Replace nums[next] with nums[cur].
            #  Then move cur to next.
            while True:
                next = (cur + k) % n
                prev, nums[next] = nums[next], prev  # the pythonic way to swap.
                cur = next
                count += 1
                if start == cur:
                    break

            start += 1

    #  Solution02: the classic three rotation.
    def rotate(self, nums: List[int], k: int) -> None:

        def reverseList(nums: List[int], p: int, q: int) -> None:
            while p <= q:
                nums[p], nums[q] = nums[q], nums[p]
                p += 1
                q -= 1

        n = len(nums)
        k = k % n
        reverseList(nums, 0, len(nums) - 1)
        reverseList(nums, 0, k - 1)
        reverseList(nums, k, n - 1)

    """
    21. Merge Two Sorted Lists(Easy)
    
    You are given the heads of two sorted linked lists list1 and list2.
    Merge the two lists in a one sorted list. The list should be made by splicing together the nodes of the first two lists.
    Return the head of the merged linked list.
    
    Example:
    Input: list1 = [1,2,4], list2 = [1,3,4]
    Output: [1,1,2,3,4,4]
    """

    #  Solution01: the iterative version.
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = cur = ListNode()
        while list1 and list2:
            if list1.val < list2.val:
                cur.next = list1
                list1 = list1.next
            else:
                cur.next = list2
                list2 = list2.next
            cur = cur.next
        cur.next = list1 or list2
        return dummy.next

    #  Solution02: the recursive version.
    def mergeTwoLists_recur(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        #  Base case:
        #  if list1 or list2 is empty, return the one is not empty.
        if not list1 or not list2:
            return list1 or list2

        if list1.val < list2.val:
            list1.next = self.mergeTwoLists_recur(list1.next, list2)
            return list1
        else:
            list2.next = self.mergeTwoLists_recur(list1, list2.next)
            return list2

    """
    88. Merge Sorted Array(Easy)
    You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, 
    and two integers m and n, representing the number of elements in nums1 and nums2 respectively.
    
    Merge nums1 and nums2 into a single array sorted in non-decreasing order.

    The final sorted array should not be returned by the function, but instead be stored inside the array nums1. 
    To accommodate this, nums1 has a length of m + n, where the first m elements denote the elements that should be merged, 
    and the last n elements are set to 0 and should be ignored. nums2 has a length of n.
    
    Example:
    Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
    Output: [1,2,2,3,5,6]
    Explanation: The arrays we are merging are [1,2,3] and [2,5,6].
    The result of the merge is [1,2,2,3,5,6] with the underlined elements coming from nums1.
    """

    def merge(self, nums1, m, nums2, n):
        while m and n:
            if nums1[m - 1] > nums2[n - 1]:
                nums1[m + n - 1] = nums1[m - 1]
                m -= 1
            else:
                nums1[m + n - 1] = nums2[n - 1]
                n -= 1
        nums1[:n] = nums2[:n]

    """
    66. Plus one(Easy)
    You are given a large integer represented as an integer array digits, where each digits[i] is the ith digit of the integer.
    The digits are ordered from most significant to least significant in left-to-right order. 
    The large integer does not contain any leading 0's.

    Increment the large integer by one and return the resulting array of digits.
    
    Example1:
    Input: digits = [1,2,3]
    Output: [1,2,4]
    Explanation: The array represents the integer 123.
    Incrementing by one gives 123 + 1 = 124.
    Thus, the result should be [1,2,4].
    """

    #  Solution01: the iterative version using while loop.
    def plusOne(self, digits: List[int]) -> List[int]:

        flag = 1
        j = len(digits) - 1

        while j >= 0 and flag + digits[j] > 9:
            flag = 1
            digits[j] = 0
            j -= 1

        if j >= 0:
            digits[j] += 1
        else:
            digits = [1] + digits

        return digits

    #  Solution02: the iterative version using for loop.
    def plusOne_iter(self, digits: List[int]) -> List[int]:
        digits[-1] += 1
        for i in reversed(range(1, len(digits))):
            if digits[i] < 10:
                break
            digits[i] = 0
            digits[i - 1] += 1

        if digits[0] > 9:
            digits[0] = 0
            return [1] + digits
        return digits

    #  Solution03: the recursive version.
    def plusOne_recur(self, digits: List[int]) -> List[int]:
        if len(digits) == 0:
            return [1]

        digits[-1] += 1
        if digits[-1] < 10:
            return digits
        else:
            digits[-1] = 0
            return self.plusOne_recur(digits[:-1]) + [0]

    """
    20. Valid Parentheses(Easy)
    Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', 
    determine if the input string is valid.

    An input string is valid if:
    Open brackets must be closed by the same type of brackets.
    Open brackets must be closed in the correct order.
    
    Example1:
    Input: s = "()[]{}"
    Output: true
    
    Example2:
    Input: s = "(]"
    Output: false
    """

    def isValid(self, s: str) -> bool:
        hashmap = {"(": ")", "[": "]", "{": "}"}
        ans = []
        for char in s:
            if char in hashmap.keys():
                ans.append(hashmap[char])
            #  Implied char in hashmap.values()
            elif not ans or ans.pop() != char:
                return False
        return ans == []

    """
    84.Largest Rectangle in Histogram(Hard)
    Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, 
    return the area of the largest rectangle in the histogram.
    
    Example
    Input: heights = [2,1,5,6,2,3]
    Output: 10
    Explanation: The above is a histogram where width of each bar is 1.
    The largest rectangle is shown in the red area, which has an area = 10 units.
    """

    #  Algorithm by using stack.
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = [(-1, -1)]  # (index, value)
        i, area, count, n = 0, 0, 0, len(heights)
        while count < n:
            if i < n and heights[i] > stack[-1][1]:
                stack.append((i, heights[i]))
                i += 1
            else:
                area = max(area, stack.pop()[1] * (i - stack[-1][0] - 1))
                count += 1
        return area

    # #  Brute-force algorithm.
    # def largestRectangleArea(self, heights: List[int]) -> int:
    #     largest_area = 0
    #     for i, num in enumerate(heights):
    #         l = r = i
    #         while l >= 0 and heights[l] >= num:
    #             l -= 1
    #         while r < len(heights) and heights[r] >= num:
    #             r += 1
    #         largest_area = max( largest_area, (r - l - 1) * num)
    #     return largest_area

    """
    42. Trapping Rain Water(Hard)
    Given n non-negative integers representing an elevation map where the width of each bar is 1, 
    compute how much water it can trap after raining.
    
    Example
    Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
    Output: 6
    Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. 
    In this case, 6 units of rain water (blue section) are being trapped.
    """

    #  Optimized version using two pointers
    def trap(self, heights: List[int]) -> int:
        n = len(heights)
        maxLeft, maxRight, l, r = heights[0], heights[n - 1], 1, n - 2
        ans = 0
        while l <= r:
            if maxLeft < maxRight:
                tmp = maxLeft - heights[l]
                if tmp > 0:
                    ans += tmp
                else:
                    maxLeft = heights[l]
                l += 1
            else:
                tmp = maxRight - heights[r]
                if tmp > 0:
                    ans += tmp
                else:
                    maxRight = heights[r]
                r -= 1
        return ans

    # #  The algorithm using a stack.
    # def trap(self, heights: List[int]) -> int:
    #     i, ans, stack = 0, 0, []
    #     while i < len(heights):
    #         if len(stack) == 0 or heights[stack[-1]] > heights[i]:
    #             stack += [i]
    #             i += 1
    #         else:
    #             h = heights[stack.pop()]
    #             if len(stack):
    #                 ans += (min(heights[i], heights[stack[-1]]) - h) * (i - stack[-1] -1)
    #     return ans

    # #  Optimized version using arrays.
    # def trap(self, heights:List[int]) -> int:
    #     max_left = [0] * len(heights)
    #     max_right = [0] * len(heights)
    #     ans = 0
    #     for i in range(1, len(heights)):
    #         max_left[i] = max(max_left[i - 1], heights[i - 1])
    #     for i in range(len(heights) - 2, -1, -1):
    #         max_right[i] = max(max_right[i + 1], heights[i + 1])
    #     for i in range(len(heights)):
    #         temp = min(max_left[i], max_right[i])
    #         if heights[i] < temp:
    #             ans += temp - heights[i]
    #     return ans

    # #  The brute-force
    # def trap(self, heights: List[int]) -> int:
    #     ans = 0
    #     for i in range(len(heights)):
    #         max_left = 0
    #         for l in reversed(range(i)):
    #             max_left = max(max_left, heights[l])
    #         max_right = 0
    #         for r in range(i, len(heights)):
    #             max_right = max(max_right, heights[r])
    #
    #         temp = min(max_left, max_right) - heights[i]
    #         if temp > 0:
    #             ans += temp
    #     return ans

    # #  The algorithm that mimics the fill and drain water.
    # def trap(self, heights:List[int]) -> int:
    #     maxLeft = maxRight  = -1
    #     ans = []
    #     #  Over-fill water depends on the leftMax bar.
    #     for h in heights:
    #         ans.append( maxLeft - h ) if h < maxLeft else ans.append(0)
    #         maxLeft = max( maxLeft, h )
    #
    #     #  Drain water depends on the rightMax bar.
    #     for i,h in reversed(list(enumerate(heights))):
    #         #  Re-calculate the amount of water depends on the rightMax bar.
    #         if maxRight > h:
    #             ans[i] = min( ans[i], maxRight - h)
    #         else:  #  No bars on the right to hold water.
    #             ans[i] = 0
    #         maxRight = max( maxRight, h)
    #     return sum(ans)

    """
    239. Sliding Window Maximum( Hard )
    You are given an array of integers nums, there is a sliding window of size k which is moving from the very left of 
    the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right 
    by one position.
    Return the max sliding window.
    
    Example:
    Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
    Output: [3,3,5,5,6,7]
    Explanation: 
    Window position                Max
    ---------------               -----
    [1  3  -1] -3  5  3  6  7       3
     1 [3  -1  -3] 5  3  6  7       3
     1  3 [-1  -3  5] 3  6  7       5
     1  3  -1 [-3  5  3] 6  7       5
     1  3  -1  -3 [5  3  6] 7       6
     1  3  -1  -3  5 [3  6  7]      7
    """

    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        ans = []
        deque = collections.deque()
        for i, num in enumerate(nums):
            #  Insert new element.
            while deque and num > nums[deque[-1]]:
                deque.pop()
            deque += [i]
            #  If the windows has passed the element on the most left in the deque.
            if deque[0] < i - k + 1:
                deque.popleft()
            #  Initialize the k-window.
            if i >= k - 1:
                ans += [nums[deque[0]]]
        return ans

    """
    242. Valid Anagram(Easy)
    Given two strings s and t, return true if t is an anagram of s, and false otherwise.
    
    Example 1:
    Input: s = "anagram", t = "nagaram"
    Output: true
    
    Example 2:
    Input: s = "rat", t = "car"
    Output: false
    """

    def isAnagram(self, s: str, t: str) -> bool:
        checklist = {}
        for char in s:
            if char in checklist:
                checklist[char] += 1
            else:
                checklist[char] = 1

        for char in t:
            if char not in checklist:
                return False
            else:
                checklist[char] -= 1

        for value in checklist.values():
            if value != 0:
                return False
        return True

    """
    49. Group Anagrams(Medium)
    Given an array of strings strs, group the anagrams together. You can return the answer in any order.
    An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, 
    typically using all the original letters exactly once.
    
    Example:
    Input: strs = ["eat","tea","tan","ate","nat","bat"]
    Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
    """

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        ans = {}
        for word in strs:
            key = "".join(sorted(word))
            ans[key] = ans.get(key, []) + [word]
        return list(ans.values())

    """
    94. Binary Tree Inorder Traversal (Easy)
    Given the root of a binary tree, return the inorder traversal of its nodes' values.
    
    Example:
    Input: root = [1,null,2,3]
    Output: [1,3,2]
    """

    #  The recursive version01.
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:

        def _inorderTraversal(root: Optional[TreeNode], ans: List[int]) -> None:
            if root:
                _inorderTraversal(root.left, ans)
                ans += [root.val]
                _inorderTraversal(root.right, ans)

        ans = []
        _inorderTraversal(root, ans)
        return ans

    # #  The recursive version02.
    # def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    #     if not root:
    #         return []
    #     return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)

    # #  Using a stack without flags..
    # def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    #     stack, ans = [], []
    #     while stack or root:
    #         if root:
    #             stack.append(root)
    #             root = root.left
    #         else:
    #             root = stack.pop()
    #             ans += [root.val]
    #             root = root.right
    #     return ans

    """
    144. Binary Tree Preorder Traversal(Easy)
    Given the root of a binary tree, return the preorder traversal of its nodes' values.
    
    Example:
    Input: root = [1,null,2,3]
    Output: [1,2,3]
    """

    #  The recursive version01.
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        def _preorderTraversal(root: Optional[TreeNode], ans: List[int]) -> None:
            if root:
                ans += [root.val]
                _preorderTraversal(root.left, ans)
                _preorderTraversal(root.right, ans)

        ans = []
        _preorderTraversal(root, ans)
        return ans

    #  The recursive version02.
    # def preorderTraversal(self, root:Optional[TreeNode]) -> List[int]:
    #     if not root:
    #         return []
    #     return [root.val] + self.preorderTraversal(root.left) + self.preorderTraversal(root.right)

    # #  Using a stack without flags.
    # def preorderTraversal(self, root:Optional[TreeNode]) -> List[int]:
    #     stack, ans = [root], []
    #     while stack:
    #         root = stack.pop()
    #         if root:
    #             ans += [root.val]
    #             stack.append( root.right )
    #             stack.append( root.left )
    #     return ans

    """
    145. Binary Tree Postorder Traversal(Easy)
    Given the root of a binary tree, return the postorder traversal of its nodes' values.
    
    Example:
    Input: root = [1,null,2,3]
    Output: [3,2,1]
    """

    #  The recursive version01.
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        def _postorderTraversal(root: Optional[TreeNode], ans: List[int]) -> None:
            if root:
                _postorderTraversal(root.left, ans)
                _postorderTraversal(root.right, ans)
                ans += [root.val]

        ans = []
        _postorderTraversal(root, ans)
        return ans

    # #  The recursive version02.
    # def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    #     if not root:
    #         return []
    #     return self.postorderTraversal(root.left) + self.postorderTraversal(root.right) + [root.val]

    # #  Using a stack without flags.
    # # [left, right, root] = reversed([root, right, left])
    # def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    #     stack, ans = [root], []
    #     while stack:
    #         root = stack.pop()
    #         if root:
    #             ans += [root.val]
    #             stack.append( root.left )
    #             stack.append( root.right )
    #     return ans[::-1]

    # #  Using a stack with flags.
    # def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    #     ans, stack = [], [ (root, False) ]
    #     while stack:
    #         root, visited = stack.pop()
    #         if not root:
    #             continue
    #         if visited:
    #             ans += [root.val]
    #         else:
    #             stack.append( (root, True) )
    #             stack.append( (root.right, False) )
    #             stack.append( (root.left, False) )
    #     return ans

    """
    589. N-ary Tree Preorder Traversal(Easy)
    Given the root of an n-ary tree, return the preorder traversal of its nodes' values.
    Nary-Tree input serialization is represented in their level order traversal. 
    Each group of children is separated by the null value (See examples)
    
    Input: root = [1,null,3,2,4,null,5,6]
    Output: [1,3,5,6,2,4]
    """

    #  The recursive version.
    def preorder(self, root: 'Node') -> List[int]:
        if not root:
            return []
        ans = [root.val]
        for child in root.children:
            ans += self.preorder(child)
        return ans

    # #  The iterative version.
    # def preorder(self, root: 'Node') -> List[int]:
    #     ans, stack = [], [root]
    #     while stack:
    #         root = stack.pop()
    #         if root:
    #             ans += [root.val]
    #             for child in reversed(root.children):
    #                 stack += [child]
    #     return ans

    """
    590. N-ary Tree Postorder Traversal( Easy )
    Given the root of an n-ary tree, return the postorder traversal of its nodes' values.
    Nary-Tree input serialization is represented in their level order traversal. 
    Each group of children is separated by the null value (See examples)
    
    Example:
    Input: root = [1,null,3,2,4,null,5,6]
    Output: [5,6,3,2,4,1]
    """

    #  The recursive version.
    def postorder(self, root: 'Node') -> List[int]:
        if not root:
            return []
        ans = []
        for child in root.children:
            ans += self.postorder(child)
        ans += [root.val]
        return ans

    # #  The iterative version.
    # def postorder(self, root: 'Node') -> List[int]:
    #     ans, stack = [], [root]
    #     while stack:
    #         root = stack.pop()
    #         if root:
    #             ans += [root.val]
    #             for child in root.children:
    #                 stack.append( child )
    #     return ans[::-1]

    """
    429. N-ary Tree Level Order Traversal( Medium )
    Given an n-ary tree, return the level order traversal of its nodes' values.
    
    Example:
    Input: root = [1,null,3,2,4,null,5,6]
    Output: [[1],[3,2,4],[5,6]]
    """

    #  Algorithm using a deque.
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        if not root: return []
        ans, deque = [], collections.deque([root])
        while deque:
            line = []
            for i in range(len(deque)):
                node = deque.popleft()
                for child in node.children:
                    deque.append(child)
                line += [node.val]
            ans += [line]
        return ans

    # #  Algorithm using two lists.
    # def levelOrder(self, root: 'Node') -> List[List[int]]:
    #     if not root: return []
    #     ans, level = [], [root]
    #     while level:
    #         ans.append([node.val for node in level])
    #         level = [child for node in level for child in node.children]
    #     return ans

    """
    22. Generate Parentheses(Medium)
    Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
    
    Example1:
    Input: n = 1
    Output: ["()"]
    
    Example2:
    Input: n = 3
    Output: ["((()))","(()())","(())()","()(())","()()()"]
    """

    #  Version01.
    def generateParenthesis(self, n: int) -> List[str]:

        def _generateParenthesis(left: int, right: int, s: str, ans: List[str]) -> None:
            if left == 0 and right == 0:
                ans += [s]
            if left > 0:
                _generateParenthesis(left - 1, right, s + "(", ans)
            if right > left:
                _generateParenthesis(left, right - 1, s + ")", ans)

        ans = []
        _generateParenthesis(n, n, "", ans)
        return ans

    # #  Version02.
    # def generateParenthesis(self, n: int) -> List[str]:
    #
    #     def _generateParenthesis( left: int, right: int, s: str, ans: List[str]) -> None:
    #         if left == 0 and right == 0:
    #             ans += [s]
    #         if left < 0 or right < left:
    #             return
    #         _generateParenthesis( left - 1, right, s + "(", ans)
    #         _generateParenthesis( left, right - 1, s + ")", ans)
    #
    #     ans = []
    #     _generateParenthesis(n, n, "", ans)
    #     return ans

    """
    226. Invert Binary Tree (Easy)
    Given the root of a binary tree, invert the tree, and return its root.
    
    Example:
    Input: root = [4,2,7,1,3,6,9]
    Output: [4,7,2,9,6,3,1]
    """

    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root:
            root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root

    """
    104. Maximum Depth of Binary Tree (Easy)
    Given the root of a binary tree, return its maximum depth.
    A binary tree's maximum depth is the number of nodes along the longest path from the root node down to 
    the farthest leaf node.
    
    Example:
    Input: root = [3,9,20,null,null,15,7]
    Output: 3
    """

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root: return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1

    """
    111. Minimum Depth of Binary Tree(Easy)
    Given a binary tree, find its minimum depth.
    The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.

    Example:
    Input: root = [3,9,20,null,null,15,7]
    Output: 2
    """

    #  The recursive version02.
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root: return 0
        if not root.left:
            return self.minDepth(root.right) + 1
        if not root.right:
            return self.minDepth(root.left) + 1
        return min(self.minDepth(root.left), self.minDepth(root.right)) + 1

    # #  The recursive version01.
    # def minDepth(self, root: Optional[TreeNode]) -> int:
    #
    #     def _minDepth(root: Optional[TreeNode]) -> float:
    #         if not root: return float("inf")
    #         if not root.left and not root.right: return 1
    #         return min( _minDepth(root.left), _minDepth(root.right) ) + 1
    #
    #     ans = _minDepth(root)
    #     return int(ans) if ans < float("inf") else 0

    """
    98. Validate Binary Search Tree( Medium )
    Given the root of a binary tree, determine if it is a valid binary search tree (BST).
    
    Example:
    Input: root = [2,1,3]
    Output: true
    """

    #  The recursive version01: the smarter one.
    def isValidBST(self, root: Optional[TreeNode]) -> bool:

        def _isValidBST(root: Optional[TreeNode], leftmax=-math.inf, rightmin=math.inf) -> bool:
            if not root: return True
            if leftmax >= root.val or rightmin <= root.val:
                return False
            return _isValidBST(root.left, leftmax, root.val) and _isValidBST(root.right, root.val, rightmin)

        return _isValidBST(root)

    # # The recursive version02: the naive one.
    # def isValidBST(self, root: Optional[TreeNode]) -> bool:
    #
    #     def _isValidBST( root: Optional[TreeNode] ) -> List[bool, int]:
    #         if not root: return [True, math.inf, -math.inf]
    #
    #         [ leftTree, leftMin, leftMax] = _isValidBST( root.left )
    #         if leftTree is False: return [False, 0, 0]
    #
    #         [rightTree, rightMin, rightMax] = _isValidBST( root.right )
    #         if rightTree is False: return [False, 0, 0]
    #
    #         flag = leftMax < root.val and root.val < rightMin
    #         curMin = leftMin if leftMin != math.inf else root.val
    #         curMax = rightMax if rightMax != -math.inf else root.val
    #         return [flag, curMin, curMax]
    #
    #     return _isValidBST( root )[0]

    # #  The iterative version
    # def isValidBST(self, root: Optional[TreeNode]) -> bool:
    #     stack = [[root, -math.inf, math.inf]]
    #     while stack:
    #         [root, leftLimit, rightLimit] = stack.pop()
    #         if root:
    #             if leftLimit >= root.val or root.val >= rightLimit:
    #                 return False
    #             stack.append( [root.right, root.val, rightLimit])
    #             stack.append( [root.left, leftLimit, root.val])
    #     return True

    # #  The recursive version using inorder traversal.
    # def isValidBST(self, root: Optional[TreeNode]) -> bool:
    #
    #     def _isValidBST( root: Optional[TreeNode]) -> bool:
    #         if not root: return True
    #         if not _isValidBST( root.left): return False
    #         if root.val <= self.prev: return False
    #         self.prev = root.val
    #         return _isValidBST( root.right )
    #
    #     self.prev = -math.inf
    #     return _isValidBST( root )

    """
    105. Construct Binary Tree from Preorder and Inorder Traversal (Medium)
    Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree 
    and inorder is the inorder traversal of the same tree, construct and return the binary tree.
    
    Example1:
    Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
    Output: [3,9,20,null,null,15,7]
    
    Example2:
    Input: preorder = [-1], inorder = [-1]
    Output: [-1]
    """

    #  The recursive algorithm.
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if preorder:
            r = inorder.index(preorder[0])
            root = TreeNode(inorder[r])
            root.left = self.buildTree(preorder[1: r + 1], inorder[:r])
            root.right = self.buildTree(preorder[r + 1:], inorder[r + 1:])
            return root

    """
    235. Lowest Common Ancestor of a Binary Search Tree(Easy)
    Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.
    
    Example1:
    Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
    Output: 6
    Explanation: The LCA of nodes 2 and 8 is 6.
    
    Example2:
    Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
    Output: 2
    Explanation: The LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.
    
    Example3:
    Input: root = [2,1], p = 2, q = 1
    Output: 2
    """

    #  The recursive solution.
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root:
            if (root.val - p.val) * (root.val - q.val) > 0:
                return self.lowestCommonAncestor((root.left, root.right)[root.val < p.val], p, q)
        return root

    # #  The iterative solution.
    # def lowestCommonAncestor(self, root: 'TreeNode', p:'TreeNode', q:'TreeNode') -> 'TreeNode':
    #     while root:
    #         if (root.val - p.val) * ( root.val - q.val) > 0:
    #             root = (root.left, root.right)[root.val < p.val]
    #         else: break
    #     return root

    """
    236. Lowest Common Ancestor of a Binary Tree (Medium)
    Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.
    
    Example1:
    Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
    Output: 3
    Explanation: The LCA of nodes 5 and 1 is 3.
    
    Example2:
    Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
    Output: 5
    Explanation: The LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the 
    LCA definition.
    
    Example3:
    Input: root = [1,2], p = 1, q = 2
    Output: 1
    """

    #  The common recursive algorithm.
    #  return the Lowest Common Ancestor if p and q are in the tree of root.
    #  return None if none of p and q are in the tree of root.
    #  return p if p is in the tree, q is not.
    #  return q if q is in the tree, p is not.
    def lowestCommonAncestor_naive(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        #  Base case.
        if not root or root == p or root == q:
            return root
        left = self.lowestCommonAncestor_naive(root.left, p, q)
        right = self.lowestCommonAncestor_naive(root.right, p, q)
        if left and right: return root
        return left or right

    """
    77. Combinations (Medium)
    Given two integers n and k, return all possible combinations of k numbers out of the range [1, n].
    You may return the answer in any order.
    
    Example1:
    Input: n = 4, k = 2
    Output:
    [
      [2,4],
      [3,4],
      [2,3],
      [1,2],
      [1,3],
      [1,4],
    ]
    
    Example2:
    Input: n = 1, k = 1
    Output: [[1]]
    """

    #  The concise recursive version.
    def combine(self, n, k):
        if k == 0:
            return [[]]  # if we just return [] the for loop in 2375 might not start since it's empty.
        return [[i] + item for i in reversed(range(1, n + 1)) for item in self.combine(i - 1, k - 1)]

        # #  Equals to the lines below:
        # ans = []
        # for i in reversed(range(1, n + 1)):
        #     for item in self.combine(i - 1, k - 1):
        #         ans += [[i] + item]
        # return ans

    # #  The tricky solution using the library, very efficient.
    # def combine(self, n: int, k: int):
    #     return list(itertools.combinations(range(1, n + 1), k))

    # #  Backtracking recursive solution.
    # #  The idea is to use a position in n,
    # #  to place all the possible numbers from [pos, ..., n] at the first index of pair.
    # #  When the length of pair reaches k, it's a valid combination. Put it into the ans.
    # def combine(self, n: int, k: int) -> List[List[int]]:
    #     def backtracking(n, k, pos) -> None:
    #         #  Base case
    #         if len(pair) == k:
    #             ans.append(pair[:])
    #             return
    #         for i in range(pos, n + 1):
    #             pair.append( i )
    #             backtracking(n, k, i + 1)
    #             pair.pop()
    #         return
    #     pair, ans = [], []
    #     backtracking(n, k, 1)
    #     return ans

    """
    46. Permutations (Medium)
    Given an array nums of distinct integers, return all the possible permutations. 
    You can return the answer in any order.
    
    Example1:
    Input: nums = [1,2,3]
    Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
    
    Example2:
    Input: nums = [0,1]
    Output: [[0,1],[1,0]]
    
    Example3:
    Input: nums = [1]
    Output: [[1]]
    """

    #  The consice recursive solution.
    def permute(self, nums: List[int]) -> List[List[int]]:
        if not nums: return [[]]
        return [[nums[i]] + l for i in range(len(nums)) for l in self.permute(nums[:i] + nums[i + 1:])]

    # #  The trick solution using library.
    # def permute(self, nums: List[int]):
    #     return list(itertools.permutations( nums ))

    # #  The easier understand recursive solution.
    # def permute(self, nums: List[int]) -> List[List[int]]:
    #     #  Make sure the items in the result of recursive functions are iterable.
    #     if not nums: return [[]]
    #     ans = []
    #     for i in range(len(nums)):
    #         nums[i], nums[0] = nums[0], nums[i]
    #         for item in self.permute(nums[1:]):
    #             #  Notice, we can only add [] + [], cannot add int + []
    #             #  Notice, ans is list of list -- [[]]
    #             ans += [[nums[0]] + item]
    #     return ans

    """
    47. Permutations II (Medium)
    Given a collection of numbers, nums, that might contain duplicates, 
    return all possible unique permutations in any order.

    Example1:
    Input: nums = [1,1,2]
    Output:
            [[1,1,2],
             [1,2,1],
             [2,1,1]]
             
    Example2:
    Input: nums = [1,2,3]
    Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
    """

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        #  Make sure the items in the result of recursive functions are iterable.
        if not nums: return [[]]
        ans, checked = [], {}
        for i in range(len(nums)):
            if nums[i] in checked:
                continue
            nums[i], nums[0] = nums[0], nums[i]
            checked[nums[0]] = nums[0]
            for item in self.permuteUnique(nums[1:]):
                #  Notice, we can only add [] + [], cannot add int + []
                #  Notice, ans is list of list -- [[]]
                ans += [[nums[0]] + item]
        return ans

    """
    78. Subsets (Medium)
    Given an integer array nums of unique elements, return all possible subsets (the power set).
    The solution set must not contain duplicate subsets. Return the solution in any order.
    
    Example 1:
    Input: nums = [1,2,3]
    Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
    
    Example 2:
    Input: nums = [0]
    Output: [[],[0]]
    """

    #  The efficient iterative version.
    def subsets(self, nums: List[int]) -> List[List[int]]:
        ans = [[]]
        for num in nums:
            ans += [subset + [num] for subset in ans]
        return ans

    # #  The naive recursive solution.
    # def subsets(self, nums: List[int]) -> List[List[int]]:
    #      def _subsets(nums: List[int], pos: int, subset: List[int]):
    #         if pos == len(nums):
    #             ans.append( subset )
    #             return
    #         _subsets(nums, pos + 1, subset + [nums[pos]] )
    #         _subsets(nums, pos + 1, subset)
    #
    #      ans = []
    #      _subsets(nums, 0, [])
    #      return ans

    """
    50. Pow(x, n) ( Medium )
    Implement pow(x, n), which calculates x raised to the power n (i.e., xn).
    
    Example 1:
    Input: x = 2.00000, n = 10
    Output: 1024.00000
    
    Example 2:
    Input: x = 2.10000, n = 3
    Output: 9.26100
    
    Example 3:
    Input: x = 2.00000, n = -2
    Output: 0.25000
    Explanation: 2-2 = 1/22 = 1/4 = 0.25
    """

    #  The normal version.
    def myPow(self, x: float, n: int) -> float:
        if n == 0: return 1
        if n < 0: return self.myPow(x, -n)
        half = self.myPow(x, n // 2)
        if n % 2:
            return half * half * x
        return half * half

    # #  The trick version.
    # def myPow(self, x: float, n: int) -> float:
    #     if n == 0: return 1
    #     if n < 0: return 1 / self.myPow( x, -n )
    #     if n % 2:
    #         return x * self.myPow( x , n - 1 )
    #     return self.myPow( x * x, n // 2)

    """
    169. Majority Element (Easy)
    Given an array nums of size n, return the majority element.
    The majority element is the element that appears more than ⌊n / 2⌋ times. 
    You may assume that the majority element always exists in the array.
    
    Example 1:
    Input: nums = [3,2,3]
    Output: 3
    
    Example 2:
    Input: nums = [2,2,1,1,1,2,2]
    Output: 2
    """

    #  The Boyer-Moore Voting Algorithm.
    def majorityElement(self, nums: List[int]) -> int:
        mode, count = None, 0
        for num in nums:
            if count == 0:
                mode = num
            if num == mode:
                count += 1
            else:
                count -= 1
        return mode

    # #  The O(nlgn) solution.
    # def majorityElement(self, nums:List[int]) -> int:
    #     nums.sort()
    #     return nums[ len(nums) // 2]

    # #  Divide-and-conquer solution.
    # def majorityElement(self, nums: List[int]) -> int:
    #     def _majorityElement(p: int, r: int) -> int:
    #         #  Base case
    #         if p == r: return nums[p]
    #
    #         q = (p + r) // 2
    #         left = _majorityElement(p, q)
    #         right = _majorityElement(q + 1, r)
    #         if left == right: return left
    #
    #         left_count = sum(1 for i in range(p, r + 1) if nums[i] == left)
    #         right_count = sum(1 for i in range(p, r + 1) if nums[i] == right)
    #
    #         return (left, right)[right_count > left_count]
    #
    #     return _majorityElement(0, len(nums) - 1)

    """
    229. Majority Element II (Medium)
    Given an integer array of size n, find all elements that appear more than ⌊ n/3 ⌋ times.
    
    Example 1:
    Input: nums = [3,2,3]
    Output: [3]
    
    Example 2:
    Input: nums = [1]
    Output: [1]
    
    Example 3:
    Input: nums = [1,2]
    Output: [1,2]
    """

    #  The Boyer-Moore Voting Algorithm.
    def majorityElement_tri(self, nums: List[int]) -> List[int]:
        mode1, mode2, count1, count2 = math.inf, math.inf, 0, 0
        for num in nums:
            if num == mode1:
                count1 += 1
            elif num == mode2:
                count2 += 1
            elif count1 == 0:
                mode1, count1 = num, 1
            elif count2 == 0:
                mode2, count2 = num, 1
            else:
                count1, count2 = count1 - 1, count2 - 1
        return [ans for ans in (mode1, mode2) if nums.count(ans) > len(nums) // 3]

    """
    17. Letter Combinations of a Phone Number (Medium)
    Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. 
    Return the answer in any order.
    A mapping of digit to letters (just like on the telephone buttons) is given below. 
    Note that 1 does not map to any letters.
    
    Example 1:
    Input: digits = "23"
    Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
    
    Example 2:
    Input: digits = ""
    Output: []
    
    Example 3:
    Input: digits = "2"
    Output: ["a","b","c"]
    """

    #  The backtracking solution.
    def letterCombinations(self, digits: str) -> List[str]:
        def _letterCombinations(digits, pos, words):
            if pos == len(digits):
                ans.append(words)
                return
            for letter in mapping[digits[pos]]:
                _letterCombinations(digits, pos + 1, words + letter)
                # #  Equals the lines below since list words is like a global variable.
                # #  So if you extend it, remember to shorten it back.
                # #  _letterCombinations(digits, pos + 1, words + letter) means use lots of lists rather than
                # #  using only one global list, therefore, no need to shorten anything.
                # words += letter
                # _letterCombinations(digits, pos + 1, words)
                # words = words[:-1]

        if not digits: return digits
        mapping = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
        ans = []
        _letterCombinations(digits, 0, "")
        return ans

    # #  The iterative version using the "dynamic loop".
    # def letterCombinations(self, digits: str) -> List[str]:
    #     if not digits: return digits
    #     mapping = {"2":"abc", "3":"def", "4":"ghi", "5":"jkl", "6":"mno", "7":"pqrs", "8":"tuv", "9":"wxyz"}
    #     ans = [""]
    #     for num in digits:
    #         ans = [ word + letter for word in ans for letter in mapping[num] ]
    #     return ans
    #
    #     # #  Equals to the lines below.
    #     # for num in digits:
    #     #     temp = []
    #     #     for word in ans:
    #     #         for letter in mapping[num]:
    #     #             l = word + letter
    #     #             temp += [l]
    #     #         ans = temp
    #     # return ans

    # #  The recursive solution.
    # def letterCombinations(self, digits: str) -> List[str]:
    #     mapping = {}
    #     mapping["1"] = ""
    #     mapping["2"] = "abc"
    #     mapping["3"] = "def"
    #     mapping["4"] = "ghi"
    #     mapping["5"] = "jkl"
    #     mapping["6"] = "mno"
    #     mapping["7"] = "pqrs"
    #     mapping["8"] = "tuv"
    #     mapping["9"] = "wxyz"
    #
    #     def _letterCombinations(digits):
    #         #  Base case
    #         if not len(digits):
    #             return [""]
    #         return [ letter + item for letter in mapping[digits[0]] for item in _letterCombinations( digits[1:]) ]
    #
    #         # Equals to the lines below:
    #         # ans = []
    #         # for letter in mapping[digits[0]]:
    #         #     for item in _letterCombinations( digits[1:]):
    #         #         ans += [letter + item]
    #         # return ans
    #
    #     if not digits: return digits
    #     return _letterCombinations(digits)

    """
    51. N-Queens (Hard)
    The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that 
    no two queens attack each other.
    Each solution contains a distinct board configuration of the n-queens' placement, 
    where 'Q' and '.' both indicate a queen and an empty space, respectively.
    
    Example 1:
    Input: n = 4
    Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
    
    Example 2:
    Input: n = 1
    Output: [["Q"]]
    """
    #  The backtracking solution.
    #  The idea is using a List[List[int]] to represent the result first.
    #  Then we transfer it into List[List[int]]
    def solveNQueens(self, n: int) -> List[List[str]]:
        def _solveNQueens( i ):
            if i == n:
                ans.append(block[:])
                return

            for j in range(n):
                if j in usedJ or i + j in usedBDiag or i - j in usedDiag:
                    continue
                block.append(j)
                usedJ.add( j )
                usedDiag.add(i - j)
                usedBDiag.add(i + j)

                #  Recursively go to the next line.
                #  The recursive line will "print" an answer if it's valid when reaches the base case.
                #  otherwise the recursive line just do nothing
                _solveNQueens( i + 1 )

                #  Backtracing
                block.pop()
                usedJ.discard(j)
                usedDiag.discard( i - j )
                usedBDiag.discard( i + j )


        #  Print the result.
        def _generateBoard():
            boardlist = []
            for block in ans:
                board = [["."] * n for _ in range(n)]
                for i in range(n):
                    board[i][block[i]] = "Q"
                    board[i] = "".join(board[i][:])
                boardlist += [board]
            return boardlist if ans else ans
            #  Equals the lines below
            # if not ans:
            #     return ans
            #
            # symbols = []
            # for l in ans:
            #     block = []
            #     for i in range(n):
            #         lines = ""
            #         for j in range(n):
            #             if j == l[i]: lines += "Q"
            #             else: lines += "."
            #         block += [lines]
            #     symbols.append(block)
            # return symbols

        ans = []
        block, usedJ, usedDiag, usedBDiag =[], set(),set(), set()
        _solveNQueens(0)
        return _generateBoard()





#  Drive code.
if __name__ == "__main__":
    #  Create an instance
    S = Solution()

    #  Data: lists.
    l0 = [2, 7, 11, 15]
    l1_list = [9, 9, 9, 9, 9, 9, 9]
    l2_list = [9, 9, 9, 9]
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
    l5 = [10, 9, 2, 5, 3, 7, 101, 18]
    l6 = [2, 3, 1, 1, 4]
    l7 = [3, 2, 1, 0, 4]
    l8 = [0, 2, 3]
    l9 = [0, 1, 0, 3, 12]
    l10 = []
    l11 = [0]
    l12 = [-1, 0, 1, 2, -1, -4]
    l13 = [-1, 0, 1, 2, -1, -4, -2, -3, 3, 0, 4]
    l14 = [1, 2, 3, 4, 5]
    l15 = [1, 2]
    l16 = [1, 1, 2]
    l17 = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
    l18 = [1, 1, 1, 2, 2, 3]
    l19 = [0, 0, 1, 1, 1, 1, 2, 3, 3]

    digits01 = [1, 2, 3]
    digits02 = [4, 3, 2, 1]
    digits03 = [0]
    digits04 = [9]

    height01 = [1, 1]
    height02 = [4, 3, 2, 1, 4]
    height03 = [1, 2, 1]
    height04 = [1, 8, 6, 2, 5, 4, 8, 3, 7]

    coins = [1, 2, 5]
    amount = 11

    #  Data: Linked lists.
    l1 = array2Linkedlist(l1_list)
    l2 = array2Linkedlist(l2_list)
    linkedlist01 = array2Linkedlist(l14)
    linkedlist02 = array2Linkedlist(l13)
    linkedlist03 = array2Linkedlist([1, 1, 2])
    linkedlist04 = array2Linkedlist([1, 1, 2, 3, 3])
    linkedlist05 = array2Linkedlist([1, 2, 3, 3, 4, 4, 5])
    linkedlist06 = array2Linkedlist([1, 1, 1, 2, 3])
    linkedlist07 = array2Linkedlist([1, 2, 4])
    linkedlist08 = array2Linkedlist([1, 3, 4])

    print(S.twoSum03(l0, 9))  # Leetcode, 01

    print(linkedlist2Array(S.addTwoNumbers02(l1, l2)))  # Leetcode, 02

    print(S.lengthOfLongestSubstring(string01))  # Leetcode, 03

    print(S.lengthOfLIS(l3))  # Leetcode, 300
    print(S.lengthOfLIS_greedy(l5))  # Leetcode, 300

    print(S.coinChange_iter([2], 3))  # Leetcode, 322

    print(S.uniquePaths(7, 3))  # Leetcode, 62

    print(S.canJump_greedy([1]))  # Leetcode, 55

    print(S.isPalindrome01(string08))  # Leetcode, 125
    print(S.isPalindrome02(".,"))  # Leetcode, 125

    S.moveZeroes03(l8)
    print(l8)  # Leetcode 283

    print(S.maxArea([2, 3, 4, 5, 18, 17, 6]))  # leetcode 11

    print(S.climbStairs(3))  # Leetcode 70

    print(S.threeSum(l13))  # leetcode 15

    print(linkedlist2Array(S.reverseList(linkedlist01)))  # leetcode 206
    print(linkedlist2Array(S.reverseList_iter(linkedlist02)))  # leetcode 206

    print(S.removeDuplicates(l17))  # leetcode 26

    print(S.removeDuplicates02([1, 1, 1, 2, 2, 3]))  # leetcode 80
    print(S.removeDuplicates02_beta(l18))  # leetcode 80
    print(S.removeDuplicates02_theta(l19))  # leetcode 80

    print(linkedlist2Array(S.deleteDuplicates_recur(linkedlist03)))  # Leetcode 83
    print(linkedlist2Array(S.deleteDuplicates_iter(linkedlist04)))  # Leetcode 83

    print(linkedlist2Array(S.deleteDuplicates02_short(linkedlist05)))  # Leetcode 82

    print(S.rotate([-1, -100, 3, 99], 2))  # Leetcode 189
    print(S.rotate01([1, 2, 3, 4, 5, 6, 7], 3))  # Leetcode 189

    print(linkedlist2Array(S.mergeTwoLists(linkedlist07, linkedlist08)))  # Leetcode 21

    print(S.merge([2, 0], 1, [1], 1))  # Leetcode 21

    print(S.plusOne(digits04))  # Leetcode 66
    print(S.plusOne_recur(digits01))  # Leetcode 66

    print(S.isValid("()"))  # Leetcode 20
    print(S.isValid("()[]{}"))  # Leetcode 20
    print(S.isValid("(]"))  # Leetcode 20
    print(S.isValid("]"))  # Leetcode 20

    #  Leetcode 155
    # Your MinStack object will be instantiated and called as such:
    obj = MinStack()
    obj.push(0)
    obj.push(1)
    obj.push(0)
    param01 = obj.getMin()
    # param02 = obj.top()
    obj.pop()
    param03 = obj.getMin()
    print(param01, param03)

    # Leetcode 641
    obj = MyCircularDeque(3)
    param01 = obj.insertLast(1)
    param02 = obj.insertLast(2)
    param03 = obj.insertFront(3)
    param04 = obj.insertFront(4)
    param05 = obj.getRear()
    param06 = obj.isFull()
    param07 = obj.deleteLast()
    param08 = obj.insertFront(4)
    param09 = obj.getFront()
    print(param01, param02, param03, param04, param05, param06, param07, param08, param09)

    #  Leetcode 84
    print(S.largestRectangleArea([2, 1, 5, 6, 2, 3]))
    print(S.largestRectangleArea([2, 4]))

    #  Leetcode 42
    print(S.trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))
    print(S.trap([4, 2, 0, 3, 2, 5]))

    #  Leetcode 239
    print(S.maxSlidingWindow([1, 3, -1, -3, 5, 3, 6, 7], 3))
    print(S.maxSlidingWindow([1], 1))

    #  Leetcode 242
    print(S.isAnagram("anagram", "nagaram"))
    print(S.isAnagram("rat", "car"))

    #  Leetcode 49
    print(S.groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
    print(S.groupAnagrams([""]))
    print(S.groupAnagrams(["a"]))

    r = TreeNode(1, None, TreeNode(2, TreeNode(3), None))
    #  Leetcode 94
    print(S.inorderTraversal(r))

    #  Leetcode 144
    print(S.preorderTraversal(r))

    #  Leetcode 145
    print(S.postorderTraversal(r))

    #  Leetcode 22
    print(S.generateParenthesis(1))
    print(S.generateParenthesis(2))
    print(S.generateParenthesis(3))

    #  Leetcode 449
    root = deserialize(
        '[41,37,44,24,39,42,48,1,35,38,40,null,43,46,49,0,2,30,36,null,null,null,null,null,null,45,47,null,null,null,null,null,4,29,32,null,null,null,null,null,null,3,9,26,null,31,34,null,null,7,11,25,27,null,null,33,null,6,8,10,16,null,null,null,28,null,null,5,null,null,null,null,null,15,19,null,null,null,null,12,null,18,20,null,13,17,null,null,22,null,14,null,null,21,23]')
    #  root = None
    ser = Codec()
    deser = Codec()
    tree = ser.serialize(root)
    deser.deserialize(tree)

    #  Leetcode 105
    print(S.buildTree([3, 9, 20, 15, 7], [9, 3, 15, 20, 7]))

    #  Leetcode 77
    print(S.combine(4, 2))

    #  Leetcode 46
    print(S.permute([1, 2, 3]))

    #  Leetcode 47
    print(S.permuteUnique([1, 1, 2]))
    print(S.permuteUnique([1, 1]))
    print(S.permuteUnique([3, 3, 0, 3]))
    print(S.permuteUnique([2, 2, 1, 1]))

    #  Leetcode 78
    print(S.subsets([1, 2, 3]))
    print(S.subsets([0]))

    #  Leetcode 50
    print(S.myPow(2, -2))
    print(S.myPow(2, 10))
    print(S.myPow(2, 3))
    print(S.myPow(0.00001, 2147483647))

    #  Leetcode 169
    print(S.majorityElement([3, 2, 3]))
    print(S.majorityElement([3, 3, 4]))
    print(S.majorityElement([6, 5, 5]))
    print(S.majorityElement([2, 2, 1, 1, 1, 2, 2]))

    #  Leetcode 229
    print(S.majorityElement_tri([]))
    print(S.majorityElement_tri([1, 2]))
    print(S.majorityElement_tri([0, 0, 0]))
    print(S.majorityElement_tri([4, 1, 2, 3, 4, 4, 3, 2, 1, 4]))

    #  Leetcode 17
    print(S.letterCombinations("23"))
    print(S.letterCombinations("2"))
    print(S.letterCombinations(""))

    print("-------------------")
    #  Leetcode 51
    # print(S.solveNQueens(1))
    # print(S.solveNQueens(2))
    # print(S.solveNQueens(3))
    print(S.solveNQueens(4))

"""
..................佛祖开光 ,永无BUG...................
                        _oo0oo_
                       o8888888o
                       88" . "88
                       (| -_- |)
                       0\  =  /0
                     ___/`---'\___
                   .' \\|     |// '.
                  / \\|||  :  |||// \
                 / _||||| -卍-|||||- \
                |   | \\\  -  /// |   |
                | \_|  ''\---/''  |_/ |
                \  .-\__  '-'  ___/-. /
              ___'. .'  /--.--\  `. .'___
           ."" '<  `.___\_<|>_/___.' >' "".
          | | :  `- \`.;`\ _ /`;.`/ - ` : | |
          \  \ `_.   \_ __\ /__ _/   .-` /  /
      =====`-.____`.___ \_____/___.-`___.-'=====
                        `=---='                       
..................佛祖开光 ,永无BUG...................
"""
