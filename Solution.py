from typing import Optional, List
from bitarray import bitarray
from functools import cache
from heapq import heappush, heappop
from collections import Counter
import bisect, math, collections, itertools, copy, mmh3


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


#  Leetcode 208
class Trie(object):
    def __init__(self):
        self.children = {}

    def insert(self, word: str) -> None:
        node = self.children
        for c in word:
            node[c] = node.get(c, {})
            node = node[c]  # Moving to the next node.
        #  Key "#" means end of the word, Value means the number of searched.
        node["#"] = 0

    def search(self, word: str) -> bool:
        node = self.children
        for c in word:
            if c not in node.keys(): return False
            node = node[c]
        return "#" in node.keys()

    def startsWith(self, prefix: str) -> bool:
        node = self.children
        for c in prefix:
            if c not in node.keys(): return False
            node = node[c]
        return True


#  Bloom Filter
class BloomFilter:
    def __init__(self, size, hash_num):
        self.size = size
        self.hash_num = hash_num
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    def add(self, s):
        for seed in range(self.hash_num):
            index = mmh3.hash(s, seed) % self.size
            self.bit_array[index] = 1

    def lookup(self, s):
        for seed in range(self.hash_num):
            index = mmh3.hash(s, seed) % self.size
            if self.bit_array[index] == 0:
                return "Nope"
        return "Probably"


"""
Leetcode 146( Medium )
https://leetcode.com/problems/lru-cache/#/
"""
class LRUCache:
    def __init__(self, capacity: int):
        self.dic = collections.OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.dic:
            return -1
        value = self.dic.pop(key)
        self.dic[key] = value
        return value

    def put(self, key: int, value: int) -> None:
        if key in self.dic:
            self.dic.pop(key)
        else:
            if self.capacity > 0:
                self.capacity -= 1
            else:
                self.dic.popitem(last=False)
        self.dic[key] = value


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

    # def canJump(self, nums: List[int]) -> bool:
    #     reach = 0
    #     for i, num in enumerate(nums):
    #         if i > reach: return False
    #         reach = max(reach, i + num)
    #     return True

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
        ans = []
        for k in range(len(nums) - 2):
            for i in range(k + 1, len(nums) - 1):
                for j in range(i + 1, len(nums)):
                    key = nums[k] + nums[i] + nums[j]
                    if key == 0:
                        l = sorted([nums[k], nums[i], nums[j]])
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
        j = len(digits) - 1
        while digits[j] + 1 == 10:
            digits[j] = 0
            j -= 1
            if j < 0:
                return [1] + digits
        digits[j] += 1
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
        def _solveNQueens(i):
            if i == n:
                ans.append(block[:])
                return

            for j in range(n):
                if j in usedJ or i + j in usedBDiag or i - j in usedDiag:
                    continue
                block.append(j)
                usedJ.add(j)
                usedDiag.add(i - j)
                usedBDiag.add(i + j)

                #  Recursively go to the next line.
                #  The recursive line will "print" an answer if it's valid when reaches the base case.
                #  otherwise the recursive line just do nothing
                _solveNQueens(i + 1)

                #  Backtracing
                block.pop()
                usedJ.discard(j)
                usedDiag.discard(i - j)
                usedBDiag.discard(i + j)

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
        block, usedJ, usedDiag, usedBDiag = [], set(), set(), set()
        _solveNQueens(0)
        return _generateBoard()

    # #  The concise version.
    # def solveNQueens(self, n: int) -> List[List[str]]:
    #     def _solveQueens(i = 0, seq = []):
    #         if i == n:
    #             ans.append( seq )
    #             return
    #         for j in range(n):
    #             if j in cols or i + j in diags or i - j in backDiags:
    #                 continue
    #             cols.add(j)
    #             diags.add( i + j )
    #             backDiags.add( i - j )
    #             _solveQueens(i + 1, seq + [j])
    #             cols.remove(j)
    #             diags.remove( i + j )
    #             backDiags.remove( i - j )
    #     cols, diags, backDiags, ans = set(), set(), set(), []
    #     _solveQueens()
    #     return [["." * i + "Q" + "." * (n - i - 1) for i in seq] for seq in ans]

    """
    52. N-Queens II ( Hard )
    https://leetcode.com/problems/n-queens-ii/
    """

    def totalNQueens(self, n: int) -> int:
        def _totalNQueens(i=0, cols=0, diags=0, backDiags=0):
            if i == n:
                self.count += 1
                return
            #  Explanation: https://www.cxyxiaowu.com/8990.html
            #  注意，因为我们擅长位运算找1，所以希望1代表可放皇后，0代表不可放皇后。
            #  而此时cols, diags, backDiags的1表示已被占用，不可放皇后， 所以一定要取反
            bits = ~(cols | diags | backDiags) & (1 << n) - 1
            while bits:  # 此时1代表可以放皇后的位置，0表示不可放皇后。
                p = bits & -bits  # 取到最右侧（低位）的1。
                bits = bits & bits - 1  # 消除最右侧（低位）的1 = 此处为0 = 此处放了皇后。
                #  注意cols, diags, backDiags的更新技巧。
                #  注意，此时cols, diags, backDiags的1表示已被占用，不可放皇后。
                _totalNQueens(i + 1, cols | p, (diags | p) >> 1, (backDiags | p) << 1)

        self.count = 0
        _totalNQueens()
        return self.count

    """
    102. Binary Tree Level Order Traversal (Medium)
    Given the root of a binary tree, return the level order traversal of its nodes' values. 
    (i.e., from left to right, level by level).
    
    Example 1:
    Input: root = [3,9,20,null,null,15,7]
    Output: [[3],[9,20],[15,7]]
    
    Example 2:
    Input: root = [1]
    Output: [[1]]
    
    Example 3:
    Input: root = []
    Output: []
    """

    #  The BFS solution.
    def levelOrder02(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root: return root
        queue, ans = [root], []
        while queue:
            ans += [[node.val for node in queue]]
            temp = []
            for node in queue:
                if node.left:
                    temp += [node.left]
                if node.right:
                    temp += [node.right]
            queue = temp
        return ans

    #  The DFS solution.
    def levelOrder02(self, root: Optional[TreeNode]) -> List[List[int]]:
        def _levelOrder02(root: Optional[TreeNode], level: int) -> None:
            #  Base case
            if not root:
                return
            ans[level] = ([root.val] if level not in ans.keys() else ans[level] + [root.val])
            _levelOrder02(root.left, level + 1)
            _levelOrder02(root.right, level + 1)

        ans = {}
        _levelOrder02(root, 0)
        return [value for value in ans.values()]

    """
    515. Find Largest Value in Each Tree Row ( Medium )
    Given the root of a binary tree, return an array of the largest value in each row of the tree (0-indexed).
    
    Example 1:
    Input: root = [1,3,2,5,3,null,9]
    Output: [1,3,9]
    
    Example 2:
    Input: root = [1,2,3]
    Output: [1,3]
    """

    #  The BFS solution.
    def largestValues(self, root: Optional[TreeNode]) -> List[int]:
        if not root: return root
        ans, queue = [], [root]
        while queue:
            ans += [max([node.val for node in queue])]
            temp = []
            for node in queue:
                if node.left: temp += [node.left]
                if node.right: temp += [node.right]
            queue = temp
        return ans

    #  The DFS solution.
    # def largestValues(self, root: Optional[TreeNode])-> List[int]:
    #     def _largestValues( root: Optional[TreeNode], level: int ) -> None:
    #         if not root: return root
    #         ans[level] = ( [root.val] if level not in ans.keys() else ans[level] + [root.val] )
    #         _largestValues( root.left, level + 1 )
    #         _largestValues( root.right, level + 1 )
    #
    #     ans = {}
    #     _largestValues( root, 0 )
    #     return [max(value) for value in ans.values()]

    """
    200. Number of Islands (Medium)
    Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.
    An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. 
    You may assume all four edges of the grid are all surrounded by water.
    
    Example 1:
    Input: grid = [
      ["1","1","1","1","0"],
      ["1","1","0","1","0"],
      ["1","1","0","0","0"],
      ["0","0","0","0","0"]
    ]
    Output: 1
    
    Example 2:
    Input: grid = [
      ["1","1","0","0","0"],
      ["1","1","0","0","0"],
      ["0","0","1","0","0"],
      ["0","0","0","1","1"]
    ]
    Output: 3
    """

    def numIslands(self, grid: List[List[str]]) -> int:
        def floodfill(i, j):
            #  Base case.
            if grid[i][j] == "0": return
            # Process current level
            grid[i][j] = "0"
            #  Recursively process up, down, left, right.
            for k in range(4):
                if 0 <= i + dx[k] < m and 0 <= j + dy[k] < n:
                    floodfill(i + dx[k], j + dy[k])
            return 1

        count, m, n = 0, len(grid), len(grid[0])
        dx, dy = [-1, 1, 0, 0], [0, 0, -1, 1]
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":
                    count += floodfill(i, j)
        return count

    """
    433. Minimum Genetic Mutation (Mutation)
    https://leetcode.com/problems/minimum-genetic-mutation/
    
    Example 1:
    Input: start = "AACCGGTT", end = "AACCGGTA", bank = ["AACCGGTA"]
    Output: 1
    
    Example 2:
    Input: start = "AACCGGTT", end = "AAACGGTA", bank = ["AACCGGTA","AACCGCTA","AAACGGTA"]
    Output: 2
    
    Example 3:
    Input: start = "AAAAACCC", end = "AACCCCCC", bank = ["AAAACCCC","AAACCCCC","AACCCCCC"]
    Output: 3
    """

    #  The concise BFS version.
    #  Use bankSet.remove to replace visited.
    def minMutation(self, start: str, end: str, bank: List[str]) -> int:
        queue, bankSet = [(start, 0)], set(bank)
        for seq, level in queue:
            for newSeq in [seq[:i] + letter + seq[i + 1:] for i in range(len(seq)) for letter in "ACGT"]:
                if newSeq in bankSet:
                    if newSeq == end: return level + 1
                    bankSet.remove(newSeq)
                    queue.append((newSeq, level + 1))
        return -1

    # #  The easy-read BFS version.
    # def minMutation(self, start: str, end: str, bank: List[str]) -> int:
    #     queue,  bankSet, visited, level = [start],  set(bank), set(), 0
    #     while queue:
    #         newQueue = []
    #         for seq in queue:
    #             for i in range(len(seq)):
    #                 for letter in ["A","C","G","T"]:
    #                     newSeq = seq[:i] + letter + seq[i + 1:]
    #                     if newSeq in bankSet and newSeq not in visited:
    #                         if newSeq == end:
    #                             return level + 1
    #                         newQueue += [newSeq]
    #                         visited.add(newSeq)
    #         queue = newQueue
    #         level += 1
    #     return -1

    """
    127. Word Ladder (Hard)
    https://leetcode.com/problems/word-ladder/
    
    Example 1:
    Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
    Output: 5
    Explanation: One shortest transformation sequence is "hit" -> "hot" -> "dot" -> "dog" -> cog", which is 5 words long.
    
    Example 2:
    Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
    Output: 0
    Explanation: The endWord "cog" is not in wordList, therefore there is no valid transformation sequence.
    """

    #  The BFS version.
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wordList, queue, level = set(wordList), [beginWord], 1
        while queue:
            temp = []
            for word in queue:
                for newWord in [word[:i] + letter + word[i + 1:] for i in range(len(word)) for letter in
                                "abcdefghijklmnopqrstuvwxyz"]:
                    #  Equals to the lines below:
                    # for i in range(len(word)):
                    #     for letter in "abcdefghijklmnopqrstuvwxyz":
                    #         newWord = word[:i] + letter + word[i + 1:]
                    if newWord in wordList:
                        if newWord == endWord: return level + 1
                        temp.append(newWord)
                        wordList.remove(newWord)
            queue = temp
            level += 1
        return 0

    #  #  The concise BFS version.
    #  #  Notice: using extending for loop as a queue seems tend to make running time longer!
    # def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
    #     queue = [(beginWord, 1)]
    #     for word, level in queue:
    #         # queue.remove((word, level))
    #         for newWord in [word[:i] + letter + word[i+1:] for i in range(len(word)) for letter in "abcdefghijklmnopqrstuvwxyz"]:
    #             if newWord in wordList:
    #                 if newWord == endWord: return level + 1
    #                 queue.append((newWord, level + 1))
    #                 wordList.remove((newWord))
    #     return 0

    """
    529. Minesweeper (Medium)
    https://leetcode.com/problems/minesweeper/
    
    Example 1:
    Input: board = [["E","E","E","E","E"],["E","E","M","E","E"],["E","E","E","E","E"],["E","E","E","E","E"]], click = [3,0]
    Output: [["B","1","E","1","B"],["B","1","M","1","B"],["B","1","1","1","B"],["B","B","B","B","B"]]
    
    Example 2:
    Input: board = [["B","1","E","1","B"],["B","1","M","1","B"],["B","1","1","1","B"],["B","B","B","B","B"]], click = [1,2]
    Output: [["B","1","E","1","B"],["B","1","X","1","B"],["B","1","1","1","B"],["B","B","B","B","B"]]
    """

    #  The classic DFS solution.
    #  Notice: 8 direction!
    def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
        def minesAround(i, j):
            count = 0
            for k in range(len(dx)):
                if -1 < i + dx[k] < rows and -1 < j + dy[k] < cols:
                    if board[i + dx[k]][j + dy[k]] == "M":
                        count += 1
            return count

        def _updateBoard(i, j):
            #  Base case 1.
            if board[i][j] == "M":
                board[i][j] = "X"
                return
            #  Base case 2.
            if board[i][j] != "E": return

            hint = minesAround(i, j)
            if hint != 0:
                board[i][j] = str(hint)
                return
            board[i][j] = "B"
            for k in range(len(dx)):
                x, y = i + dx[k], j + dy[k]
                if -1 < x < rows and -1 < y < cols:
                    _updateBoard(x, y)

        dx = [-1, 1, 0, 0, -1, -1, 1, 1]
        dy = [0, 0, -1, 1, -1, 1, -1, 1]
        rows, cols = len(board), len(board[0])
        _updateBoard(click[0], click[1])
        return board

    """
    126. Word Ladder II (Hard)
    https://leetcode.com/problems/word-ladder-ii/
    
    Example:
    Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
    Output: [["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]]
    Explanation: There are 2 shortest transformation sequences:
    "hit" -> "hot" -> "dot" -> "dog" -> "cog"
    "hit" -> "hot" -> "lot" -> "log" -> "cog"
    """

    #  The BFS solution with path stored.
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        queue, wordList, ans = [(beginWord, [beginWord])], set(wordList), []
        if beginWord in wordList: wordList.remove(beginWord)
        # not ans is important, means we only return the shortest path.
        while queue and not ans:
            temp, localVisited = [], set()
            for word, path in queue:
                for i in range(len(word)):
                    for letter in "abcdefghijklmnopqrstuvwxyz":
                        newWord = word[:i] + letter + word[i + 1:]
                        if newWord in wordList:
                            if newWord == endWord:
                                ans += [path + [newWord]]
                            temp.append((newWord, path + [newWord]))
                            localVisited.add(newWord)
            queue = temp
            for word in localVisited:
                wordList.remove(word)
        return ans

    # #  The BFS & DFS solution.
    # def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
    #     #  Using BFS to construct a fuzzy path.
    #     def _findLadders(wordList):
    #         queue, wordList, path = [beginWord], set(wordList), {}
    #         if beginWord in wordList: wordList.remove(beginWord)
    #         while queue and endWord not in path.keys():
    #             temp, localVisited = set(), set()
    #             for word in queue:
    #                 for i in range(len(word)):
    #                     for letter in "abcdefghijklmnopqrstuvwxyz":
    #                         newWord = word[:i] + letter + word[i + 1:]
    #                         if newWord in wordList:
    #                             temp.add(newWord)
    #                             path[newWord] = path[newWord] + [word] if newWord in path.keys() else [word]
    #                             localVisited.add(newWord)
    #             queue = list(temp)
    #             for word in localVisited: wordList.remove(word)
    #         return path
    #
    #     #  Using DFS to reconstruct the shortest path.
    #     #  Put all routes into ans.
    #     def dfs4Path(word, route):
    #         #  Base case.
    #         if word == beginWord:
    #             route += [word]
    #             ans.append(route[::-1])
    #             return
    #         if path.get(word) is None: return
    #
    #         for nextWord in path[word]:
    #             dfs4Path(nextWord, route + [word])
    #
    #     ans = []
    #     path = _findLadders(wordList)
    #     dfs4Path(endWord, [])
    #     return ans

    # #  The other version of dfs4Path and its corresponding main code.
    # #  when we use a global route.
    # def dfs4Path(word):
    #     if word == beginWord:
    #         route.append( word )
    #         ans.append(route[::-1])
    #         route.pop()
    #         return
    #     if path.get(word) is None: return
    #     for nextWord in path[word]:
    #         route.append( word )
    #         dfs4Path( nextWord )
    #         route.pop()
    #
    # ans, route = [], []
    # path = _findLadders(wordList)
    # dfs4Path( endWord )
    # return ans

    """
    455. Assign Cookies (Easy)
    https://leetcode.com/problems/assign-cookies/description/
    
    Example:
    Input: g = [1,2,3], s = [1,1]
    Output: 1
    Explanation: You have 3 children and 2 cookies. The greed factors of 3 children are 1, 2, 3. 
    And even though you have 2 cookies, since their size is both 1, you could only make the child whose greed factor is 1 content.
    You need to output 1.
    """

    #  The greedy solution.
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()
        i, j, m, n = 0, 0, len(g), len(s)
        while i < m and j < n:
            if g[i] <= s[j]: i += 1
            j += 1
        return i

    """
    860. Lemonade Change (Easy)
    https://leetcode.com/problems/lemonade-change/description/
    
    Example 01:
    Input: bills = [5,5,5,10,20]
    Output: true
    Explanation: 
    From the first 3 customers, we collect three $5 bills in order.
    From the fourth customer, we collect a $10 bill and give back a $5.
    From the fifth customer, we give a $10 bill and a $5 bill.
    Since all customers got correct change, we output true.
    
    Example 02:
    Input: bills = [5,5,10,10,20]
    Output: false
    Explanation: 
    From the first two customers in order, we collect two $5 bills.
    For the next two customers in order, we collect a $10 bill and give back a $5 bill.
    For the last customer, we can not give the change of $15 back because we only have two $10 bills.
    Since not every customer received the correct change, the answer is false.
    """

    #  The concise solution:
    #  The trick is:
    #  there is no need to use a greedy coin change since only $5, $10, $20 are available.
    #  therefore, we only need $5 & $10 to make changes.
    def lemonadeChange(self, bills: List[int]) -> bool:
        five, ten = 0, 0
        for cash in bills:
            if cash == 5:
                five += 1
            elif cash == 10:
                five, ten = five - 1, ten + 1
            elif ten > 0:
                five, ten = five - 1, ten - 1
            else:
                five -= 3
            if five < 0: return False
        return True

    # #  The naive solution.
    # def lemonadeChange(self, bills: List[int]) -> bool:
    #     def isValidChange(money):
    #         for key in reversed(save.keys()):
    #             if money == 0: return True
    #             x = money // key
    #             if save[key] >= x:
    #                 save[key] -= x
    #                 money %= key
    #         return money == 0
    #
    #     save, require = {5: 0, 10: 0, 20: 0}, 0
    #     for cash in bills:
    #         if not isValidChange(cash - 5): return False
    #         save[cash] += 1
    #     return True

    """
    122. Best Time to Buy and Sell Stock II (Medium)
    https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/description/
    
    Example :
    Input: prices = [7,1,5,3,6,4]
    Output: 7
    Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
    Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
    Total profit is 4 + 3 = 7.
    """

    #  The greedy solution.
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        for i in range(len(prices) - 1):
            if prices[i] < prices[i + 1]: profit += prices[i + 1] - prices[i]
        return profit

    # #  The one-line solution.
    # def maxProfit(self, prices:List[int]) -> int:
    #     return sum([max(prices[i + 1] - prices[i],0) for i in range(len(prices) - 1)])

    # #  The states machine method.
    # #  buy: 当到达i时，1, ..., i中(不一定时i)最后一个操作时buy时的maximum profit.
    # #  sell: 当到达i时，1, ..., i中(不一定时i)最后一个操作时sell时的maximum profit.
    # def maxProfit(self, prices: List[int]) -> int:
    #     if len(prices) == 1: return 0
    #     buy, sell = -prices[0], 0
    #     for price in prices:
    #         buy = max(buy, sell - price)
    #         sell = max(sell, buy + price)
    #     return sell

    """
    874. Walking Robot Simulation (Medium)
    https://leetcode.com/problems/walking-robot-simulation/description/
    
    Example:
    Input: commands = [4,-1,3], obstacles = []
    Output: 25
    Explanation: The robot starts at (0, 0):
    1. Move north 4 units to (0, 4).
    2. Turn right.
    3. Move east 3 units to (3, 4).
    The furthest point the robot ever gets from the origin is (3, 4), which squared is 32 + 42 = 25 units away.
    """

    def robotSim(self, commands: List[int], obstacles: List[List[int]]) -> int:
        direction = {"up": [0, 1, "left", "right"],
                     "down": [0, -1, "right", "left"],
                     "left": [-1, 0, "down", "up"],
                     "right": [1, 0, "up", "down"]}
        curDir = "up"
        x, y, ans = 0, 0, 0
        #  important! To check faster!
        obstacles = set(map(tuple, obstacles))
        for command in commands:
            #  turn right
            if command == -1:
                curDir = direction[curDir][3]
            #  turn left
            elif command == -2:
                curDir = direction[curDir][2]
            else:
                for step in range(command):
                    if (x + direction[curDir][0], y + direction[curDir][1]) in obstacles:
                        break
                    x += direction[curDir][0]
                    y += direction[curDir][1]
                    ans = max(ans, x ** 2 + y ** 2)
        return ans

    """
    45. Jump Game II (Medium)
    https://leetcode.com/problems/jump-game-ii/
    
    Example:
    Input: nums = [2,3,1,1,4]
    Output: 2
    Explanation: The minimum number of jumps to reach the last index is 2. 
    Jump 1 step from index 0 to 1, then 3 steps to the last index.
    """

    #  The greedy solution.
    def jump(self, nums: List[int]) -> int:
        reach, nextReach, count = 0, 0, 0
        for i in range(len(nums)):
            if reach >= len(nums) - 1: return count
            nextReach = max(nextReach, i + nums[i])
            if i == reach:
                reach = nextReach
                count += 1

    """
    Leetcode 69 (Easy)
    https://leetcode.com/problems/sqrtx/
    
    Example 01:
    Input: x = 4
    Output: 2
    
    Example 02:
    Input: x = 8
    Output: 2
    Explanation: The square root of 8 is 2.82842..., and since the decimal part is truncated, 2 is returned.
    """

    #  The binary seach solution.
    def mySqrt(self, x: int) -> int:
        low, high = 1, x
        while low <= high:
            mid = (low + high) // 2
            if mid ** 2 == x:
                return mid
            elif mid ** 2 > x:
                high = mid - 1
            else:
                low = mid + 1
        return high

    # #  The Newton's method.
    # def mySqrt(self, x: int) -> int:
    #     r = x
    #     while r * r > x:
    #         r = math.floor(0.5 * (r + x / r))
    #     return r

    """
    367. Valid Perfect Square (Easy)
    Given a positive integer num, write a function which returns True if num is a perfect square else False.
    
    Example 01:
    Input: num = 16
    Output: true
    
    Example 02:
    Input: num = 14
    Output: false
    """

    #  The binary search solution.
    def isPerfectSquare(self, num: int) -> int:
        low, high = 1, num
        while low <= high:
            mid = (low + high) // 2
            if mid * mid == num:
                return True
            elif mid * mid > num:
                high = mid - 1
            else:
                low = mid + 1
        return False

    # #  The Newton's Method.
    # def isPerfectSquare(self, num: int) -> bool:
    #     x = num
    #     while x * x > num:
    #         x = math.floor( 0.5 * (x + num / x))
    #     return x * x == num

    """
    33. Search in Rotated Sorted Array (Medium)
    https://leetcode.com/problems/search-in-rotated-sorted-array/
    
    Example 01:
    Input: nums = [4,5,6,7,0,1,2], target = 0
    Output: 4
    
    Example 02:
    Input: nums = [4,5,6,7,0,1,2], target = 3
    Output: -1
    
    Example 03:
    Input: nums = [1], target = 0
    Output: -1
    """

    #  The naive solution.
    #  The idea is to find the rule that when should we go to the left subpart.
    #  To be more specific:
    #  if left part is increasing and the target within this range, we go to the left.
    #  Or if right part is increasing and the target doesn't within this range, we go to the right.
    def search(self, nums: List[int], target: int) -> int:
        low, high = 0, len(nums) - 1
        while low <= high:
            mid = (low + high) // 2
            if nums[mid] == target:
                return mid
            elif (nums[low] <= nums[mid] and nums[low] <= target < nums[mid]) or (
                    nums[mid] <= nums[high] and (target < nums[mid] or target > nums[high])):
                high = mid - 1
            else:
                low = mid + 1
        return -1

    """
    153. Find Minimum in Rotated Sorted Array (Medium)
    https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/
    
    Example 01:
    Input: nums = [3,4,5,1,2]
    Output: 1
    Explanation: The original array was [1,2,3,4,5] rotated 3 times.
    
    Example 02:
    Input: nums = [4,5,6,7,0,1,2]
    Output: 0
    Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.
    
    Example 03:
    Input: nums = [11,13,15,17]
    Output: 11
    Explanation: The original array was [11,13,15,17] and it was rotated 4 times. 
    """

    def findMin(self, nums: List[int]) -> int:
        low, high = 0, len(nums) - 1
        while high - low > 1:
            mid = (low + high) // 2
            if nums[low] <= nums[mid] < nums[high]:
                return nums[low]
            elif nums[low] <= nums[mid] > nums[high]:
                low = mid
            else:
                high = mid
        return min(nums[low], nums[high])

    """
    74. Search a 2D Matrix (Medium)
    https://leetcode.com/problems/search-a-2d-matrix/
    
    Example 01:
    Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
    Output: true
    
    Example 02:
    Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
    Output: false
    """

    # The naive binary search solution.
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        def _binarySearch(nums, target):
            p, r = 0, len(nums) - 1
            while p <= r:
                q = (p + r) // 2
                if nums[q] == target:
                    return True
                elif target < nums[q]:
                    r = q - 1
                else:
                    p = q + 1
            return False

        rows, cols = len(matrix), len(matrix[0])
        low, high = 0, rows - 1
        while low <= high:
            mid = (low + high) // 2
            if matrix[mid][0] <= target <= matrix[mid][cols - 1]:
                return _binarySearch(matrix[mid][:], target)
            elif target < matrix[mid][0]:
                high = mid - 1
            else:
                low = mid + 1
        return False

    # #  The library solution.
    # def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    #     rows, cols = len(matrix), len(matrix[0])
    #     #  Determine which row the target stays.
    #     r = bisect.bisect_right([matrix[i][0] for i in range(rows)], target) - 1
    #     #  Find target in row r.
    #     if r < 0: return False  #  We could erase this line to continue search in the last row, still return False.
    #     i = bisect.bisect_right(matrix[r][:], target) - 1
    #     return matrix[r][i] == target

    """
    746. Min Cost Climbing Stairs(Easy)
    https://leetcode.com/problems/min-cost-climbing-stairs/
    
    Example 01:
    Input: cost = [10,15,20]
    Output: 15
    Explanation: You will start at index 1.
    - Pay 15 and climb two steps to reach the top.
    The total cost is 15.
    
    Example 02:
    Input: cost = [1,100,1,1,1,100,1,1,100,1]
    Output: 6
    Explanation: You will start at index 0.
    - Pay 1 and climb two steps to reach index 2.
    - Pay 1 and climb two steps to reach index 4.
    - Pay 1 and climb two steps to reach index 6.
    - Pay 1 and climb one step to reach index 7.
    - Pay 1 and climb two steps to reach index 9.
    - Pay 1 and climb one step to reach the top.
    The total cost is 6.
    """

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        if len(cost) < 3: return min(cost)
        cost.append(0)
        f1, f2, f3 = cost[0], cost[1], 0
        for i in range(2, len(cost)):
            f3 = min(f1, f2) + cost[i]
            f1 = f2
            f2 = f3
        return f3

    """
    1137. N-th Tribonacci Number (Easy)
    https://leetcode.com/problems/n-th-tribonacci-number/
    
    Example:
    Input: n = 4
    Output: 4
    Explanation:
    T_3 = 0 + 1 + 1 = 2
    T_4 = 1 + 1 + 2 = 4
    """

    def tribonacci(self, n: int) -> int:
        ans, t = 0, [0, 1, 1]
        if n < 4: return t[n]
        for i in range(3, n + 1):
            ans = t[0] + t[1] + t[2]
            t[0], t[1], t[2] = t[1], t[2], ans
        return ans

    """
    509. Fibonacci Number (Easy)
    https://leetcode.com/problems/fibonacci-number/
    
    Example:
    Input: n = 3
    Output: 2
    Explanation: F(3) = F(2) + F(1) = 1 + 1 = 2.
    """

    def fib(self, n: int) -> int:
        if n < 2: return n
        f0, f1 = 0, 1
        for _ in range(2, n + 1):
            f1, f0 = f0 + f1, f1
        return f1

    """
    120. Triangle(Medium)
    https://leetcode.com/problems/triangle/description/
    
    Example:
    Input: triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
    Output: 11
    Explanation: The triangle looks like:
       2
      3 4
     6 5 7
    4 1 8 3
    The minimum path sum from top to bottom is 2 + 3 + 5 + 1 = 11 (underlined above).
    """

    #  The dynamic programming solution.
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        for i in reversed(range(len(triangle) - 1)):
            for j in range(len(triangle[i])):
                triangle[i][j] += min(triangle[i + 1][j], triangle[i + 1][j + 1])
        return triangle[0][0]

    # #  The naive recursive solution.
    # def minimumTotal(self, triangle: List[List[int]]) -> int:
    #     rows = len(triangle)
    #     def _minimumTotal(triangle, row, col ):
    #         if row == rows - 1: return triangle[row][col]
    #         return triangle[row][col] + min(_minimumTotal( triangle, row + 1, col ), _minimumTotal( triangle, row + 1, col + 1))
    #
    #     return _minimumTotal(triangle, 0, 0)

    """
    53. Maximum Subarray (Easy)
    https://leetcode.com/problems/maximum-subarray/
    
    Example:
    Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
    Output: 6
    Explanation: [4,-1,2,1] has the largest sum = 6.
    """

    def maxSubArray(self, nums: List[int]) -> int:
        ans = nums[0]
        for i in range(1, len(nums)):
            nums[i] = max(nums[i], nums[i - 1] + nums[i])
            ans = max(ans, nums[i])
        return ans

    # #  The classic Kadane's algorithm with range calculated.
    # def maxSubArray(self, nums: List[int]) -> int:
    #     i, j, ans, dp = 0, 0, nums[0], [0]*len(nums)
    #     dp[0] = nums[0]
    #     for k in range(1,len(nums)):
    #         if nums[k] > dp[k - 1] + nums[k]:
    #             i = k
    #             dp[k] = nums[k]
    #         else:
    #             dp[k] = dp[k - 1] + nums[k]
    #         if dp[k] > ans:
    #             ans = dp[k]
    #             j = k
    #     return ans

    """
    152. Maximum Product Subarray (Medium)
    https://leetcode.com/problems/maximum-product-subarray/description/
    
    Example:
    Input: nums = [2,3,-2,4]
    Output: 6
    Explanation: [2,3] has the largest product 6.
    """

    #  The naive solution.
    def maxProduct(self, nums: List[int]) -> int:
        ans, curMin, curMax = nums[0], nums[0], nums[0]
        for i in range(1, len(nums)):
            curMin, curMax = min(nums[i], nums[i] * curMin, nums[i] * curMax), max(nums[i], nums[i] * curMin,
                                                                                   nums[i] * curMax)
            ans = max(ans, curMax)
        return ans

    """
    198. House Robber (Medium)
    https://leetcode.com/problems/house-robber/
    
    Example 01:
    Input: nums = [1,2,3,1]
    Output: 4
    Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
    Total amount you can rob = 1 + 3 = 4.
    
    Example 02:
    Input: nums = [2,7,9,3,1]
    Output: 12
    Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
    Total amount you can rob = 2 + 9 + 1 = 12.
    """

    #  The classic dp solution.
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1: return nums[0]
        dp = [0] * len(nums)
        dp[0], dp[1] = nums[0], max(nums[0], nums[1])  # Equals to max(dp[0], nums[1] + 0)
        for i in range(2, len(nums)):
            dp[i] = max(dp[i - 1], nums[i] + dp[i - 2])
        return dp[-1]

    # #  The straightforward dp solution with extra memory storage.
    # def rob(self, nums: List[int]) -> int:
    #     memo = [[0] * len(nums), nums ]
    #     for i in range(1, len(nums)):
    #         memo[0][i] = max( memo[0][i - 1], memo[1][i - 1])
    #         memo[1][i] = memo[0][i - 1] + nums[i]
    #     return max(memo[0][-1], memo[1][-1])

    """
    213. House Robber II ( Medium )
    https://leetcode.com/problems/house-robber-ii/
    
    Example 01:
    Input: nums = [2,3,2]
    Output: 3
    Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2), because they are adjacent houses.
    
    Example 02:
    Input: nums = [1,2,3,1]
    Output: 4
    Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
    Total amount you can rob = 1 + 3 = 4.
    
    Example 03:
    Input: nums = [1,2,3]
    Output: 3
    """

    #  The concise solution using previous function.
    def rob_circle(self, nums: List[int]) -> int:
        if len(nums) == 1: return nums[0]
        return max(self.rob(nums[1:]), self.rob(nums[:-1]))

    # #  The on-leetcode solusion version.
    # def rob(self, nums: List[int]) -> int:
    #     def _rob(nums):
    #         f1, f2 = nums[0], max(nums[0], nums[1])
    #         f3 = max(f1, f2)
    #         for i in range(2, len(nums)):
    #             f3 = max(f2, f1 + nums[i])
    #             f1, f2 = f2, f3
    #         return f3
    #
    #     if len(nums) == 1: return nums[0]
    #     if len(nums) == 2: return max(nums[0], nums[1])
    #     return max(_rob(nums[1:]), _rob(nums[:-1]))

    """
    337. House Robber III ( Medium )
    https://leetcode.com/problems/house-robber-iii/
    
    Example 01:
    Input: root = [3,2,3,null,3,null,1]
    Output: 7
    Explanation: Maximum amount of money the thief can rob = 3 + 3 + 1 = 7.
    
    Example 02:
    Input: root = [3,4,5,1,3,null,1]
    Output: 9
    Explanation: Maximum amount of money the thief can rob = 4 + 5 = 9.
    """

    #  The recursive solution with flag indicates whether node can be robbed.
    #  The original recursive idea in the House Rob problem.
    @cache
    def rob_tree01(self, root, canRob=True):
        if not root: return 0
        rob = root.val + self.rob_tree01(root.left, False) + self.rob_tree01(root.right, False) if canRob else -1
        noRob = self.rob_tree01(root.left) + self.rob_tree01(root.right)
        return max(rob, noRob)

    rob_tree01.cache_clear()

    #  The recursive solution without flags.
    #  Implement the final equation: dp[i] = max( dp[i - 1], nums[i] + dp[i - 2]) of the tree version.
    @cache
    def rob_tree02(self, root):
        if not root: return 0
        noRob = self.rob_tree02(root.left) + self.rob_tree02(root.right)
        rob = root.val
        if root.left: rob += self.rob_tree02(root.left.left) + self.rob_tree02(root.left.right)
        if root.right: rob += self.rob_tree02(root.right.left) + self.rob_tree02(root.right.right)
        return max(rob, noRob)

    rob_tree02.cache_clear()

    #  If we analyze rob_tree01 & rob_tree02,
    #  Both of them traversal the tree twice since we want to compare rob and noRob.
    #  So why not traversal the tree one time and records 2 numbers represent noRob and rob.
    #  The optimized dp solution.
    def rob_tree03(self, root):
        def _rob(root):
            if not root: return [0, 0]
            l = _rob(root.left)
            r = _rob(root.right)
            return [max(l) + max(r), root.val + l[0] + r[0]]

        return max(_rob(root))

    """
    740. Delete and Earn ( Medium )
    https://leetcode.com/problems/delete-and-earn/
    
    Example 01:
    Input: nums = [3,4,2]
    Output: 6
    Explanation: You can perform the following operations:
    - Delete 4 to earn 4 points. Consequently, 3 is also deleted. nums = [2].
    - Delete 2 to earn 2 points. nums = [].
    You earn a total of 6 points.
    
    Example 02:
    Input: nums = [2,2,3,3,3,4]
    Output: 9
    Explanation: You can perform the following operations:
    - Delete a 3 to earn 3 points. All 2's and 4's are also deleted. nums = [3,3].
    - Delete a 3 again to earn 3 points. nums = [3].
    - Delete a 3 once more to earn 3 points. nums = [].
    You earn a total of 9 points.
    """

    #  The concise version.
    def deleteAndEarn(self, nums: List[int]) -> int:
        aux = [0] * (max(nums) + 1)
        for num in nums:
            aux[num] += num
        return self.rob(aux)

    # #  The original solution.
    # def deleteAndEarn(self, nums: List[int]) -> int:
    #     nums.sort()
    #     aux = [0] * (nums[-1] + 1)
    #     i = 0
    #     while i < len(nums):
    #         j, pivot, count = i + 1, nums[i], nums[i]
    #         while j < len(nums) and nums[j] == pivot:
    #             count += pivot
    #             j += 1
    #         aux[pivot] = count
    #         i = j
    #     return self.rob( aux )

    """
    2140. Solving Questions With Brainpower (Medium)
    https://leetcode.com/problems/solving-questions-with-brainpower/
    
    Example 01:
    Input: questions = [[3,2],[4,3],[4,4],[2,5]]
    Output: 5
    Explanation: The maximum points can be earned by solving questions 0 and 3.
    - Solve question 0: Earn 3 points, will be unable to solve the next 2 questions
    - Unable to solve questions 1 and 2
    - Solve question 3: Earn 2 points
    Total points earned: 3 + 2 = 5. There is no other way to earn 5 or more points.
    
    Example 02:
    Input: questions = [[1,1],[2,2],[3,3],[4,4],[5,5]]
    Output: 7
    Explanation: The maximum points can be earned by solving questions 1 and 4.
    - Skip question 0
    - Solve question 1: Earn 2 points, will be unable to solve the next 2 questions
    - Unable to solve questions 2 and 3
    - Solve question 4: Earn 5 points
    Total points earned: 2 + 5 = 7. There is no other way to earn 7 or more points.
    """

    #  The recursive solution.
    def mostPoints(self, questions: List[List[int]]) -> int:
        @cache
        def _mostPoints(i: int):
            if i > len(questions) - 1: return 0
            return max(questions[i][0] + _mostPoints(i + 1 + questions[i][1]), _mostPoints(i + 1))

        return _mostPoints(0)

    # #  The iterative dp solution.
    # def mostPoints(self, questions: List[List[int]]) -> int:
    #     n = len(questions)
    #     dp = [0] * n
    #     dp[-1] = questions[-1][0]
    #     for i in reversed(range(n - 1)):
    #         idx = i + 1 + questions[i][1]
    #         if idx > n - 1:
    #             dp[i] = max( dp[i + 1], questions[i][0])
    #         else:
    #             dp[i] = max( dp[i + 1], questions[i][0] + dp[idx])
    #     return dp[0]

    """
    121. Best Time to Buy and Sell Stock ( Easy )
    https://leetcode.com/problems/best-time-to-buy-and-sell-stock/#/description
    
    Example 01:
    Input: prices = [7,1,5,3,6,4]
    Output: 5
    Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
    Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
    
    Example 02:
    Input: prices = [7,6,4,3,1]
    Output: 0
    Explanation: In this case, no transactions are done and the max profit = 0.
    """

    #  The straightforward intuitive solution.
    def maxProfit121(self, prices: List[int]) -> int:
        curMin, ans = math.inf, 0
        for num in prices:
            curMin = min(curMin, num)
            ans = max(ans, num - curMin)
        return ans

    # #  The states machine method.
    # #  buy: 到达i时，在1,...,i中（不一定必须是i）最后一个操作是buy的maximum profit。
    # #  sell: 到达i时，在1,...,i中（不一定必须是i）最后一个操作是sell的maximum profit。
    # def maxProfit121(self, prices: List[int]) -> int:
    #     buy, sell = -prices[0], -math.inf
    #     for price in prices:
    #         buy = max(buy, -price)  # do nothing or buy at current price.
    #         sell = max(sell, buy + price)  #  do nothing or sell at current price.
    #     return sell

    """
    123. Best Time to Buy and Sell Stock III ( Hard )
    https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/
    
    Example 01:
    Input: prices = [3,3,5,0,0,3,1,4]
    Output: 6
    Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
    Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 = 3.
    
    Example 02:
    Input: prices = [1,2,3,4,5]
    Output: 4
    Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
    Note that you cannot buy on day 1, buy on day 2 and sell them later, 
    as you are engaging multiple transactions at the same time. You must sell before buying again.
    
    Example 03:
    Input: prices = [7,6,4,3,1]
    Output: 0
    Explanation: In this case, no transaction is done, i.e. max profit = 0.
    """

    #  The states machine method.
    def maxProfit123(self, prices: List[int]) -> int:
        if not prices: return 0
        s1, s2, s3, s4 = -math.inf, -math.inf, -math.inf, -math.inf
        for price in prices:
            s1 = max(s1, -price)
            s2 = max(s2, s1 + price)
            s3 = max(s3, s2 - price)
            s4 = max(s4, s3 + price)
        return s4

    """
    188. Best Time to Buy and Sell Stock IV ( Hard )
    https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/
    
    Example 01:
    Input: k = 2, prices = [2,4,1]
    Output: 2
    Explanation: Buy on day 1 (price = 2) and sell on day 2 (price = 4), profit = 4-2 = 2.
    
    Example 02:
    Input: k = 2, prices = [3,2,6,5,0,3]
    Output: 7
    Explanation: Buy on day 2 (price = 2) and sell on day 3 (price = 6), profit = 6-2 = 4. 
    Then buy on day 5 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
    """

    def maxProfit188(self, k: int, prices: List[int]) -> int:
        if not prices: return 0
        dp = [0] + [-math.inf for _ in range(2 * k)]
        for price in prices:
            for i in range(1, len(dp), 2):
                dp[i] = max(dp[i], dp[i - 1] - price)
                dp[i + 1] = max(dp[i + 1], dp[i] + price)
        return dp[-1]

    """
    309. Best Time to Buy and Sell Stock with Cooldown (Medium)
    https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/
    
    Example:
    Input: prices = [1,2,3,0,2]
    Output: 3
    Explanation: transactions = [buy, sell, cooldown, buy, sell]
    """

    #  The states machine method.
    #  buy: 到达i时，0, ..., i中（不一定时i）最后一个操作是 buy / 起始buy 的maximum profit.
    #  sell: 到达i时，0, ..., i中（不一定时i）最后一个操作是 sell / 起始sell 的maximum profit.
    #  cooldown: 到达i时，0, ..., i中（不一定时i）最后一个操作是 cooldown / 起始cooldown 的 最后两个 maximum profit.
    #  cooldown为一个左进右出的queue, 最近的一次cooldown为cooldown[0], 上一次cooldown为cooldown[1]
    def maxProfit309(self, prices: List[int]) -> int:
        if len(prices) == 1: return 0
        buy, sell, cooldown = -prices[0], 0, [0, 0]
        for price in prices:
            buy = max(buy, cooldown[1] - price)  # do nothing or buy from the second last cooldown.
            sell = max(sell, buy + price)  # do nothing or sell at the current price.
            cooldown = [sell, cooldown[0]]  # Update the queue to move one spot to the right.
        return cooldown[0]

    """
    714. Best Time to Buy and Sell Stock with Transaction Fee(Medium)
    https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/
    
    Example 01:
    Input: prices = [1,3,2,8,4,9], fee = 2
    Output: 8
    Explanation: The maximum profit can be achieved by:
    - Buying at prices[0] = 1
    - Selling at prices[3] = 8
    - Buying at prices[4] = 4
    - Selling at prices[5] = 9
    The total profit is ((8 - 1) - 2) + ((9 - 4) - 2) = 8.
    
    Example 02:
    Input: prices = [1,3,7,5,10,3], fee = 3
    Output: 6
    """

    def maxProfit714(self, prices: List[int], fee: int) -> int:
        if len(prices) == 1: return 0
        buy, sell = -prices[0], 0
        for price in prices:
            buy = max(buy, sell - price)
            sell = max(sell, buy + price - fee)
        return sell

    """
    221. Maximal Square (Medium)
    https://leetcode.com/problems/maximal-square/
    
    Example 01:
    Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],
    ["1","0","0","1","0"]]
    Output: 4
    
    Example 02:
    Input: matrix = [["0","1"],["1","0"]]
    Output: 1
    
    Example 03:
    Input: matrix = [["0"]]
    Output: 0
    """

    #  The bottom-up iterative solution.
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        ans, rows, cols = 0, len(matrix), len(matrix[0])
        matrix = [[int(matrix[i][j]) for j in range(cols)] for i in range(rows)]
        for i in range(rows):
            for j in range(cols):
                if i > 0 and j > 0 and matrix[i][j]:
                    matrix[i][j] = min(matrix[i - 1][j], matrix[i][j - 1], matrix[i - 1][j - 1]) + 1
                ans = max(ans, matrix[i][j])
        return ans * ans

    # #  The recursive dp solution with a memo.
    # def maximalSquare(self, matrix: List[List[str]]) -> int:
    #     @cache
    #     def _maximalSquare(i, j):
    #         if 0 <= i < rows and 0 <= j < cols and matrix[i][j]:
    #             return min(_maximalSquare( i - 1, j), _maximalSquare(i, j - 1), _maximalSquare(i - 1, j - 1)) + 1
    #         return 0
    #
    #     ans, rows, cols = 0, len(matrix), len(matrix[0])
    #     matrix = [[int(matrix[i][j]) for j in range(cols)] for i in range(rows)]
    #     for i in range(rows):
    #         for j in range(cols):
    #             ans = max( ans, _maximalSquare(i, j))
    #     return ans * ans

    """
    91. Decode Ways (Medium)
    https://leetcode.com/problems/decode-ways/
    
    Example 01：
    Input: s = "12"
    Output: 2
    Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).
    
    Example 02:
    Input: s = "226"
    Output: 3
    Explanation: "226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).
    
    Example 03:
    Input: s = "06"
    Output: 0
    Explanation: "06" cannot be mapped to "F" because of the leading zero ("6" is different from "06").
    """

    #  The naive recursive dp solution.
    def numDecodings(self, s: str) -> int:
        #  Base case
        #  Key idea: single "0" is impossible.
        if not s: return 1  # after subtract letter from 10 to 26
        if len(s) == 1:  # the last letter from 1 to 9
            return 1 if s != "0" else 0
        ans = 0
        if 1 <= int(s[-1]) <= 9:
            ans += self.numDecodings(s[:-1])
        if 10 <= int(s[-2:]) <= 26:
            ans += self.numDecodings(s[:-2])
        return ans

    # #  The recursive dp solution with a memo.
    # def numDecodings(self, s: str) -> int:
    #     @cache
    #     def _numDecoding( k ):
    #         if k < 0:
    #             return 1
    #         if k == 0:
    #             return 1 if s[0] != "0" else 0
    #         ans = 0
    #         if  1 <= int(s[ k ]) <= 9:
    #             ans += _numDecoding(k - 1)
    #         if 10 <= int(s[k - 1 : k + 1 ]) <= 26:
    #             ans += _numDecoding(k - 2)
    #         return ans
    #     return _numDecoding( len(s) - 1 )

    # #  The iterative dp solution.
    # def numDecodings(self, s: str) -> int:
    #     dp = [0] * (len(s) + 1)
    #     dp[0] = 1
    #     dp[1] = 1 if s[0] != "0" else 0
    #     for i in range(2, len(dp)):
    #         if 1 <= int(s[i - 1] )<= 9:
    #             dp[i] += dp[i - 1]
    #         if 10 <= int(s[i - 2 : i]) <= 26:
    #             dp[i] += dp[i - 2]
    #     return dp[-1]

    """
    64. Minimum Path Sum (Medium)
    https://leetcode.com/problems/minimum-path-sum/
    
    Example 01:
    Input: grid = [[1,3,1],[1,5,1],[4,2,1]]
    Output: 7
    Explanation: Because the path 1 → 3 → 1 → 1 → 1 minimizes the sum.
    
    Example 02:
    Input: grid = [[1,2,3],[4,5,6]]
    Output: 12
    """

    #  The iterative dp solution.
    def minPathSum(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        for j in range(1, cols):
            grid[0][j] += grid[0][j - 1]
        for i in range(1, rows):
            grid[i][0] += grid[i - 1][0]
        for i in range(1, rows):
            for j in range(1, cols):
                grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
        return grid[rows - 1][cols - 1]

    # #  The recursive dp solution with a memo.
    # def minPathSum(self, grid: List[List[int]]) -> int:
    #     @cache
    #     def _minPathSum( i, j ):
    #         if i < 0 or j < 0: return math.inf
    #         if i == 0 and j == 0: return grid[0][0]
    #         return min(_minPathSum(i - 1, j), _minPathSum(i, j - 1) ) + grid[i][j]
    #     return _minPathSum( len(grid) - 1, len(grid[0]) - 1)

    """
    72. Edit Distance (Hard)
    https://leetcode.com/problems/edit-distance/
    
    Example 01:
    Input: word1 = "horse", word2 = "ros"
    Output: 3
    Explanation: 
    horse -> rorse (replace 'h' with 'r')
    rorse -> rose (remove 'r')
    rose -> ros (remove 'e')
    
    Example 02:
    Input: word1 = "intention", word2 = "execution"
    Output: 5
    Explanation: 
    intention -> inention (remove 't')
    inention -> enention (replace 'i' with 'e')
    enention -> exention (replace 'n' with 'x')
    exention -> exection (replace 'n' with 'c')
    exection -> execution (insert 'u')
    """

    #  The recursive dp solution with a memo.
    def minDistance(self, word1: str, word2: str) -> int:
        @cache
        def _minDistance(i, j):
            #  Base case
            if i < 0 and j < 0: return 0
            if i < 0 and j >= 0: return j + 1
            if i >= 0 and j < 0: return i + 1
            if word1[i] == word2[j]:
                return _minDistance(i - 1, j - 1)
            else:
                remove = _minDistance(i - 1, j)
                #  The delete of word2 equals the insert of word1.
                insert = _minDistance(i, j - 1)
                replace = _minDistance(i - 1, j - 1)
                return min(remove, insert, replace) + 1

        return _minDistance(len(word1) - 1, len(word2) - 1)

    # #  The iterative dp solution.
    # def minDistance(self, word1: str, word2: str) -> int:
    #     m, n = len(word1), len(word2)
    #     dp = [[0] * (n + 1) for _ in range(m + 1)]
    #     for i in range(m + 1): dp[i][0] = i
    #     for j in range(n + 1): dp[0][j] = j
    #     for i in range(1, m + 1):
    #         for j in range(1, n + 1):
    #             if word1[i - 1] == word2[j - 1]:
    #                 dp[i][j] = dp[i - 1][j - 1]
    #             else:
    #                 dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    #     return dp[-1][-1]

    """
    32. Longest Valid Parentheses (Hard)
    https://leetcode.com/problems/longest-valid-parentheses/
    
    Example 01:
    Input: s = "(()"
    Output: 2
    Explanation: The longest valid parentheses substring is "()".
    
    Example 02:
    Input: s = ")()())"
    Output: 4
    Explanation: The longest valid parentheses substring is "()()".
    
    Example 03:
    Input: s = ""
    Output: 0
    """

    #  The solution using a stack.
    #  ()() & ((...)) only these two cases are consecutive valid parentheses.
    #  The idea is using a stack to store all the "(" or the only ")" before the longest ((...))
    def longestValidParentheses(self, s: str) -> int:
        ans, stack = 0, [-1]
        for i, p in enumerate(s):
            if p == ")":
                stack.pop()
                if stack:
                    ans = max(ans, i - stack[-1])
                    continue
            stack.append(i)
        return ans

    # #  The recursive dp solution.
    # def longestValidParentheses(self, s: str) -> int:
    #     #  _longestValidParentheses(j) represents the longest valid parentheses ends with s[j]
    #     @cache
    #     def _longestValidParentheses( j ):
    #         #  Base case
    #         if j <= 0 or s[j] == "(": return 0
    #         if s[j] == ")":
    #             if s[j - 1] == "(":  # case: "()"
    #                 return 2 + _longestValidParentheses( j - 2 )
    #             else:  #  case: "...))"
    #                 index = j - _longestValidParentheses( j - 1 ) - 1
    #                 if index >= 0 and s[index] == "(":  #  case: "((...))"
    #                     return _longestValidParentheses( index - 1 ) +_longestValidParentheses( j - 1 ) + 2
    #                 else:  # case: ")(....))"
    #                     return 0
    #     return max([_longestValidParentheses(j) for j in reversed(range(len(s)))]) if s else 0

    # #  The iterative dp solution.
    # def longestValidParentheses(self, s: str) -> int:
    #     if not s: return 0
    #     dp = [0] * len(s)
    #     for i in range( 1, len(s) ):
    #         if s[i] == "(" : continue
    #         if s[i] == ")" and s[i - 1] == "(":
    #             dp[i] = dp[i - 2] + 2
    #         else:
    #             index = i - dp[i - 1] - 1
    #             if index >= 0 and s[index] == "(":
    #                 dp[i] = dp[index - 1] + dp[i - 1] + 2
    #     return max(dp)

    """
    621. Task Scheduler (Medium)
    https://leetcode.com/problems/task-scheduler/
    
    Example 01:
    Input: tasks = ["A","A","A","B","B","B"], n = 2
    Output: 8
    Explanation: 
    A -> B -> idle -> A -> B -> idle -> A -> B
    There is at least 2 units of time between any two same tasks.
    
    Example 02:
    Input: tasks = ["A","A","A","B","B","B"], n = 0
    Output: 6
    Explanation: On this case any permutation of size 6 would work since n = 0.
    ["A","A","A","B","B","B"]
    ["A","B","A","B","A","B"]
    ["B","B","B","A","A","A"]
    ...
    And so on.
    
    Example 03:
    Input: tasks = ["A","A","A","A","A","A","B","C","D","E","F","G"], n = 2
    Output: 16
    Explanation: 
    One possible solution is
    A -> B -> C -> A -> D -> E -> A -> F -> G -> A -> idle -> idle -> A -> idle -> idle -> A
    """

    #  The tricky solution.
    #  Find the most frequent letter, define the frequency = m
    #  Case 1: A(..n..) A... A... A   ans = (n + 1) * (m - 1) + 1
    #          But when several letters have the same biggest frequency m > n,
    #          ABC(..n..) ABC... ABC... ABC  ans = (n + 1) * (m - 1) + #letters with the biggest frquency.
    #  Case 2: ABABAB   ans = len(tasks)
    def leastInterval(self, tasks: List[str], n: int) -> int:
        counts = list(Counter(tasks).values())  # Counter(tasks) -> dict-like structure.
        maxCounts = max(counts)  # Get the most frequency.
        factor = counts.count(maxCounts)  # Get the number of letters with the biggest frquency.
        return max(len(tasks), (n + 1) * (maxCounts - 1) + factor)

    # #  The mimic solution.
    # #  The idea is to reduce idle as much as possible.
    # #  Which means we should always process the tasks with bigger frequency.
    # def leastInterval(self, tasks: List[str], n: int) -> int:
    #     ans, heap = 0, []
    #     #  Initialize the heap
    #     #  Notice: heapq satisfy heap[k] <= heap[2*k+1] and heap[k] <= heap[2*k+2].
    #     #  Sort by the first value of the heap node.
    #     for task, freq in Counter(tasks).items():
    #         heappush( heap, (freq * -1, task) )
    #
    #     while heap:
    #         counts, nextHeap = 0, []
    #         #  Inner loop: count each n + 1 sequence.
    #         while counts <= n:
    #             #  Base case: finished when counts <= n.
    #             #  e.q. AB_AB_AB
    #             if not heap and not nextHeap: return ans
    #             if heap:
    #                 freq, task = heappop( heap )
    #                 #  Remove the task when its frequency is 1.
    #                 if freq != -1:
    #                     nextHeap.append((freq + 1, task))
    #             counts += 1
    #             ans += 1
    #         #  Update heap.
    #         for item in nextHeap:
    #            heappush( heap, item )
    #     return ans

    """
    403. Frog Jump (Hard)
    https://leetcode.com/problems/frog-jump/
    """

    #  Try to avoid TLE, so using a set.
    #  MUST define the range of the set!
    #  The straightforward solution.
    def canCross(self, stones: List[int]) -> bool:
        if stones[1] != 1:
            return False
        steps = {x: set() for x in stones}
        steps[1].add(1)  # Reach the 1 unit stone with maximum 1 step.
        for i in range(1, len(stones)):
            for step in steps[stones[i]]:
                for length in range(step - 1, step + 2):
                    if length > 0 and stones[i] + length in steps:
                        steps[stones[i] + length].add(length)
        return steps[stones[-1]] != set()

    # #  The dfs solution.
    # def canCross(self, stones: List[int]) -> bool:
    #     @cache
    #     def _canCross( cur, step ):
    #         if step <= 0: return False
    #         if cur not in setStones: return False
    #         if cur == stones[-1]: return True
    #         left = _canCross(cur + step - 1, step - 1)
    #         mid = _canCross(cur + step, step)
    #         right = _canCross(cur + step + 1, step + 1)
    #         return left or mid or right
    #
    #     # #  Using a self-memo
    #     # visited = set()
    #     # def _canCross( cur, step ):
    #     #     if cur == stones[-1]:
    #     #         return True
    #     #     if (cur, step) in visited:
    #     #         return False
    #     #     visited.add((cur, step))
    #     #     left = mid = right = False
    #     #     if step - 1 > 0 and cur + step - 1 in setStones:
    #     #         left = _canCross( cur + step - 1, step - 1 )
    #     #     if cur + step in setStones:
    #     #         mid = _canCross( cur + step, step )
    #     #     if cur + step + 1 in setStones:
    #     #         right = _canCross( cur + step + 1, step + 1)
    #     #     return left or mid or right
    #
    #     if stones[1] != 1: return False
    #     setStones = set(stones)
    #     return _canCross( 1, 1 )

    """
    552. Student Attendance Record II (Hard)
    https://leetcode.com/problems/student-attendance-record-ii/
    """

    # #  The naive straightforward solution, O(2^n).
    # #  TLE.
    # #  Maximum recursion depth exceeded.
    # def checkRecord(self, n: int) -> int:
    #     @cache
    #     def _checkRecord( i, hasA, twoL ):
    #         if i > n: return 1
    #         #  Choose P
    #         count = _checkRecord( i + 1, hasA, 0 )
    #         #  Choose A
    #         if hasA == False:
    #             count += _checkRecord( i + 1, True, 0)
    #          #  Choose L
    #         if twoL != 2:
    #             count += _checkRecord( i + 1, hasA, twoL + 1 )
    #         return count
    #
    #     return _checkRecord(1, False, 0 )

    #  dp[i][j] = the number of possible attendance records of length i, ends of j (= "A" or "L" or "P").
    #  So in the end, return sum(dp[n][j])

    #          the i th day                 <==       the j of the i - 1 th day could be
    #  dp[i][0] = without A, ends with a A.                   1, 2, 3
    #  dp[i][1] = without A, ends with a P.                   1, 2, 3
    #  dp[i][2] = without A, ends with a L.                   1
    #  dp[i][3] = without A, ends with two LLs.               2
    #  dp[i][4] = with A, ends with a P.                      0, 4, 5, 6
    #  dp[i][5] = with A, ends with a L.                      0, 4
    #  dp[i][6] = with A, ends with a two LLs.                5
    def checkRecord(self, n: int) -> int:
        MOD = 1000000007
        dp = [[0] * 7 for _ in range(n)]
        dp[0][0] = dp[0][1] = dp[0][2] = 1
        for i in range(1, n):
            dp[i][0] = (dp[i - 1][1] + dp[i - 1][2] + dp[i - 1][3]) % MOD
            dp[i][1] = (dp[i - 1][1] + dp[i - 1][2] + dp[i - 1][3]) % MOD
            dp[i][2] = dp[i - 1][1]
            dp[i][3] = dp[i - 1][2]
            dp[i][4] = (dp[i - 1][0] + dp[i - 1][4] + dp[i - 1][5] + dp[i - 1][6]) % MOD
            dp[i][5] = (dp[i - 1][0] + dp[i - 1][4]) % MOD
            dp[i][6] = dp[i - 1][5]
        return sum(dp[n - 1][j] for j in range(7)) % MOD

    """
    410. Split Array Largest Sum (Medium)
    https://leetcode.com/problems/split-array-largest-sum/
    """

    #  The Binary search solution.
    def splitArray(self, nums: List[int], m: int) -> int:
        def validSplit(ceil):
            count, subarray = 1, 0
            for num in nums:
                if subarray + num <= ceil:
                    subarray += num
                else:
                    subarray = num
                    count += 1
            return count <= m

        left = max(nums)
        right = sum(nums)
        while left < right:
            mid = (left + right) // 2
            if validSplit(mid):
                right = mid
            else:
                left = mid + 1
        return left

    # #  The dp solution but TLE.
    # def splitArray(self, nums: List[int], m: int) -> int:
    #     dp = [[0] * m for _ in range(len(nums))]
    #     dp[0][0] = nums[0]
    #     for i in range(len(nums)):
    #         dp[i][0] = dp[i - 1][0] + nums[i]
    #     for i in range(len(nums)):
    #         for j in range(1, m):
    #             dp[i][j] = math.inf
    #     for j in range(1, m):
    #         for i in range(len(nums)):
    #             for k in range(i):
    #                 x = dp[k][j - 1]
    #                 y = sum(nums[k + 1: i + 1])
    #                 dp[i][j] = min(dp[i][j], max(dp[k][j - 1], sum(nums[k + 1: i + 1])))
    #     return dp[-1][-1]

    """
    647. Palindromic Substrings ( Medium )
    https://leetcode.com/problems/palindromic-substrings/
    """

    #  The straightforward recursive solution with a memo.
    #  Check every boundary i & j to the center.   i ----> center <---- j
    def countSubstrings(self, s: str) -> int:
        @cache
        def isPalindrome(i, j):
            if i > j: return True
            if s[i] != s[j]: return False
            return isPalindrome(i + 1, j - 1)

        count, n = 0, len(s)
        for i in range(n):
            for j in range(i, n):
                if isPalindrome(i, j):
                    count += 1
        return count

    # #  The iterative solution.
    # #  Check every odd & even substring from the center to the boundary.
    # def countSubstrings(self, s: str) -> int:
    #     def countFromCen(i, j):
    #         count = 0
    #         while i >= 0 and j < n and s[i] == s[j]:
    #             count += 1
    #             i -= 1
    #             j += 1
    #         return count
    #
    #     n, ans = len(s), 0
    #     for i in range(n):
    #         ans += countFromCen( i, i )  #  For the odd substrings
    #         ans += countFromCen( i, i + 1)  #  For the even substrings.
    #     return ans

    """
    76. Minimum Window Substring (Hard)
    https://leetcode.com/problems/minimum-window-substring/
    """

    #  Sliding window --> Two pointers.
    #  The right pointer to expand / move the window.
    #  The left pointer to shrink / constrain the window.

    #  Two more things to consider:
    #  1. We don't care about the order means using a dict.
    #     1) How to cope with the duplicated elements.
    #  2. How to confirm that the window has all letters in t.
    def minWindow(self, s: str, t: str) -> str:
        target = collections.Counter(t)
        left, right, count, n, ans = 0, 0, math.inf, len(t), ""
        while right < len(s):
            #  When target[s[right]] <= 0 means s has more the same character than them in t.
            #  Such as: s: ABAAC   t: ABC
            if target[s[right]] > 0:
                n -= 1
            target[s[right]] -= 1

            #  When we find a valid window.
            while n == 0:
                #  Update the minimum window.
                if not ans or right - left + 1 < count:
                    count = right - left + 1
                    ans = s[left: right + 1]
                #  Moving left
                target[s[left]] += 1
                if target[s[left]] > 0:
                    n += 1
                left += 1
            right += 1
        return ans

    """
    363. Max Sum of Rectangle No Larger Than K (Hard)
    https://leetcode.com/problems/max-sum-of-rectangle-no-larger-than-k/
    """

    # #  The brute-force solution.
    # #  TLE - O(m^2 n^2)
    # #  Check every rectangle which means we need track the left-top and right-end indices.
    # #  Assume we have a left-top, then (i, j) is the right-end point.
    # #  dp[i, j] = the sum of the area from the left-top point to the right-end point.
    # #  转移方程：
    # #  Let's assume that we know the dp table, how to calculate matrix[i][j]?
    # #  matrix[i][j] = dp[i][j] - dp[i-1][j] - dp[i][j-1] + dp[i-1][j-1]
    # #  Transform the equation we get the 转移方程：
    # #  dp[i][j] = dp[i-1][j] + dp[i][j-1] - dp[i-1][j-1] + matrix.
    # def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
    #     rows, cols, ans = len(matrix), len(matrix[0]), 0
    #     for row in range(1, rows + 1):
    #         for col in range(1, cols + 1):
    #             #  (row, col) is the left-top point.
    #             #  Initialize the dp table.
    #             dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    #             dp[row][col] = matrix[row - 1][col - 1]
    #             #  (i, j) is the right-end point.
    #             for i in range(row, rows + 1):
    #                 for j in range(col, cols + 1):
    #                     dp[i][j] = dp[i - 1][j] + dp[i][j - 1] - dp[i - 1][j - 1] + matrix[i - 1][j - 1]
    #                     if dp[i][j] <= k:
    #                         ans = max(ans, dp[i][j])
    #     return ans

    #  The solution using prefix to reduce one loop.
    #  The idea is that when we fix two cols,
    #  the sum of prefix of their rows could represents the area of each rectangle.
    #  Still TLE.
    def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
        rows, cols, ans = len(matrix), len(matrix[0]), -math.inf
        for l in range(cols):
            preSum = [0] * rows
            for r in range(l, cols):
                for p in range(rows):
                    #  preSum = the rectangle from l ro r of each row (横向长方形).
                    preSum[p] += matrix[p][r]
                #  再纵向比较
                for i in range(rows):
                    area = 0
                    for j in range(i, rows):
                        area += preSum[j]
                        if area == k:
                            return k
                        elif area < k:
                            ans = max(ans, area)
        return ans

    """
    Matrix Chain Multiplication (Hard)
    https://practice.geeksforgeeks.org/problems/matrix-chain-multiplication0303/1#
    
    Example 01:
    Input: nums = [40, 20, 30, 10, 30]
    Output: 26000
    Explaination: There are 4 matrices of dimension 40x20, 20x30, 30x10, 10x30. Say the matrices are 
    named as A, B, C, D. Out of all possible combinations, the most efficient way is (A*(B*C))*D. 
    The number of operations are 20*30*10 + 40*20*10 + 40*10*30 = 26000.
    
    Example 02:
    Input: nums = [10, 30, 5, 60
    Output: 4500
    Explaination: The matrices have dimensions 10*30, 30*5, 5*60. Say the matrices are A, B 
    and C. Out of all possible combinations,the most efficient way is (A*B)*C. The 
    number of multiplications are 10*30*5 + 10*5*60 = 4500.
    """

    # #  The recursive dp solution using a memo.
    # def matrixMultiplication(self, nums):
    #     @cache
    #     def _matrixMultiplication(i, j):
    #         #  Base case
    #         if i == j: return 0
    #         ans = math.inf
    #         for k in range(i, j):
    #             ans = min(ans,
    #                       _matrixMultiplication(i, k) + _matrixMultiplication(k + 1, j) + nums[i - 1] * nums[k] * nums[
    #                           j])
    #         return ans
    #     return _matrixMultiplication(1, len(nums) - 1)

    #  The iterative dp solution.
    def matrixMultiplication(self, nums):
        n = len(nums)
        dp = [[math.inf] * n for _ in range(n)]
        for i in range(1, n):
            dp[i][i] = 0
        for l in range(1, n - 1):
            for i in range(1, n - l):
                j = i + l
                for k in range(i, j):
                    dp[i][j] = min(dp[i][j], dp[i][k] + dp[k + 1][j] + nums[i - 1] * nums[k] * nums[j])
        return dp[1][n - 1]

    """
    312. Burst Balloons (Hard)
    https://leetcode.com/problems/burst-balloons/
    """

    # #  The recursive dp solution with a memo.
    # #  (i,j) = the max coins from index i to index j.
    # #  Subproblems: (i,j) = (i, k) + (k + 1, j) + process.
    # #  The trick is, k means the balloons k should be the last balloon to be burst.
    # def maxCoins(self, nums: List[int]) -> int:
    #     @cache
    #     def _maxCoins(i, j):
    #         if i > j: return 0
    #         ans = 0
    #         for k in range(i, j + 1):
    #             ans = max(ans, _maxCoins(i, k - 1) + _maxCoins(k + 1, j) + nums[i - 1] * nums[k] * nums[j + 1])
    #         return ans
    #
    #     nums = [1] + nums + [1]
    #     return _maxCoins(1, len(nums) - 2)

    #  The iterative dp solution.
    def maxCoins(self, nums: List[int]) -> int:
        nums = [1] + nums + [1]
        n = len(nums)
        dp = [[0] * n for _ in range(n)]
        for l in range(1, n - 1):
            for i in range(1, n - l):
                j = i + l
                for k in range(i, j):
                    dp[i][j] = max(dp[i][j], dp[i][k] + dp[k + 1][j] + nums[i - 1] * nums[k] * nums[j])
        return dp[1][n - 1]

    """
    279. Perfect Squares (Medium)
    https://leetcode.com/problems/perfect-squares/
    
    Example 01
    Input: n = 12
    Output: 3
    Explanation: 12 = 4 + 4 + 4.
    
    Example 02
    Input: n = 13
    Output: 2
    Explanation: 13 = 4 + 9.
    """

    #  The straightforward recursive solution with a memo.
    @cache
    def numSquares(self, n: int) -> int:
        #  Base case
        if n < 2: return n
        ans = math.inf
        for i in range(1, int(n ** 0.5) + 1):
            ans = min(ans, 1 + self.numSquares(n - i * i))
        return ans

    # #  The iterative solution.
    # def numSquares(self, n: int) -> int:
    #     dp = [0] + [math.inf] * n
    #     for i in range(1, n + 1):
    #         dp[i] = min(dp[i - j * j] for j in range(1, int(i ** 0.5) + 1)) + 1
    #     return dp[-1]

    # #  The BFS solution, version 01.
    # def numSquares(self, n: int) -> int:
    #     #  Base case
    #     if n < 2: return n
    #     squares = [i ** 2 for i in range(1, int(n ** 0.5) + 1)]
    #     queue, level = [n], 0
    #     while queue:
    #         level += 1
    #         temp = []
    #         for _ in range(len(queue)):
    #             node = queue.pop(0)
    #             for square in squares:
    #                 if node == square:
    #                     return level
    #                 if node > square:
    #                     temp += [node - square]
    #         queue = list(set(temp))
    #
    # #  The BFS solution, version 02.
    # def numSquares(self, n: int) -> int:
    #     #  Base case
    #     if n < 2: return n
    #     queue, level = [n], 0
    #     while queue:
    #         level += 1
    #         temp = []
    #         for _ in range(len(queue)):
    #             node = queue.pop(0)
    #             for i in range(1, int(node ** 0.5) + 1):
    #                 square = i ** 2
    #                 if node == square:
    #                     return level
    #                 temp += [node - square]
    #         queue = list(set(temp))

    """
    518. Coin Change 2 ( xMedium )
    https://leetcode.com/problems/coin-change-2/
    """

    # #  The recursive dp solution with a memo.
    # def change(self, amount: int, coins: List[int]) -> int:
    #     #  tuple(coins) because List is unhashable.
    #     @cache
    #     def _change( amount, coins ):
    #         if amount == 0: return 1
    #         if not coins or amount < 0: return 0
    #         return  _change(amount - coins[-1], coins) + _change(amount, coins[: -1])
    #     return _change(amount, tuple(coins))

    #  The iterative dp solution.
    def change(self, amount: int, coins: List[int]) -> int:
        rows, cols = len(coins) + 1, amount + 1
        #  dp[i][j] = # ways to get j amount using coins[0, ..., i]
        dp = [[0] * cols for _ in range(rows)]
        for i in range(rows):
            dp[i][0] = 1
        for i in range(1, rows):
            for j in range(1, cols):
                if j >= coins[i - 1]:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - coins[i - 1]]
                else:
                    dp[i][j] = dp[i - 1][j]
        return dp[-1][-1]

    """
    63.Unique Paths II (Medium)
    https://leetcode.com/problems/unique-paths-ii/
    """

    # #  The recursive dp solution with a cache.
    # def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
    #     m, n = len(obstacleGrid), len(obstacleGrid[0])
    #     @cache
    #     def _uniquePathsWithObstacles( i, j ):
    #         if obstacleGrid[i][j]: return 0
    #         if i < 0 or j < 0: return 0
    #         if i == j == 0: return 1
    #         return _uniquePathsWithObstacles(i - 1, j) + _uniquePathsWithObstacles(i, j - 1)
    #     return _uniquePathsWithObstacles( m - 1, n - 1 )

    #  The iterative dp solution.
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[0] * n for _ in range(m)]
        if not obstacleGrid[0][0]: dp[0][0] = 1
        for j in range(1, n):
            if not obstacleGrid[0][j]:
                dp[0][j] = dp[0][j - 1]
            else:
                dp[0][j] = 0
        for i in range(1, m):
            if not obstacleGrid[i][0]:
                dp[i][0] = dp[i - 1][0]
            else:
                dp[i][0] = 0
        for i in range(1, m):
            for j in range(1, n):
                if not obstacleGrid[i][j]:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
                else:
                    dp[i][j] = 0
        return dp[-1][-1]

    """
    980. Unique Paths III (Hard)
    https://leetcode.com/problems/unique-paths-iii/
    """

    # #  The DFS solution.
    # def uniquePathsIII(self, grid: List[List[int]]) -> int:
    #     def _uniquePathsIIIDFS(i, j, count):
    #         if not (0 <= i < rows and 0 <= j < cols and grid[i][j] >= 0): return
    #         if grid[i][j] == 2:
    #             self.ans += count == 0
    #             return
    #         grid[i][j] = -1
    #         _uniquePathsIIIDFS(i - 1, j, count - 1)
    #         _uniquePathsIIIDFS(i + 1, j, count - 1)
    #         _uniquePathsIIIDFS(i, j - 1, count - 1)
    #         _uniquePathsIIIDFS(i, j + 1, count - 1)
    #         grid[i][j] = 0
    #     rows, cols = len(grid), len(grid[0])
    #     x, y = 0, 0
    #     count, self.ans = 1, 0
    #     for i in range(rows):
    #         for j in range(cols):
    #             if grid[i][j] == 1:
    #                 x, y = i, j
    #             count += grid[i][j] == 0
    #     _uniquePathsIIIDFS( x, y, count )
    #     return self.ans

    #  The Backtracking solution.
    def uniquePathsIII(self, grid: List[List[int]]) -> int:
        def _uniquePathsIII(i, j):
            if not (0 <= i < rows and 0 <= j < cols and grid[i][j] >= 0): return
            if grid[i][j] == 2:
                self.ans += self.count == 0
                return
            for dx, dy in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                grid[i][j] = -1
                self.count -= 1
                _uniquePathsIII(dx, dy)
                grid[i][j] = 0
                self.count += 1

        rows, cols = len(grid), len(grid[0])
        self.count, self.ans = 1, 0
        x, y = 0, 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    x, y = i, j
                self.count += grid[i][j] == 0
        _uniquePathsIII(x, y)
        return self.ans

    """
    212. Word Search II (Hard)
    https://leetcode.com/problems/word-search-ii/
    """

    #  The dfs solution using a trie.
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        def dfs(i, j, currWord, currTrie):
            #  Base case.
            if "#" in currTrie:
                ans.add(currWord)
            if not (0 <= i < m and 0 <= j < n and board[i][j] != "@" and board[i][j] in currTrie): return
            currChar = board[i][j]
            board[i][j] = "@"
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                dfs(i + dy, j + dx, currWord + currChar, currTrie[currChar])
            board[i][j] = currChar

        m, n = len(board), len(board[0])
        ans, currWord = set(), ""
        #  Initialize the trie.
        trie = {}
        for word in words:
            root = trie
            for char in word:
                root[char] = root.get(char, {})
                root = root[char]
            root["#"] = "#"

        for i in range(m):
            for j in range(n):
                if board[i][j] in trie.keys():
                    dfs(i, j, currWord, trie)
        return list(ans)

    """
    547. Number of Provinces (Medium)
    https://leetcode.com/problems/number-of-provinces/
    """

    # #  The dfs solution.
    # def findCircleNum(self, isConnected: List[List[int]]) -> int:
    #     def dfs(i, j):
    #         if not (0 <= i < rows and 0 <= j < cols and isConnected[i][j] == 1):
    #             return
    #         isConnected[i][j] = 0
    #         for y in range(cols):
    #             dfs(i, y)
    #         for x in range(rows):
    #             dfs(x, j)
    #     rows, cols = len(isConnected), len(isConnected[0])
    #     ans = 0
    #     for i in range(rows):
    #         for j in range(cols):
    #             if isConnected[i][j]:
    #                 ans += 1
    #                 dfs(i, j)
    #     return ans

    #  The disjoint set solution.
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        def _union(i, j):
            pi = _parent(i)
            pj = _parent(j)
            p[pj] = pi

        def _parent(i):
            #  The root in a disjoint set should be p[i] = i
            root = i
            while p[root] != root:
                root = p[root]
            while p[i] != i:
                temp = p[i]
                p[i] = root
                i = temp
            return root

        m, n = len(isConnected), len(isConnected[0])
        p = [i for i in range(n)]
        for i in range(m):
            for j in range(n):
                if isConnected[i][j]:
                    _union(i, j)
        #  IMPORTANT!: _parent(i) not p[i]
        return len(set([_parent(i) for i in range(n)]))

    """
    130. Surrounded Regions ( Medium )
    https://leetcode.com/problems/surrounded-regions/
    """

    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """

        def _dfs(i, j):
            #  Base case
            if not (0 <= i < m and 0 <= j < n) or board[i][j] != "O": return
            board[i][j] = "T"
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                _dfs(i + dy, j + dx)

        m, n = len(board), len(board[0])
        for i in (0, m - 1):
            for j in range(n):
                _dfs(i, j)
        for i in range(m):
            for j in (0, n - 1):
                _dfs(i, j)
        for i in range(m):
            for j in range(n):
                if board[i][j] == "T":
                    board[i][j] = "O"
                elif board[i][j] == "O":
                    board[i][j] = "X"

    """
    36. Valid Sudoku ( Medium )
    https://leetcode.com/problems/valid-sudoku/description/
    """

    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rows = [set(range(1, 10)) for _ in range(9)]
        cols = [set(range(1, 10)) for _ in range(9)]
        blocks = [set(range(1, 10)) for _ in range(9)]
        for i in range(9):
            for j in range(9):
                if board[i][j] != ".":
                    pivot = int(board[i][j])
                    k = (i // 3) * 3 + j // 3
                    if pivot not in rows[i]:
                        return False
                    if pivot not in cols[j]:
                        return False
                    if pivot not in blocks[k]:
                        return False
                    rows[i].remove(pivot)
                    cols[j].remove(pivot)
                    blocks[k].remove(pivot)
        return True

    """
    37. Sudoku Solver (Hard)
    https://leetcode.com/problems/sudoku-solver/#/description
    """

    #  The efficient recursive solution.
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """

        def _solveSudoku(iter=0):
            #  Base case
            if iter == len(remain):
                return True
            i, j = remain[iter]
            k = (i // 3) * 3 + j // 3
            for val in rows[i] & cols[j] & blocks[k]:
                rows[i].remove(val)
                cols[j].remove(val)
                blocks[k].remove(val)
                board[i][j] = str(val)
                if _solveSudoku(iter + 1):
                    return True
                rows[i].add(val)
                cols[j].add(val)
                blocks[k].add(val)

        rows = [set(range(1, 10)) for _ in range(9)]
        cols = [set(range(1, 10)) for _ in range(9)]
        blocks = [set(range(1, 10)) for _ in range(9)]
        remain = []
        for i in range(9):
            for j in range(9):
                if board[i][j] != ".":
                    val, k = int(board[i][j]), (i // 3) * 3 + j // 3
                    rows[i].remove(val)
                    cols[j].remove(val)
                    blocks[k].remove(val)
                else:
                    remain.append((i, j))
        _solveSudoku()

    # #  The return ans version.
    # def solveSudoku(self, board: List[List[str]]) -> None:
    #     """
    #     Do not return anything, modify board in-place instead.
    #     """
    #     def _solveSudoku( iter = 0 ):
    #         #  Base case
    #         if iter == len(remain):
    #             #  IMPORTANT: 对于多维数组来说，要用deepcopy才能复制数组。
    #             ans.append(copy.deepcopy( board ))
    #             return
    #         i, j = remain[iter]
    #         k = ( i // 3 ) * 3 + j // 3
    #         for val in rows[i] & cols[j] & blocks[k]:
    #             rows[i].remove( val )
    #             cols[j].remove( val )
    #             blocks[k].remove( val )
    #             board[i][j] = str(val)
    #             _solveSudoku( iter + 1 )
    #             rows[i].add( val )
    #             cols[j].add( val )
    #             blocks[k].add( val )
    #
    #     rows = [set(range(1, 10)) for _ in range(9)]
    #     cols = [set(range(1, 10)) for _ in range(9)]
    #     blocks = [set(range(1, 10)) for _ in range(9)]
    #     remain, ans = [], []
    #     for i in range(9):
    #         for j in range(9):
    #             if board[i][j] != ".":
    #                 val, k = int(board[i][j]), ( i // 3 ) * 3 + j // 3
    #                 rows[i].remove(val)
    #                 cols[j].remove(val)
    #                 blocks[k].remove(val)
    #             else:
    #                 remain.append((i, j))
    #     _solveSudoku()
    #     return ans

    """
    Leetcode 1091 (Medium)
    https://leetcode.com/problems/shortest-path-in-binary-matrix/
    """

    #  The DP solution can't work.
    #  Because https://leetcode.com/problems/shortest-path-in-binary-matrix/discuss/667137/Why-does-DP-not-work.
    #  The BFS solution.
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        if grid[0][0] or grid[-1][-1]: return -1
        queue, level, n = [(0, 0)], 1, len(grid)
        while queue:
            nextQueue = []
            for i, j in queue:
                if i == n - 1 and j == n - 1: return level
                for x, y in [[i - 1, j - 1], [i - 1, j], [i - 1, j + 1], [i, j - 1], [i, j + 1], [i + 1, j - 1],
                             [i + 1, j], [i + 1, j + 1]]:
                    if not (0 <= x < n and 0 <= y < n and grid[x][y] == 0): continue
                    nextQueue.append((x, y))
                    grid[x][y] = "#"
            level += 1
            queue = nextQueue
        return -1

    """
    773. Sliding Puzzle (Hard)
    https://leetcode.com/problems/sliding-puzzle/
    """

    #  The BFS solution
    def slidingPuzzle(self, board: List[List[int]]) -> int:
        moves = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4], 4: [1, 3, 5], 5: [2, 4]}
        board = board[0] + board[1]
        s = "".join(str(c) for c in board)
        visited, queue, level = set([s]), [(board, board.index(0))], 0
        while queue:
            nextQueue = []
            for seq, index in queue:
                if seq == [1, 2, 3, 4, 5, 0]:
                    return level
                for move in moves[index]:
                    newSeq = seq[:]
                    newSeq[index], newSeq[move] = newSeq[move], newSeq[index]
                    if "".join(str(c) for c in newSeq) not in visited:
                        nextQueue.append((newSeq, move))
                visited.add("".join(str(c) for c in seq))
            level += 1
            queue = nextQueue
        return - 1

    """
    191. Number of 1 Bits (Easy)
    https://leetcode.com/problems/number-of-1-bits/
    """

    #  The bit-operation solution.
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n != 0:
            count += 1
            n = n & n - 1
        return count

    # #  The library solution.
    # def hammingWeight(self, n: int) -> int:
    #     return bin(n).count("1")

    # #  The straightforward solution.
    # def hammingWeight(self, n: int) -> int:
    #     count = 0
    #     while n != 0:
    #         if n % 2:
    #             count += 1
    #         n = n // 2
    #     return count

    """
    231. Power of Two (Easy)
    https://leetcode.com/problems/power-of-two/
    """

    def isPowerOfTwo(self, n: int) -> bool:
        return n != 0 and n & (n - 1) == 0

    """
    190. Reverse Bits (Easy)
    https://leetcode.com/problems/reverse-bits/
    """

    def reverseBits(self, n: int) -> int:
        ans = 0
        for _ in range(32):
            #  ans左移一位，给最右侧腾位置；
            #  n & 1 => n的最后一位（0 或 1）；
            #  综上，把n的最后一位插到ans的右侧。
            ans = (ans << 1) + (n & 1)
            #  n右移1位 = 把n最右侧的1位删除。
            n = n >> 1
        return ans

    """
    338. Counting Bits(Easy)
    https://leetcode.com/problems/counting-bits/description/
    """

    # #  The Brian Kernighan Algorithm solution.
    # def countBits(self, n: int) -> List[int]:
    #     #  ans[i] = i中1的数量。
    #     #  Base case: 0中有0个1，即ans[0] = 0
    #     ans = [0] * (n + 1)
    #     for i in range(1, n + 1):
    #         #  i & i - 1 有两重含义
    #         #  1. 消除i（二进制）的最后1个1。
    #         #  2. 得到1个数，设为k。
    #         #  由1和2可知，k中1的个数比i中1的个数少1，即ans[i] = ans[k] + 1 = ans[i & i - 1] + 1。
    #         ans[i] = ans[i & i - 1] + 1
    #     return ans

    #  The dp solution.
    def countBits(self, n: int) -> List[int]:
        #  ans[i] = the number of 1 in i.
        #  if i is even, ans[i] = ans[i >> 1] = ans[i / 2]
        #  if i is odd, ans[i] = ans[i - 1] + 1
        #  0            0
        #  1            1
        #  2           10
        #  3           11
        #  4          100
        #  5          101
        #  6          110
        #  7          111
        #  8         1000
        #  9         1001
        ans = [0] * (n + 1)
        for i in range(1, n + 1):
            ans[i] = ans[i >> 1] + (i & 1)
        return ans


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

    #  Leetcode 51
    print(S.solveNQueens(1))
    print(S.solveNQueens(2))
    print(S.solveNQueens(3))
    print(S.solveNQueens(4))

    print("------------------------------------")
    #  Leetcode 52
    print(S.totalNQueens(1))
    print(S.totalNQueens(4))
    print("------------------------------------")

    #  Leetcode 102
    print(S.levelOrder02(deserialize('[3,9,20,null,null,15,7]')))
    print(S.levelOrder02(deserialize('[1]')))
    print(S.levelOrder02([]))

    #  Leetcode 515
    print(S.largestValues(deserialize('[1,3,2,5,3,null,9]')))
    print(S.largestValues(deserialize('[1,2,3]')))

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

    #  Leetcode 433
    print(S.minMutation("AACCGGTT", "AACCGGTA", ["AACCGGTA"]))
    print(S.minMutation("AACCGGTT", "AAACGGTA", ["AACCGGTA", "AACCGCTA", "AAACGGTA"]))
    print(S.minMutation("AAAAACCC", "AACCCCCC", ["AAAACCCC", "AAACCCCC", "AACCCCCC"]))

    #  Leetcode 127
    print(S.ladderLength("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]))
    print(S.ladderLength("hit", "cog", ["hot", "dot", "dog", "lot", "log"]))

    #  Leetcode 529
    print(S.updateBoard(
        [["E", "E", "E", "E", "E"], ["E", "E", "M", "E", "E"], ["E", "E", "E", "E", "E"], ["E", "E", "E", "E", "E"]],
        [3, 0]))
    print(S.updateBoard(
        [["B", "1", "E", "1", "B"], ["B", "1", "M", "1", "B"], ["B", "1", "1", "1", "B"], ["B", "B", "B", "B", "B"]],
        [1, 2]))
    print(S.updateBoard([["E", "E", "E", "E", "E", "E", "E", "E"], ["E", "E", "E", "E", "E", "E", "E", "M"],
                         ["E", "E", "M", "E", "E", "E", "E", "E"], ["M", "E", "E", "E", "E", "E", "E", "E"],
                         ["E", "E", "E", "E", "E", "E", "E", "E"], ["E", "E", "E", "E", "E", "E", "E", "E"],
                         ["E", "E", "E", "E", "E", "E", "E", "E"], ["E", "E", "M", "M", "E", "E", "E", "E"]]
                        , [0, 0]))

    #  Leetcode 126
    print(S.findLadders("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]))
    print(S.findLadders("hit", "cog", ["hot", "dot", "dog", "lot", "log"]))
    print(S.findLadders("a", "c", ["a", "b", "c"]))
    print(S.findLadders("red", "tax", ["ted", "tex", "red", "tax", "tad", "den", "rex", "pee"]))

    #  Leetcode 455
    print(S.findContentChildren([1, 2, 3], [1, 1]))
    print(S.findContentChildren([1, 2], [1, 2, 3]))

    #  Leetcode 869
    print(S.lemonadeChange([5, 5, 5, 10, 20]))
    print(S.lemonadeChange([5, 5, 10, 10, 20]))
    print(S.lemonadeChange([5, 5, 10, 20, 5, 5, 5, 5, 5, 5, 5, 5, 5, 10, 5, 5, 20, 5, 20, 5]))
    print(S.lemonadeChange(
        [5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20,
         5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20,
         5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20,
         5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20,
         5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20,
         5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20,
         5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20,
         5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20,
         5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20,
         5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20,
         5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20,
         5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20,
         5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20, 5, 10, 5, 20]))

    #  Leetcode 122
    print(S.maxProfit([7, 1, 5, 3, 6, 4]))
    print(S.maxProfit([1, 2, 3, 4, 5]))
    print(S.maxProfit([7, 6, 4, 3, 1]))

    #  Leetcode 874
    print(S.robotSim([4, -1, 3], []))
    print(S.robotSim([4, -1, 4, -2, 4], [[2, 4]]))
    print(S.robotSim([6, -1, -1, 6], []))

    #  Leetcode 45
    print(S.jump([2, 3, 1, 1, 4]))
    print(S.jump([2, 3, 0, 1, 4]))
    print(S.jump([7, 0, 9, 6, 9, 6, 1, 7, 9, 0, 1, 2, 9, 0, 3]))

    #  Leetcode 69
    print(S.mySqrt(8))
    print(S.mySqrt(5))

    #  Leetcode 367
    print(S.isPerfectSquare(16))
    print(S.isPerfectSquare(14))

    #  Leetcode 33
    print(S.search([1, 3], 3))
    print(S.search([5, 1, 3], 5))
    print(S.search([4, 5, 6, 7, 8, 1, 2, 3], 8))

    #  Leetcode 153
    print(S.findMin([3, 4, 5, 1, 2]))
    print(S.findMin([4, 5, 6, 7, 0, 1, 2]))
    print(S.findMin([11, 13, 15, 17]))
    print(S.findMin([1, 3]))
    print(S.findMin([5, 1, 3]))
    print(S.findMin([4, 5, 6, 7, 8, 1, 2, 3]))

    #  Leetcode 74
    print(S.searchMatrix([[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]], -5))
    print(S.searchMatrix([[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]], 13))

    #  Leetcode 746
    print(S.minCostClimbingStairs([10, 15, 20]))
    print(S.minCostClimbingStairs([1, 100, 1, 1, 1, 100, 1, 1, 100, 1]))

    #  Leetcode 1137
    print(S.tribonacci(4))
    print(S.tribonacci(25))

    #  Leetcode 509
    print(S.fib(2))
    print(S.fib(3))
    print(S.fib(4))

    #  Leetcode 120
    print(S.minimumTotal([[2], [3, 4], [6, 5, 7], [4, 1, 8, 3]]))
    print(S.minimumTotal([[-10]]))

    #  Leetcode 53
    print(S.maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))
    print(S.maxSubArray([1]))
    print(S.maxSubArray([5, 4, -1, 7, 8]))

    #  Leetcode 152
    print(S.maxProduct([2, 3, -2, 4]))
    print(S.maxProduct([-2, 0, -1]))
    print(S.maxProduct([-2, 3, -4]))

    #  Leetcode 198
    print(S.rob([1, 2, 3, 1]))
    print(S.rob([2, 7, 9, 3, 1]))

    #  Leetcode 213
    print(S.rob_circle([2, 3, 2]))
    print(S.rob_circle([1, 2, 3, 1]))
    print(S.rob_circle([1, 2, 3]))

    #  Leetcode 337
    root = deserialize('[3,2,3,null,3,null,1]')
    print(S.rob_tree03(root))
    root = deserialize('[3,4,5,1,3,null,1]')
    print(S.rob_tree03(root))
    root = deserialize('[2,1,3,null,4]')
    print(S.rob_tree03(root))

    #  Leetcode 740
    print(S.deleteAndEarn([2, 2, 3, 3, 3, 4]))
    print(S.deleteAndEarn([3, 4, 2]))

    #  Leetcode 2140
    print(S.mostPoints([[3, 2], [4, 3], [4, 4], [2, 5]]))
    print(S.mostPoints([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]))

    #  Leetcode 121
    print(S.maxProfit121([7, 1, 5, 3, 6, 4]))
    print(S.maxProfit121([7, 6, 4, 3, 1]))

    #  Leetcode 123
    print(S.maxProfit123([3, 3, 5, 0, 0, 3, 1, 4]))
    print(S.maxProfit123([1, 2, 3, 4, 5]))
    print(S.maxProfit123([7, 6, 4, 3, 1]))

    #  Leetcode 188
    print(S.maxProfit188(2, [2, 4, 1]))
    print(S.maxProfit188(2, [3, 2, 6, 5, 0, 3]))

    #  Leetcode 309
    print(S.maxProfit309([1, 2, 3, 0, 2]))
    print(S.maxProfit309([1]))

    #  Leetcode 714
    print(S.maxProfit714([1, 3, 2, 8, 4, 9], 2))
    print(S.maxProfit714([1, 3, 7, 5, 10, 3], 3))

    #  Leetcode 221
    print(S.maximalSquare(
        [["1", "0", "1", "0", "0"], ["1", "0", "1", "1", "1"], ["1", "1", "1", "1", "1"], ["1", "0", "0", "1", "0"]]))
    print(S.maximalSquare([["0", "1"], ["1", "0"]]))
    print(S.maximalSquare([["0"]]))

    #  Leetcode 91
    print(S.numDecodings("12"))
    print(S.numDecodings("226"))
    print(S.numDecodings("06"))
    print(S.numDecodings("10"))

    #  Leetcode 64
    print(S.minPathSum([[1, 3, 1], [1, 5, 1], [4, 2, 1]]))
    print(S.minPathSum([[1, 2, 3], [4, 5, 6]]))

    #  Leetcode 72
    print(S.minDistance("horse", "ros"))
    print(S.minDistance("intention", "execution"))

    #  Leetcode 32
    print(S.longestValidParentheses("(()"))
    print(S.longestValidParentheses(")()())"))
    print(S.longestValidParentheses(""))
    print(S.longestValidParentheses(")("))
    print(S.longestValidParentheses("()(())"))
    print(S.longestValidParentheses("())"))

    #  Leetcode 621
    print(S.leastInterval(["A", "A", "A", "B", "B", "B"], 2))
    print(S.leastInterval(["A", "A", "A", "B", "B", "B"], 0))
    print(S.leastInterval(["A", "A", "A", "A", "A", "A", "B", "C", "D", "E", "F", "G"], 2))
    print(S.leastInterval(["A", "A", "A", "B", "B", "B", "C", "C", "C", "D", "D", "E"], 2))

    #  Leetcode 403
    print(S.canCross([0, 1, 3, 5, 6, 8, 12, 17]))
    print(S.canCross([0, 1, 3, 6, 10, 15, 16, 21]))
    print(S.canCross([0, 1, 2, 3, 4, 8, 9, 11]))
    print(S.canCross([0, 1, 3, 6, 10, 13, 15, 18]))

    #  Leetcode 552
    print(S.checkRecord(2))
    print(S.checkRecord(1))
    print(S.checkRecord(10101))

    #  Leetcode 410
    print(S.splitArray([7, 2, 5, 10, 8], 2))
    print(S.splitArray([1, 2, 3, 4, 5], 2))
    print(S.splitArray([1, 4, 4], 3))

    #  Leetcode 674
    print(S.countSubstrings("abc"))
    print(S.countSubstrings("aaa"))

    #  Leetcode 76
    print(S.minWindow("ADOBECODEBANC", "ABC"))
    print(S.minWindow("ABAACBAB", "ABC"))
    print(S.minWindow("a", "a"))
    print(S.minWindow("a", "aa"))

    #  Leetcode 363
    print(S.maxSumSubmatrix([[1, 0, 1], [0, -2, 3]], 2))
    print(S.maxSumSubmatrix([[2, 2, -1]], 3))
    print(S.maxSumSubmatrix([[2, 2, -1]], 0))

    #  GeeksforGeeks
    print(S.matrixMultiplication([2, 3, 5, 2, 4, 3]))  # ans = 78
    print(S.matrixMultiplication([40, 20, 30, 10, 30]))  # ans = 26000
    print(S.matrixMultiplication([10, 30, 5, 60]))  # ans = 4500

    #  Leetcode 312
    print(S.maxCoins([3, 1, 5, 8]))
    print(S.maxCoins([1, 5]))

    #  Leetcode 279
    print(S.numSquares(12))
    print(S.numSquares(13))

    #  Leetcode 518
    print(S.change(5, [1, 2, 5]))
    print(S.change(3, [2]))
    print(S.change(10, [10]))

    #  Leetcode 63
    print(S.uniquePathsWithObstacles([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))
    print(S.uniquePathsWithObstacles([[0, 1], [0, 0]]))
    print(S.uniquePathsWithObstacles([[0]]))
    print(S.uniquePathsWithObstacles([[1]]))
    print(S.uniquePathsWithObstacles([[1, 0]]))

    #  Leetcode 980
    print(S.uniquePathsIII([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, -1]]))
    print(S.uniquePathsIII([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]]))
    print(S.uniquePathsIII([[0, 1], [2, 0]]))

    #  Leetcode 208
    obj = Trie()
    obj.insert("apple")
    param_2 = obj.search("apple")
    param_3 = obj.search("app")
    param_4 = obj.startsWith("app")
    obj.insert("app")
    param_5 = obj.search("app")

    #  Leetcode 212
    print(S.findWords([["o", "a", "a", "n"], ["e", "t", "a", "e"], ["i", "h", "k", "r"], ["i", "f", "l", "v"]],
                      ["oath", "pea", "eat", "rain"]))
    print(S.findWords([["a", "b"], ["c", "d"]], ["abcb"]))
    print(S.findWords([["o", "a", "b", "n"], ["o", "t", "a", "e"], ["a", "h", "k", "r"], ["a", "f", "l", "v"]],
                      ["oa", "oaa"]))

    #  Leetcode 547
    print(S.findCircleNum([[1, 1, 0], [1, 1, 0], [0, 0, 1]]))
    print(S.findCircleNum([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    print(S.findCircleNum([[1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1]]))
    print(S.findCircleNum(
        [[1, 1, 1, 0, 1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 0, 0, 0, 1, 0], [1, 0, 0, 1, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 0, 0, 1, 0, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]))

    #  Leetcode 130
    board = [["O", "X", "X", "O", "X"], ["X", "O", "O", "X", "O"], ["X", "O", "X", "O", "X"], ["O", "X", "O", "O", "O"],
             ["X", "X", "O", "X", "O"]]
    S.solve(board)
    print(board)

    board = [["X", "X", "X", "X"], ["X", "O", "O", "X"], ["X", "X", "O", "X"], ["X", "O", "X", "X"]]
    S.solve(board)
    print(board)

    board = [["X"]]
    S.solve(board)
    print(board)

    #  Leetcode 36
    print(S.isValidSudoku([["5", "3", ".", ".", "7", ".", ".", ".", "."]
                              , ["6", ".", ".", "1", "9", "5", ".", ".", "."]
                              , [".", "9", "8", ".", ".", ".", ".", "6", "."]
                              , ["8", ".", ".", ".", "6", ".", ".", ".", "3"]
                              , ["4", ".", ".", "8", ".", "3", ".", ".", "1"]
                              , ["7", ".", ".", ".", "2", ".", ".", ".", "6"]
                              , [".", "6", ".", ".", ".", ".", "2", "8", "."]
                              , [".", ".", ".", "4", "1", "9", ".", ".", "5"]
                              , [".", ".", ".", ".", "8", ".", ".", "7", "9"]]))
    print(S.isValidSudoku([["8", "3", ".", ".", "7", ".", ".", ".", "."]
                              , ["6", ".", ".", "1", "9", "5", ".", ".", "."]
                              , [".", "9", "8", ".", ".", ".", ".", "6", "."]
                              , ["8", ".", ".", ".", "6", ".", ".", ".", "3"]
                              , ["4", ".", ".", "8", ".", "3", ".", ".", "1"]
                              , ["7", ".", ".", ".", "2", ".", ".", ".", "6"]
                              , [".", "6", ".", ".", ".", ".", "2", "8", "."]
                              , [".", ".", ".", "4", "1", "9", ".", ".", "5"]
                              , [".", ".", ".", ".", "8", ".", ".", "7", "9"]]))
    print(S.isValidSudoku([[".", ".", "4", ".", ".", ".", "6", "3", "."], [".", ".", ".", ".", ".", ".", ".", ".", "."],
                           ["5", ".", ".", ".", ".", ".", ".", "9", "."], [".", ".", ".", "5", "6", ".", ".", ".", "."],
                           ["4", ".", "3", ".", ".", ".", ".", ".", "1"], [".", ".", ".", "7", ".", ".", ".", ".", "."],
                           [".", ".", ".", "5", ".", ".", ".", ".", "."], [".", ".", ".", ".", ".", ".", ".", ".", "."],
                           [".", ".", ".", ".", ".", ".", ".", ".", "."]]))

    #  Leetcode 37
    board = [["5", "3", ".", ".", "7", ".", ".", ".", "."], ["6", ".", ".", "1", "9", "5", ".", ".", "."],
             [".", "9", "8", ".", ".", ".", ".", "6", "."], ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
             ["4", ".", ".", "8", ".", "3", ".", ".", "1"], ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
             [".", "6", ".", ".", ".", ".", "2", "8", "."], [".", ".", ".", "4", "1", "9", ".", ".", "5"],
             [".", ".", ".", ".", "8", ".", ".", "7", "9"]]
    S.solveSudoku(board)
    print(board)

    # Leetcode 1091
    print(S.shortestPathBinaryMatrix([[0]]))
    print(S.shortestPathBinaryMatrix(
        [[0, 0, 0, 0, 1], [1, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 0]]))
    print(S.shortestPathBinaryMatrix([[0, 1], [1, 0]]))
    print(S.shortestPathBinaryMatrix([[0, 0, 0], [1, 1, 0], [1, 1, 0]]))
    print(S.shortestPathBinaryMatrix([[1, 0, 0], [1, 1, 0], [1, 1, 0]]))
    print(S.shortestPathBinaryMatrix(
        [[0, 1, 1, 0, 0, 0], [0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 1, 0], [1, 1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1, 0]]))
    print(S.shortestPathBinaryMatrix([[0, 0, 0], [1, 1, 0], [1, 1, 1]]))
    print(S.shortestPathBinaryMatrix(
        [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 0],
         [0, 0, 1, 0, 1, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0], [1, 0, 1, 1, 1, 0, 0, 0]]))

    #  Leetcode 773
    print(S.slidingPuzzle([[1, 2, 3], [4, 0, 5]]))
    print(S.slidingPuzzle([[1, 2, 3], [5, 4, 0]]))
    print(S.slidingPuzzle([[4, 1, 2], [5, 0, 3]]))

    #  Leetcode 191
    #  pass

    #  Leetcode 231
    print(S.isPowerOfTwo(1))
    print(S.isPowerOfTwo(16))
    print(S.isPowerOfTwo(3))

    #  Leetcode 190
    #  pass

    #  Leetcode 338
    print(S.countBits(2))
    print(S.countBits(5))

    #  The example of Bloom Filter
    bf = BloomFilter(500000, 7)
    bf.add("dantezhao")
    print(bf.lookup("dantezhao"))
    print(bf.lookup("yyj"))

    print("----------------------------------------")
    #  Leetcode 146
    # Your LRUCache object will be instantiated and called as such:
    # obj = LRUCache(capacity)
    # param_1 = obj.get(key)
    # obj.put(key,value)
    obj = LRUCache(2)
    obj.put(1, 1)
    obj.put(2, 2)
    obj.get(1)
    obj.put(3, 3)
    obj.get(2)
    obj.put(4, 4)
    obj.get(1)
    obj.get(3)
    obj.get(4)


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
