from heapq import heappush, heappop, heapify
from bisect import bisect_left, insort
from functools import cache
from typing import List, Optional
from Solution import deserialize, drawtree
from bitarray import bitarray
from collections import Counter
import math
import collections
import mmh3

class Sort:
    def __init__(self, nums):
        self.nums = nums

    #  遍历n遍数组，每次遍历只要碰到2个元素是逆序就交换它们。
    #  本质上经过每轮交换逆序的操作后就能确定一个最大值，即确定最大值、第二大值、第三大值的过程。
    def bubbleSort(self):
        nums = self.nums
        for i in range(len(nums)):
            for j in range(len(nums) - i - 1):
                if nums[j] > nums[j + 1]:
                    nums[j], nums[j + 1] = nums[j + 1], nums[j]
        self.nums = nums

    def selectionSort(self):
        nums = self.nums
        for i in range(len(nums) - 1):
            smallest = nums[i]
            for j in range(i + 1, len(nums)):
                if nums[j] < smallest:
                    smallest, index = nums[j], j
            nums[i], nums[index] = nums[index], nums[i]
        self.nums = nums

    def insertionSort(self):
        nums = self.nums
        for i in range(1, len(nums)):
            j, pivot = i - 1, nums[i]
            while j >= 0 and nums[j] > pivot:
                nums[j + 1] = nums[j]
                j = j - 1
            nums[j + 1] = pivot
        self.nums = nums

    def quickSort(self):
        def _partition(low, high):
            pivot, i = self.nums[high], low - 1
            for j in range(low, high):
                if self.nums[j] <= pivot:
                    i += 1
                    self.nums[i], self.nums[j] = self.nums[j], self.nums[i]
            self.nums[i + 1], self.nums[high] = self.nums[high], self.nums[i + 1]
            return i + 1

        def _quickSort(low, high):
            if low < high:
                mid = _partition(low, high)
                _quickSort(low, mid - 1)
                _quickSort(mid + 1, high)
        _quickSort(0, len(self.nums) - 1)

    def mergeSort(self):
        def _mergeSort(low, high):
            if low < high:
                mid = (low + high) >> 1
                _mergeSort(low, mid)
                _mergeSort(mid + 1, high)
                _merge(low, mid, high)

        def _merge(low, mid, high):
            l = [self.nums[j] for j in range(low, mid + 1)] + [math.inf]
            r = [self.nums[j] for j in range(mid + 1, high + 1)] + [math.inf]
            i, j = 0, 0
            for k in range(low, high + 1):
                if l[i] <= r[j]:
                    self.nums[k] = l[i]
                    i += 1
                else:
                    self.nums[k] = r[j]
                    j += 1
        _mergeSort(0, len(self.nums) - 1)

    def heapSort(self):
        sortedNums = []
        #  Make your heap
        heapify(self.nums)
        while self.nums:
            sortedNums += [heappop(self.nums)]
        self.nums = sortedNums


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class DLLNode:
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev, self.next = prev, next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        return 'TreeNode({})'.format(self.val)


class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


#  297(hard)
class Codec:
    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        if not root: return "^"
        return str(root.val) + " " + self.serialize(root.left) + " " + self.serialize(root.right)

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """

        def helper(data):
            char = data.popleft()
            if char == "^": return None
            root = TreeNode(char)
            root.left = helper(data)
            root.right = helper(data)
            return root

        return helper(collections.deque(data.split(" ")))


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

#  Bloom Filter满足2个功能：
#  1. 添加字符串；
#  2. 查找字符串；
#  创建一个Bloom Filter需要2个基本参数：
#  1. 能hash的maximum二进制位数；
#  2. 每个输入字符串被hash成几位bits。
#  因此需要2个库：bitarray和mmh3 (bitarray也可以list创建)。
class BloomFilter:
    def __init__(self, size, hash_num):
        self.size = size
        self.hash_num = hash_num
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    def add(self, string):
        for seed in range(self.hash_num):
            index = mmh3.hash(string, seed) % self.size
            self.bit_array[index] = 1

    def lookup(self, string):
        for seed in range(self.hash_num):
            index = mmh3.hash(string, seed) % self.size
            if not self.bit_array[index]: return "Nope"
        return "Probably"


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.od = collections.OrderedDict()

    def get(self, key: int) -> int:
        ans = self.od.get(key, -1)
        if ans != -1:
            #  最左端表示最老的，最右端 = 栈顶 = (last == True) 最新的
            self.od.move_to_end(key)
        return ans

    def put(self, key: int, value: int) -> None:
        #  key存在的case
        if key in self.od:
            self.od.move_to_end(key)
        #  key不存在，还有capacity的case
        elif self.capacity:
            self.capacity -= 1
        #  key不存在，且没有capacity的case
        else:
            self.od.popitem(False)
        self.od[key] = value

#  146(medium)
class DLLNode2:
    # 正常的Double Linked List是不用key的，只有value，prev & next指针即可；
    # 此处因为题目需要每个node要有key和value；
    # 且需要O(1)找到node，所以才需要hashmap的技术处理，即hashmap[key] = node
    def __init__(self, key=0, value=0, prev=None, next=None):
        self.key, self.value = key, value
        self.prev, self.next = prev, next

# You should set the item to the newest(tail) when successfully called the get / put method.
# hashmap[key1], ..., hashmap[key2]
#  |  , ...           , |
# head --------------> node-----------> tail(always empty)
class LRUCache2:
    def __init__(self, capacity: int):
        self.capacity, self.hashmap = capacity, {}
        self.head = self.tail = DLLNode2()

    #  Add a node to the tail(right)
    def addNode(self, key, value):
        #  Put key-value into the tail and update the hashmap.
        self.tail.key, self.tail.value = key, value
        self.hashmap[key] = self.tail
        #  Set up a new tail
        newTail = DLLNode2(0, 0, self.tail, None)
        self.tail.next = newTail
        #  Move tail to the new tail.
        self.tail = newTail

    #  Pop a node
    def popNode(self, key):
        #  Find the node through hashmap
        node = self.hashmap[key]
        #  Delete the node from hashmap
        self.hashmap.pop(key)
        #  Relink the linked list
        prev, next = node.prev, node.next
        if prev: prev.next = next
        if next: next.prev = prev
        #  Remenber to move the head if the node is head.
        if node == self.head:
            self.head = next

    #  Find a node and set it to the tail.
    #  Then return the value of the node.
    def get(self, key: int) -> int:
        if key not in self.hashmap:
            return - 1
        value = self.hashmap[key].value
        self.popNode(key)
        self.addNode(key, value)
        return value

    #  Add/Update a node and set it to the tail.
    #  Pop the head if the cache is full.
    def put(self, key: int, value: int) -> None:
        if key in self.hashmap:
            self.popNode(key)
        elif self.capacity:
            self.capacity -= 1
        else:
            # Pop the head
            self.hashmap.pop(self.head.key)
            self.head = self.head.next
            self.head.prev = None
        self.addNode(key, value)


#  303(easy)
class NumArray:
    def __init__(self, nums: List[int]):
        self.nums = nums
        preSum = [0] + nums[:]
        for i in range(1, len(preSum)):
            preSum[i] = preSum[i - 1] + nums[i - 1]
        self.preSum = preSum

    def sumRange(self, left: int, right: int) -> int:
        return self.preSum[right + 1] - self.preSum[left]


#  304(medium)
class NumMatrix:
    def __init__(self, matrix: List[List[int]]):
        rows, cols = len(matrix) + 1, len(matrix[0]) + 1
        dp = [[0] * cols for _ in range(rows)]
        for i in range(1, rows):
            for j in range(1, cols):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1] - dp[i - 1][j - 1] + matrix[i - 1][j - 1]
        self.matrix = matrix
        self.dp = dp

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        area, dp = 0, self.dp
        return dp[row2 + 1][col2 + 1] - dp[row2 + 1][col1] - dp[row1][col2 + 1] + dp[row1][col1]


#  208(medium)
class Trie:

    def __init__(self):
        self.root = {}

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            node[char] = node.get(char, {})
            node = node[char]
        node["#"] = "#"

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.keys(): return False
            node = node[char]
        return "#" in node.keys()

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.keys(): return False
            node = node[char]
        return True


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

    #  11. Container With Most Water (medium)
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

    #  16(medium)
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        ans, minDelta, n = 0, math.inf, len(nums)
        for i in range(n - 2):
            l, r = i + 1, n - 1
            while l < r:
                total = nums[i] + nums[l] + nums[r]
                if abs(target - total) < minDelta:
                    minDelta = abs(target - total)
                    ans = total
                if target - total > 0:
                    l += 1
                elif target - total < 0:
                    r -= 1
                else:
                    return ans
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

    #  26 (easy)
    def removeDuplicates(self, nums: List[int]) -> int:
        i = 0
        for j in range(len(nums)):
            if j > 0 and nums[j] != nums[j - 1]:
                i += 1
                nums[i] = nums[j]
        return i + 1

    # 88 (easy)
    def merge(self, nums1: List[int], m, nums2: List[int], n):
        while m and n:
            if nums1[m - 1] <= nums2[n - 1]:
                nums1[m + n - 1] = nums2[n - 1]
                n -= 1
            else:
                nums1[m + n - 1] = nums1[m - 1]
                m -= 1
        nums1[:n] = nums2[:n]

    #  24 (medium)
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head

        new_head = head.next
        next = head.next.next

        new_head.next = head
        head.next = self.swapPairs(next)
        return new_head

    #  189 (medium)
    def rotate(self, nums: List[int], k: int) -> None:
        count, n, start = 0, len(nums), 0
        while count != n:
            cur = start
            prev = nums[cur]
            while True:
                next = (cur + k) % n
                nums[next], prev = prev, nums[next]
                cur = next
                count += 1
                if cur == start:
                    break
            start += 1

    #  142 (medium)
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = fast = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if slow == fast:
                break
        else:
            return None

        while head != slow:
            head = head.next
            slow = slow.next
        return head

    #  25 (hard)
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        #  Check
        cur = head
        for _ in range(k):
            #  Base case: less than k nodes.
            if not cur: return head
            cur = cur.next

        # reverse k nodes.
        prev, cur, next, count = None, head, head.next, 0
        while count != k:
            next = cur.next
            cur.next = prev
            prev, cur = cur, next
            count += 1

        #  recursive step
        head.next = self.reverseKGroup(cur, k)
        return prev

    #  20 (easy)
    def isValid(self, s: str) -> bool:
        hashmap = {"(": ")", "[": "]", "{": "}"}
        stack = []
        for char in s:
            if char in hashmap.keys():
                stack.append(char)
            elif not stack or hashmap[stack.pop()] != char:
                return False
            return stack == []

    #  155 (medium)
    class MinStack:
        def __init__(self):
            self.stack = []
            self.minstack = []

        def push(self, val: int) -> None:
            self.stack.append(val)
            if not self.minstack or val < self.minstack[-1]:
                self.minstack.append(val)
            else:
                self.minstack.append(self.minstack[-1])

        def pop(self) -> None:
            self.stack.pop()
            self.minstack.pop()

        def top(self) -> int:
            return self.stack[-1]

        def getMin(self) -> int:
            return self.minstack[-1]

    #  84 (hard)
    def largestRectangleArea(self, heights: List[int]) -> int:
        heights.append(-1)
        largest, stack = -math.inf, [(-1, -1)]
        for i, height in enumerate(heights):
            while height < stack[-1][1]:
                largest = max(largest, stack.pop()[1] * (i - stack[-1][0] - 1))
            stack.append((i, height))
        return largest

    #  85(hard)
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        def helper(nums):
            stack, area = [[-1, -1]], -math.inf
            nums.append(-1)
            for i, num in enumerate(nums):
                while num < stack[-1][1]:
                    area = max(area, stack.pop()[1] * (i - stack[-1][0] - 1))
                stack.append([i, num])
            return area

        rows, cols, ans = len(matrix), len(matrix[0]), -math.inf
        bars = [0] * cols
        for i in range(rows):
            for j in range(cols):
                bars[j] = 0 if matrix[i][j] == "0" else bars[j] + 1
            ans = max(ans, helper(bars[:]))
        return ans

    #  239 (hard)
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        ans, deque = [], collections.deque()
        for i in range(len(nums)):
            #  Step1. Add the num that might be the largest.
            while deque and nums[i] > nums[deque[-1]]:
                deque.popleft()
            deque.append(i)
            #  Step2. Check whether the leftmost num still in the window.
            if deque[0] < i - k + 1:
                deque.popleft()
            #  Step3. Add the max num to the final answer when window begins.
            if i >= k - 1:
                ans.append(nums[deque[0]])
        return ans

    #  641 (medium)
    class MyCircularDeque:
        def __init__(self, k: int):
            self.size, self.capacity = 0, k
            front = DLLNode()
            rear = DLLNode()
            front.next = rear
            rear.prev = front
            self.front, self.rear = front, rear

        def insertFront(self, value: int) -> bool:
            if self.isFull(): return False

            self.front.val = value

            node = DLLNode(0, None, self.front)
            self.front.prev = node
            self.front = node

            self.size += 1
            return True

        def insertLast(self, value: int) -> bool:
            if self.isFull(): return False

            self.rear.val = value
            node = DLLNode(0, self.rear, None)
            self.rear.next = node
            self.rear = node

            self.size += 1
            return True

        def deleteFront(self) -> bool:
            if self.isEmpty(): return False
            self.front = self.front.next
            self.size -= 1
            return True

        def deleteLast(self) -> bool:
            if self.isEmpty(): return False
            self.rear = self.rear.prev
            self.size -= 1
            return True

        def getFront(self) -> int:
            if self.isEmpty(): return -1
            return self.front.next.val

        def getRear(self) -> int:
            if self.isEmpty(): return -1
            return self.rear.prev.val

        def isEmpty(self) -> bool:
            return self.size == 0

        def isFull(self) -> bool:
            return self.capacity == self.size

    #  42 (hard)
    def trap(self, heights: List[int]) -> int:
        leftmax, rightmax, ans = -1, -1, len(heights) * [0]
        for i, h in enumerate(heights):
            if h < leftmax:
                ans[i] = leftmax - h
            leftmax = max(leftmax, h)

        for i in range(len(heights) - 1, -1, -1):
            if heights[i] < rightmax:
                ans[i] = min(ans[i], rightmax - heights[i])
            else:
                ans[i] = 0
            rightmax = max(rightmax, heights[i])
        return sum(ans)

    #  242(easy)
    def isAnagram(self, s: str, t: str) -> bool:
        dict1, dict2 = {}, {}
        for char in s:
            dict1[char] = dict1.get(char, 0) + 1
        for char in t:
            dict2[char] = dict2.get(char, 0) + 1
        return dict1 == dict2

    #  49(medium)
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        ans = {}
        for word in strs:
            key = "".join(sorted(word))
            ans[key] = ans.get(key, []) + [word]
        return list(ans.values())

    #  94(easy)
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        stack, ans = [], []
        while stack or root:
            if root:
                stack.append(root)
                root = root.left
            else:
                root = stack.pop()
                ans.append(root.val)
                root = root.right
        return ans

    #  144(easy)
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        stack, ans = [], []
        while stack or root:
            if root:
                ans.append(root.val)
                stack.append(root)
                root = root.left
            else:
                root = stack.pop().right
        return ans

    #  589(easy)
    def preorder(self, root: 'Node') -> List[int]:
        stack, ans = [root], []
        while stack:
            node = stack.pop()
            if node:
                ans += [node.val]
                for child in reversed(node.children):
                    stack += [child]
        return ans

    #  590(easy)
    def postorder(self, root: 'Node') -> List[int]:
        if not root:
            return []
        ans = []
        for child in root.children:
            ans += self.postorder(child)
        ans += [root.val]
        return ans

    #  429(medium)
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        if not root: return []
        level, ans = [root], []
        while level:
            ans.append([node.val for node in level])
            level = [child for node in level for child in node.children]
        return ans

    #  257(easy)
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        def dfs(root, path):
            if not root.left and not root.right:
                ans.append("->".join(path))
            if root.left:
                dfs(root.left, path + [str(root.left.val)])
            if root.right:
                dfs(root.right, path + [str(root.right.val)])

        if not root: return
        ans = []
        dfs(root, [str(root.val)])
        return ans

    #  783(easy) IMPORTANT!
    def minDiffInBST(self, root: Optional[TreeNode]) -> int:
        def helper(root, low, high):
            if not root: return high - low
            left = helper(root.left, low, root.val)
            right = helper(root.right, root.val, high)
            return min(left, right)

        return helper(root, -math.inf, math.inf)

    #  938(easy)
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        if not root: return 0
        ans = 0 if root.val < low or root.val > high else root.val
        return self.rangeSumBST(root.left, low, high) + ans + self.rangeSumBST(root.right, low, high)

    #  98(easy)
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def isValidBSTHelper(root, low, high):
            if not root: return True
            if low < root.val < high:
                return isValidBSTHelper(root.left, low, root.val) and isValidBSTHelper(root.right, root.val, high)

        return isValidBSTHelper(root, -float("inf"), float("inf"))

    #  530(easy)
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        def helper(root, low, high):
            if not root: return high - low
            left = helper(root.left, low, root.val)
            right = helper(root.right, root.val, high)
            return min(left, right)

        return helper(root, -float("inf"), float("inf"))

    #  105(medium)
    def buildTree105(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder and not inorder: return None
        root = TreeNode(preorder[0])
        pos = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1:pos + 1], inorder[:pos])
        root.right = self.buildTree(preorder[pos + 1:], inorder[pos + 1:])
        return root

    #  106(medium)
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if not inorder: return None
        if len(inorder) == 1: return TreeNode(inorder[0])

        root = TreeNode(postorder[-1])
        pos = inorder.index(postorder[-1])

        root.left = self.buildTree(inorder[:pos], postorder[:pos])
        root.right = self.buildTree(inorder[pos + 1:], postorder[pos:-1])
        return root

    #  230(medium)
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        def helper(root):
            if not root: return
            if self.count <= 0: return
            helper(root.left)
            self.count -= 1
            if self.count == 0:
                self.ans = root.val
                return
            helper(root.right)

        self.count, self.ans = k, float("inf")
        helper(root)
        return self.ans

    #  687(medium)
    def longestUnivaluePath(self, root: Optional[TreeNode]) -> int:
        def helper(root):
            if not root: return 0
            left, right = helper(root.left), helper(root.right)
            left = left + 1 if root.left and root.val == root.left.val else 0
            right = right + 1 if root.rigt and root.val == root.right.val else 0
            self.ans = max(self.ans, left + right)
            return max(left, right)

        self.ans = float("-inf")
        helper(root)
        return self.ans

    #  543(easy)
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        def helper(root):
            if not root: return 0
            left, right = helper(root.left), helper(root.right)
            left = left + 1 if root.left else 0
            right = right + 1 if root.right else 0
            self.ans = max(self.ans, left + right)
            return max(left, right)

        self.ans = 0
        helper(root)
        return self.ans

    #  22(medium)
    @cache
    def generateParenthesis(self, n: int) -> List[str]:
        def helper(seq, left, right):
            if not left and not right:
                self.ans += [seq]
                return
            if left > 0:
                helper(seq + "(", left - 1, right)
            if right > left:
                helper(seq + ")", left, right - 1)

        self.ans, seq = [], ""
        helper(seq, n, n)
        return self.ans

    #  226(easy)
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root: return
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root

    #  104(easy)
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root: return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

    #  111(easy)
    def minDepth(self, root: Optional[TreeNode]) -> int:
        def _minDepth(root):
            if not root: return float("inf")
            if not root.left and not root.right: return 1
            return 1 + min(_minDepth(root.left), _minDepth(root.right))

        return _minDepth(root) if _minDepth(root) != float("inf") else 0

    #  236(medium)
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root or root == p or root == q: return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        else:
            return left or right

    #  77(medium)
    def combine(self, n: int, k: int) -> List[List[int]]:
        def backtracking(n, k):
            if n < k:
                return
            if not k:
                ans.append(pair[:])
                return
            for num in reversed(range(1, n + 1)):
                pair.append(num)
                backtracking(num - 1, k - 1)
                pair.pop()

        pair, ans = [], []
        backtracking(n, k)
        return ans

    #  46(medium)
    def permute(self, nums: List[int]) -> List[List[int]]:
        if not nums: return [[]]
        return [[nums[i]] + item for i in range(len(nums)) for item in self.permute(nums[:i] + nums[i + 1:])]

    #  47(medium)
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        def backtracking(nums):
            if not nums:
                ans.append(pair[:])
                return
            checked = {}
            for i in range(len(nums)):
                if nums[i] in checked:
                    continue
                checked[nums[i]] = nums[i]
                pair.append(nums[i])
                backtracking(nums[:i] + nums[i + 1:])
                pair.pop()

        ans, pair = [], []
        backtracking(nums)
        return ans

    #  78(medium)
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def backtracking(nums):
            ans.append(seq[:])
            if not nums: return
            for i in range(len(nums)):
                seq.append(nums[i])
                backtracking(nums[i + 1:])
                seq.pop()

        ans, seq = [], []
        backtracking(nums)
        return ans

    #  90(medium)
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        def backtracking(nums):
            ans.append(seq[:])
            if not nums: return
            for i in range(len(nums)):
                if i > 0 and nums[i] == nums[i - 1]:
                    continue
                seq.append(nums[i])
                backtracking(nums[i + 1:])
                seq.pop()

        ans, seq = [], []
        nums.sort()
        backtracking(nums)
        return ans

    #  50(medium)
    def myPow(self, x: float, n: int) -> float:
        if n == 0: return 1
        if n < 0: return 1 / self.myPow(x, -n)
        if n % 2:
            return x * self.myPow(x, n - 1)
        else:
            return self.myPow(x * x, n // 2)

    #  169(easy)
    def majorityElement(self, nums: List[int]) -> int:
        flag = 0
        for i, num in enumerate(nums):
            if not flag:
                pivot, flag = num, 1
            elif num == pivot:
                flag += 1
            else:
                flag -= 1
        return pivot

    #  17(medium)
    def letterCombinations(self, digits: str) -> List[str]:
        hashmap = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
        ans = [""]
        for num in digits:
            ans = [item + char for item in ans for char in hashmap[num]]
        return ans if digits else digits

    #  51(hard)
    def solveNQueens(self, n: int) -> List[List[str]]:
        def helper(i):
            if i == n:
                ans.append(placed[:])
                return
            for j in range(n):
                if j in cols or j - i in slash or j + i in backslach:
                    continue
                placed.append(j)
                cols.append(j)
                slash.append(j - i)
                backslach.append(j + i)

                helper(i + 1)

                placed.pop()
                cols.pop()
                slash.pop()
                backslach.pop()

        ans, placed, cols, slash, backslach = [], [], [], [], []
        helper(0)
        return [["." * j + "Q" + "." * (n - j - 1) for j in seq] for seq in ans]

    #  102(median)
    def levelOrder102(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root: return root
        queue, ans = collections.deque([root]), []
        while queue:
            ans += [[node.val for node in queue]]
            for i in range(len(queue)):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return ans

    #  515(median)
    def largestValues(self, root: Optional[TreeNode]) -> List[int]:
        if not root: return root
        deque, ans = collections.deque([root]), []
        while deque:
            ans.append(max([node.val for node in deque]))
            for i in range(len(deque)):
                node = deque.popleft()
                if node.left:
                    deque.append(node.left)
                if node.right:
                    deque.append(node.right)
        return ans

    #  200(medium)
    def numIslands(self, grid: List[List[str]]) -> int:
        def helper(i, j):
            if i < 0 or i == rows or j < 0 or j == cols or grid[i][j] == "0":
                return 0
            grid[i][j] = "0"
            helper(i - 1, j)
            helper(i + 1, j)
            helper(i, j - 1)
            helper(i, j + 1)
            return 1

        count, visited = 0, set()
        rows, cols = len(grid), len(grid[0])
        for i in range(rows):
            for j in range(cols):
                count += helper(i, j)
        return count

    #  529(medium)
    def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
        def minesAround(i, j):
            count = 0
            for k in range(len(dx)):
                x, y = i + dx[k], j + dy[k]
                if -1 < x < rows and -1 < y < cols and board[x][y] == "M":
                    count += 1
            return count

        def _updateBoard(i, j):
            if board[i][j] == "M":
                board[i][j] = "X"
                return
            if board[i][j] != "E":
                return
            count = minesAround(i, j)
            if count:
                board[i][j] = str(count)
            else:
                board[i][j] = "B"
                for k in range(len(dx)):
                    if -1 < i + dx[k] < rows and -1 < j + dy[k] < cols:
                        _updateBoard(i + dx[k], j + dy[k])

        dx = [-1, 1, 0, 0, -1, -1, 1, 1]
        dy = [0, 0, -1, 1, -1, 1, -1, 1]
        rows, cols = len(board), len(board[0])
        _updateBoard(click[0], click[1])
        return board

    #  433(medium)
    def minMutation(self, start: str, end: str, bank: List[str]) -> int:
        queue, bank = [(start, 0)], set(bank)
        while queue:
            for seq, level in queue:
                for newseq in [seq[:i] + char + seq[i + 1:] for i in range(len(seq)) for char in ["A", "C", "G", "T"]]:
                    if newseq in bank:
                        if newseq == end:
                            return level + 1
                        queue.append((newseq, level + 1))
                        bank.remove(newseq)
        return -1

    # 127(hard)
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        queue, visited, wordList, level = [beginWord], set(), set(wordList), 1
        while queue:
            nextQueue = []
            for word in queue:
                for i in range(len(word)):
                    for char in "abcdefghijklmnopqrstuvwxyz":
                        newWord = word[:i] + char + word[i + 1:]
                        if newWord in wordList and newWord not in visited:
                            if newWord == endWord: return level + 1
                            visited.add(newWord)
                            nextQueue.append(newWord)
            level += 1
            queue = nextQueue
        return 0

    #  126(hard)
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        def _showPath(key):
            if key == beginWord:
                ans.append(route[::-1])
                return
            if not path.get(key): return
            for value in path[key]:
                route.append(value)
                _showPath(value)
                route.pop()

        queue, wordList, path = [beginWord], set(wordList), {}
        if beginWord in wordList: wordList.remove(beginWord)
        while queue:
            localVisited = set()
            for word in queue:
                for i in range(len(word)):
                    for char in "abcdefghijklmnopqrstuvwxyz":
                        newWord = word[:i] + char + word[i + 1:]
                        if newWord not in wordList: continue
                        path[newWord] = path.get(newWord, []) + [word]
                        localVisited.add(newWord)
            for word in localVisited:
                wordList.remove(word)
            queue = list(localVisited)

        ans, route = [], [endWord]
        _showPath(endWord)
        return ans

    #  860(easy)
    def lemonadeChange(self, bills: List[int]) -> bool:
        five, ten = 0, 0
        for paid in bills:
            if paid == 5:
                five += 1
            elif paid == 10:
                five, ten = five - 1, ten + 1
            elif ten > 0:
                five, ten = five - 1, ten - 1
            else:
                five -= 3
            if five < 0: return False
        return True

    #  322(medium)
    def coinChange(self, coins: List[int], amount: int) -> int:
        @cache
        def helper(amount):
            if amount < 0: return float("inf")
            if amount == 0: return 0
            ans = float("inf")
            for coin in coins:
                ans = min(ans, helper(amount - coin))
            return ans + 1

        ans = helper(amount)
        return ans if ans != float("inf") else -1

    #  55(medium)
    def canJump(self, nums: List[int]) -> bool:
        reach = nums[0]
        for i, num in enumerate(nums):
            if i <= reach:
                reach = max(reach, i + nums[i])
            if reach >= len(nums) - 1:
                return True
        return False

    #  45(medium)
    def jump(self, nums: List[int]) -> int:
        reach, nextReach, steps = 0, 0, 0
        for i, num in enumerate(nums):
            if reach >= len(nums) - 1: return steps
            nextReach = max(nextReach, i + num)
            if i == reach:
                reach = nextReach
                steps += 1

    #  455(easy)
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        i, j, m, n = 0, 0, len(g), len(s)
        g.sort()
        s.sort()
        while i < m and j < n:
            if g[i] <= s[j]: i += 1
            j += 1
        return i

    #  122(medium)
    def maxProfit(self, prices: List[int]) -> int:
        ans = 0
        for i in range(1, len(prices)):
            delta = prices[i] - prices[i - 1]
            if delta > 0: ans += delta
        return ans

    #  874(medium)
    def robotSim(self, commands: List[int], obstacles: List[List[int]]) -> int:
        direction = {"north": [0, 1, "west", "east"],
                     "south": [0, -1, "east", "west"],
                     "west": [-1, 0, "south", "north"],
                     "east": [1, 0, "north", "south"]}
        cur, x, y, obstacles, ans = "north", 0, 0, set(map(tuple, obstacles)), 0
        for command in commands:
            #  Turn right
            if command == -1:
                cur = direction[cur][3]
            #  Turn left
            elif command == -2:
                cur = direction[cur][2]
            else:
                for step in range(command):
                    dx, dy = x + direction[cur][0], y + direction[cur][1]
                    if (dx, dy) in obstacles:
                        break
                    x, y = dx, dy
                    ans = max(ans, x * x + y * y)
        return ans

    #  69(easy)
    def mySqrt(self, x: int) -> int:
        low, high = 1, x
        while low <= high:
            mid = (low + high) // 2
            pivot = mid * mid
            if pivot == x:
                return mid
            elif pivot < x:
                low = mid + 1
            else:
                high = mid - 1
        return high

    #  33(medium)
    def search(self, nums: List[int], target: int) -> int:
        low, high = 0, len(nums) - 1
        while low <= high:
            mid = (low + high) // 2
            if nums[mid] == target:
                return mid
            elif (nums[mid] < nums[high] and nums[mid] < target <= nums[high]) or (
                    nums[low] <= nums[mid] and (target > nums[mid] or target < nums[low])):
                low = mid + 1
            else:
                high = mid - 1
        return -1

    #  367(easy)
    def isPerfectSquare(self, num: int) -> bool:
        low, high = 1, num
        while low <= high:
            mid = (low + high) // 2
            pivot = mid * mid
            if pivot == num:
                return True
            elif pivot < num:
                low = mid + 1
            else:
                high = mid - 1
        return False

    #  74(medium)
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        nums = []
        for row in matrix:
            nums += row
        low, high = 0, len(nums) - 1
        while low <= high:
            mid = (low + high) // 2
            if nums[mid] == target:
                return True
            elif target < nums[mid]:
                high = mid - 1
            else:
                low = mid + 1
        return False

    #  153(medium)
    def findMin(self, nums: List[int]) -> int:
        low, high = 0, len(nums) - 1
        while low < high:
            mid = (low + high) // 2
            if nums[low] <= nums[mid] < nums[high]:
                break
            elif nums[low] <= nums[mid] and nums[mid] > nums[high]:
                low = mid + 1
            else:
                high = mid
        return nums[low]

    #  62(medium)
    def uniquePaths(self, m: int, n: int) -> int:
        prev, cur = [1] * n, [0] * n
        for i in range(m - 1):
            for j in range(n):
                if j > 0:
                    cur[j] = prev[j] + cur[j - 1]
                else:
                    cur[j] = prev[j]
            prev = cur
            cur = [0] * n
        return prev[-1]

    #  63(medium)
    def uniquePathsWithObstacles(self, grid: List[List[int]]) -> int:
        if grid[0][0]: return 0
        m, n = len(grid), len(grid[0])
        dp = [[0 for j in range(n)] for i in range(m)]
        dp[0][0] = 1
        for i in range(1, m):
            if grid[i][0]:
                dp[i][0] = 0
            else:
                dp[i][0] = dp[i - 1][0]
        for j in range(1, n):
            if grid[0][j]:
                dp[0][j] = 0
            else:
                dp[0][j] = dp[0][j - 1]
        for i in range(1, m):
            for j in range(1, n):
                if grid[i][j]:
                    dp[i][j] = 0
                else:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

    #  1143(medium)
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        prev = [0 if text1[0] != text2[j] else 1 for j in range(n)]
        for j in range(1, n):
            prev[j] = max(prev[j], prev[j - 1])

        for i in range(1, m):
            cur = [0] * n
            for j in range(n):
                if j == 0:
                    cur[j] = max(prev[j], 1) if text1[i] == text2[j] else prev[j]
                else:
                    if text1[i] == text2[j]:
                        cur[j] = 1 + prev[j - 1]
                    else:
                        cur[j] = max(cur[j - 1], prev[j])
            prev = cur
        return prev[-1]

    #  120(medium)
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        for i in reversed(range(len(triangle) - 1)):
            for j in range(len(triangle[i])):
                triangle[i][j] += min(triangle[i + 1][j], triangle[i + 1][j + 1])
        return triangle[0][0]

    #  53(medium)
    def maxSubArray(self, nums: List[int]) -> int:
        ans = nums[0]
        for i in range(1, len(nums)):
            nums[i] = max(nums[i - 1] + nums[i], nums[i])
            ans = max(ans, nums[i])
        return ans

    #  152(medium)
    def maxProduct(self, nums: List[int]) -> int:
        n, ans = len(nums), nums[0]
        smallest, largest = [nums[0]] * n, [nums[0]] * n
        for i in range(1, n):
            smallest[i] = min(nums[i], nums[i] * smallest[i - 1], nums[i] * largest[i - 1])
            largest[i] = max(nums[i], nums[i] * smallest[i - 1], nums[i] * largest[i - 1])
            ans = max(ans, largest[i])
        return ans

    #  121(medium)
    def maxProfit121(self, prices: List[int]) -> int:
        buy, sell = -prices[0], -float("inf")
        for price in prices:
            buy = max(buy, -price)
            sell = max(sell, buy + price)
        return sell

    #  123(hard)
    def maxProfit123(self, prices: List[int]) -> int:
        buy1, sell1, buy2, sell2 = -prices[0], 0, -float("inf"), -float("inf")
        for price in prices:
            buy1 = max(buy1, -price)
            sell1 = max(sell1, buy1 + price)
            buy2 = max(buy2, sell1 - price)
            sell2 = max(sell2, buy2 + price)
        return sell2

    #  188(hard)
    def maxProfit188(self, k: int, prices: List[int]) -> int:
        dp = [0] + [-float("inf") for _ in range(2 * k)]
        for price in prices:
            for i in range(1, len(dp), 2):
                dp[i] = max(dp[i], dp[i - 1] - price)
                dp[i + 1] = max(dp[i + 1], dp[i] + price)
        return dp[-1] if prices else 0

    #  714(medium)
    def maxProfit714(self, prices: List[int], fee: int) -> int:
        buy, sell = -prices[0], 0
        for price in prices:
            buy = max(buy, sell - price)
            sell = max(sell, buy + price - fee)
        return sell

    #  309(medium)
    def maxProfit309(self, prices: List[int]) -> int:
        buy, sell, cooldown = -prices[0], 0, [0, 0]
        for price in prices:
            buy = max(buy, cooldown[1] - price)
            sell = max(sell, buy + price)
            cooldown = [sell, cooldown[0]]
        return sell

    #  198(medium)
    def rob(self, nums: List[int]) -> int:
        nums = [0] + nums
        dp = nums[:]
        for i in range(2, len(nums)):
            dp[i] = max(dp[i - 1], nums[i] + dp[i - 2])
        return dp[-1]

    # #  Using the state machine from buy-and-sell problems
    # def rob(self, nums: List[int]) -> int:
    #     rob, cooldown = 0, [0, 0]
    #     for num in nums:
    #         rob = max(rob, num + cooldown[1])
    #         cooldown = [rob, cooldown[0]]
    #     return rob

    #  213(medium)
    def rob213(self, nums: List[int]) -> int:
        def helper(nums):
            if len(nums) == 1: return nums[0]
            f0, f1 = nums[0], max(nums[0], nums[1])
            for i in range(2, len(nums)):
                f2 = max(f1, f0 + nums[i])
                f0, f1 = f1, f2
            return f1 if len(nums) == 2 else f2

        return nums[0] if len(nums) == 1 else max(helper(nums[1:]), helper(nums[:-1]))

    #  279(medium)
    def numSquares(self, n: int) -> int:
        dp = [0] + [math.inf] * n
        for i in range(1, n + 1):
            for j in range(1, int(i ** 0.5) + 1):
                dp[i] = min(dp[i], dp[i - j * j] + 1)
        return dp[-1]

    #  72(hard)
    def minDistance(self, word1: str, word2: str) -> int:
        @cache
        def helper(m, n):
            if not m or not n: return m or n
            if word1[m - 1] == word2[n - 1]:
                return helper(m - 1, n - 1)
            else:
                return 1 + min(helper(m - 1, n - 1), helper(m - 1, n), helper(m, n - 1))

        return helper(len(word1), len(word2))

    #  980(hard)
    def uniquePathsIII(self, grid: List[List[int]]) -> int:
        def dfsHelper(i, j, untravelled):
            if not (0 <= i < rows and 0 <= j < cols and grid[i][j] >= 0): return
            if grid[i][j] == 2:
                self.ans += untravelled == 0
                return
            for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                grid[i][j] = -1
                dfsHelper(i + dx, j + dy, untravelled - 1)
                grid[i][j] = 0

        rows, cols, untravelled = len(grid), len(grid[0]), 1
        self.ans = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    x, y = i, j
                untravelled += grid[i][j] == 0
        dfsHelper(x, y, untravelled)
        return self.ans

    #  518(medium)
    def change(self, amount: int, coins: List[int]) -> int:
        rows, cols = len(coins) + 1, amount + 1
        dp = [[0] * cols for _ in range(rows)]
        for i in range(rows):
            dp[i][0] = 1
        for i in range(1, rows):
            for j in range(1, cols):
                dp[i][j] = dp[i][j - coins[i - 1]] + dp[i - 1][j] if j >= coins[i - 1] else dp[i - 1][j]
        return dp[-1][-1]

    #  64(medium)
    def minPathSum(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        for j in range(1, cols):
            grid[0][j] = grid[0][j - 1] + grid[0][j]
        for i in range(1, rows):
            grid[i][0] = grid[i - 1][0] + grid[i][0]
        for i in range(1, rows):
            for j in range(1, cols):
                grid[i][j] = min(grid[i - 1][j], grid[i][j - 1]) + grid[i][j]
        return grid[-1][-1]

    #  32(hard)
    def longestValidParentheses(self, s: str) -> int:
        n = len(s)
        dp = [0] * n
        for i in range(1, n):
            if s[i - 1] == "(" and s[i] == ")":
                dp[i] = 2 + dp[i - 2] if i - 2 >= 0 else 2
            elif s[i - 1] == ")" and s[i] == ")":
                j = i - dp[i - 1] - 1
                if j >= 0 and s[j] == "(":
                    dp[i] = dp[j - 1] + dp[i - 1] + 2 if j >= 1 else dp[i - 1] + 2
        return max(dp) if dp else 0

    #  91(medium)
    def numDecodings(self, s: str) -> int:
        dp = [0] * len(s)
        for i in range(len(s)):
            if 1 <= int(s[i]) <= 9:
                dp[i] += dp[i - 1] if i >= 1 else 1
            if i >= 2 and 10 <= int(s[i - 1: i + 1]) <= 26:
                dp[i] += dp[i - 2]
            elif i == 1 and 10 <= int(s[i - 1: i + 1]) <= 26:
                dp[i] += 1
        return dp[-1] if dp else 0

    #  221(medium)
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        rows, cols, ans = len(matrix), len(matrix[0]), 0
        dp = [[0] * cols for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == "1":
                    dp[i][j] = 1
        for i in range(rows):
            for j in range(cols):
                if i > 0 and j > 0 and dp[i][j]:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                ans = max(ans, dp[i][j])
        return ans * ans

    #  560(medium)
    def subarraySum(self, nums: List[int], k: int) -> int:
        hashmap = {}
        hashmap[0], count, preSum = 1, 0, 0
        for i in range(len(nums)):
            preSum += nums[i]
            count += hashmap.get(preSum - k, 0)
            hashmap[preSum] = hashmap.get(preSum, 0) + 1
        return count

    #  363(hard)
    def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
        ans, rows, cols = -math.inf, len(matrix), len(matrix[0])
        for l in range(cols):
            compressed = [0] * rows
            for r in range(l, cols):
                checked = [0]
                preSum = 0
                for i in range(rows):
                    compressed[i] += matrix[i][r]
                    preSum += compressed[i]
                    index = bisect_left(checked, preSum - k)
                    if index < len(checked):
                        if preSum - checked[index] == k:
                            return k
                        else:
                            ans = max(ans, preSum - checked[index])
                    insort(checked, preSum)
        return ans

    # def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
    #     ans, rows, cols = -math.inf, len(matrix), len(matrix[0])
    #     for top in range(rows):
    #         compressed = [0] * cols
    #         for bottom in range(top, rows):
    #             checked = [0]
    #             preSum = 0
    #             for j in range(cols):
    #                 compressed[j] += matrix[bottom][j]
    #                 preSum += compressed[j]
    #                 index = bisect_left(checked, preSum - k)
    #                 if index < len(checked):
    #                     if preSum - checked[index] == k:
    #                         return k
    #                     else:
    #                         ans = max(ans, preSum - checked[index])
    #                 insort(checked, preSum)
    #     return ans

    #  403(hard)
    def canCross(self, stones: List[int]) -> bool:
        @cache
        def dfs(pos, jumps):
            if pos == stones[-1]: return True
            for nextJump in [jumps - 1, jumps, jumps + 1]:
                if nextJump > 0 and pos + nextJump in stonesSet:
                    if dfs(pos + nextJump, nextJump): return True

        stonesSet = set(stones)
        return dfs(0, 0)

    #  621(medium)
    def leastInterval(self, tasks: List[str], n: int) -> int:
        heap, ans = [], 0
        for freq in Counter(tasks).values():
            heappush(heap, -freq)
        while heap:
            count, nextHeap = 0, []
            while count <= n:
                if not heap and not nextHeap: return ans
                if heap:
                    freq = heappop(heap)
                    freq += 1
                    if freq != 0:
                        heappush(nextHeap, freq)
                count += 1
                ans += 1
            for node in nextHeap:
                heappush(heap, node)
        return ans

    #  647(medium)
    def countSubstrings(self, s: str) -> int:
        def palindromicNum(i, j):
            count = 0
            while i >= 0 and j < len(s) and s[i] == s[j]:
                count += 1
                i -= 1
                j += 1
            return count

        ans = 0
        for i in range(len(s)):
            ans += palindromicNum(i, i)
            ans += palindromicNum(i, i + 1)
        return ans

    #  5(medium)
    def longestPalindrome(self, s: str) -> str:
        largest, begin = 1, 0
        for i in range(1, len(s)):
            if i - largest - 1 >= 0 and s[i - largest - 1:i + 1] == s[i - largest - 1:i + 1][::-1]:
                begin = i - largest - 1
                largest += 2

            elif i - largest >= 0 and s[i - largest:i + 1] == s[i - largest:i + 1][::-1]:
                begin = i - largest
                largest += 1

        return s[begin: begin + largest]

    #  516(medium)
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 1
        for j in range(1, n):
            for i in range(n - j):
                if s[i] == s[i + j]:
                    dp[i][i + j] = dp[i + 1][i + j - 1] + 2
                else:
                    dp[i][i + j] = max(dp[i][i + j - 1], dp[i + 1][i + j])
        return dp[0][n - 1]

    #  410(hard)
    def splitArray(self, nums: List[int], m: int) -> int:
        def validSplit(ceil):
            count, subnums = 1, 0
            for num in nums:
                if subnums + num <= ceil:
                    subnums += num
                else:
                    subnums = num
                    count += 1
            return count <= m

        low, high = max(nums), sum(nums)
        while low < high:
            mid = (low + high) // 2
            if validSplit(mid):
                high = mid
            else:
                low = mid + 1
        return low

    #  312(hard)
    def maxCoins(self, nums: List[int]) -> int:
        nums = [1] + nums + [1]
        n = len(nums)
        dp = [[0] * n for _ in range(n)]
        for l in range(n - 2):
            for i in range(1, n - l - 1):
                j = i + l
                for k in range(i, j + 1):
                    dp[i][j] = max(dp[i][j], dp[i][k - 1] + dp[k + 1][j] + nums[i - 1] * nums[k] * nums[j + 1])
        return dp[1][n - 2]

    #  76(hard)
    def minWindow(self, s: str, t: str) -> str:
        ans, count, hashmap = math.inf, len(t), collections.Counter(t)
        i, j, left, right = -1, -1, 0, -1
        while i <= j:
            if not count and j - i < ans:
                ans = j - i
                left, right = i + 1, j

            if j < len(s) and count > 0:
                j += 1
                if j < len(s) and s[j] in hashmap.keys():
                    hashmap[s[j]] -= 1
                    if hashmap[s[j]] >= 0: count -= 1
            else:
                i += 1
                if i < len(s) and s[i] in hashmap.keys():
                    hashmap[s[i]] += 1
                    if hashmap[s[i]] > 0: count += 1

        return s[left: right + 1]

    #  552(hard)
    def checkRecord(self, n: int) -> int:
        #  dp[i][j] means the #records ends at ith postion of n with case j.
        #  dp[i][0] without A, ends with a A            dp[i - 1][1], dp[i - 1][2], dp[i - 1][3]
        #  dp[i][1] without A, ends with a P            dp[i - 1][1], dp[i - 1][2], dp[i - 1][3]
        #  dp[i][2] without A, ends with a L            dp[i - 1][1]
        #  dp[i][3] without A, ends with a LL           dp[i - 1][2]
        #  dp[i][4] with a A already, ends with a P     dp[i - 1][0], dp[i - 1][4], dp[i - 1][5], dp[i - 1][6]
        #  dp[i][5] with a A already, ends with a L     dp[i - 1][0], dp[i - 1][4]
        #  dp[i][6] with a A already, ends with a LL    dp[i - 1][5]
        MOD = 1000000007
        dp = [[0] * 7 for _ in range(n)]
        dp[0][0], dp[0][1], dp[0][2] = 1, 1, 1
        for i in range(1, n):
            dp[i][0] = (dp[i - 1][1] + dp[i - 1][2] + dp[i - 1][3]) % MOD
            dp[i][1] = (dp[i - 1][1] + dp[i - 1][2] + dp[i - 1][3]) % MOD
            dp[i][2] = (dp[i - 1][1])
            dp[i][3] = (dp[i - 1][2])
            dp[i][4] = (dp[i - 1][0] + dp[i - 1][4] + dp[i - 1][5] + dp[i - 1][6]) % MOD
            dp[i][5] = (dp[i - 1][0] + dp[i - 1][4]) % MOD
            dp[i][6] = (dp[i - 1][5])
        return sum(dp[n - 1][j] for j in range(7)) % MOD

    #  79(medium)
    def exist(self, board: List[List[str]], word: str) -> bool:
        def helper(i, j, k):
            if k == n: return True
            if not (0 <= i < rows and 0 <= j < cols and board[i][j] == word[k]):
                return False
            ori = board[i][j]
            board[i][j] = "#"
            for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
                if helper(i + dx, j + dy, k + 1): return True
            board[i][j] = ori
            return False

        rows, cols, n = len(board), len(board[0]), len(word)
        for i in range(rows):
            for j in range(cols):
                if helper(i, j, 0): return True
        return False

    #  212(hard)
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        #  Key idea of the optimization: once we found an answer, delete it from the Trie.
        #  Step1. delete the {"#":0} -> {}
        #  Step2. In the previous level, delete { letter:{} }

        #  DFS in Trie:
        def helper(i, j, curWord, curNode):
            #  Base case
            if "#" in curNode:
                ans.append(curWord[:])
                del curNode["#"]
                #  We don't return here because words in Trie might have the same prefix.

            #  When current letter is invalid
            if not (0 <= i < rows and 0 <= j < cols and board[i][j] in curNode): return
            #  When current letter is valid (matches the node in the Tire).
            oriWord = board[i][j]
            board[i][j] = "$"
            for x, y in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                helper(x, y, curWord + oriWord, curNode[oriWord])
            if not curNode[oriWord]: del curNode[oriWord]
            board[i][j] = oriWord

        #  Build the Trie
        trie = {}
        for word in words:
            node = trie
            for char in word:
                node[char] = node.get(char, {})
                node = node[char]
            node["#"] = "#"

        rows, cols, ans = len(board), len(board[0]), []
        for i in range(rows):
            for j in range(cols):
                helper(i, j, "", trie)
        return ans

    #  547(medium)
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        def _parent(i):
            node = i
            while p[node] != node:
                node = p[node]
            root, node = node, i
            while p[node] != root:
                nextNode = p[node]
                root = p[node]
                node = nextNode
            return root

        def _union(i, j):
            p1 = _parent(i)
            p2 = _parent(j)
            p[p2] = p1

        # Initialize the disjoint set.
        n, ans = len(isConnected), 0
        p = [i for i in range(n)]

        for i in range(n):
            for j in range(n):
                if i != j and isConnected[i][j]:
                    _union(i, j)
        for i in range(n):
            if p[i] == i: ans += 1
        return ans

    #  130(medium)
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        def dfsHelper(i, j):
            if not (0 <= i < rows and 0 <= j < cols and board[i][j] == "O"): return
            board[i][j] = "$"
            for di, dj in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                dfsHelper(i + di, j + dj)

        rows, cols = len(board), len(board[0])
        for i in [0, rows - 1]:
            for j in range(cols):
                dfsHelper(i, j)
        for i in range(rows):
            for j in [0, cols - 1]:
                dfsHelper(i, j)
        for i in range(rows):
            for j in range(cols):
                if board[i][j] == "$":
                    board[i][j] = "O"
                elif board[i][j] == "O":
                    board[i][j] = "X"

    #  36(medium)
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rows = [set(range(1, 10)) for _ in range(9)]
        cols = [set(range(1, 10)) for _ in range(9)]
        blocks = [set(range(1,10)) for _ in range(9)]
        for i in range(9):
            for j in range(9):
                if board[i][j] == ".": continue
                pivot = int(board[i][j])
                k = (i // 3) * 3 + j // 3
                if pivot not in rows[i]: return False
                if pivot not in cols[j]: return False
                if pivot not in blocks[k]: return False
                rows[i].remove(pivot)
                cols[j].remove(pivot)
                blocks[k].remove(pivot)
        return True

    #  37(hard)
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """

        def _helper(iter=0):
            if iter == len(remain): return True
            (i, j) = remain[iter]
            k = (i // 3) * 3 + j // 3
            for val in (rows[i] & cols[j] & blocks[k]):
                board[i][j] = str(val)
                rows[i].remove(val)
                cols[j].remove(val)
                blocks[k].remove(val)

                if _helper(iter + 1): return True

                rows[i].add(val)
                cols[j].add(val)
                blocks[k].add(val)
                board[i][j] = "."

        rows = [set(range(1, 10)) for _ in range(9)]
        cols = [set(range(1, 10)) for _ in range(9)]
        blocks = [set(range(1, 10)) for _ in range(9)]
        remain = []
        for i in range(9):
            for j in range(9):
                if board[i][j] == ".":
                    remain.append((i, j))
                    continue
                key, k = int(board[i][j]), (i // 3) * 3 + j // 3
                rows[i].remove(key)
                cols[j].remove(key)
                blocks[k].remove(key)
        _helper()

    #  1091(medium)
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        if grid[0][0] or grid[-1][-1]: return -1
        queue, n = [(0, 0, 1)], len(grid)
        for i, j, levels in queue:
            if i == j == n - 1: return levels
            for di, dj in [[i, j - 1], [i, j + 1], [i - 1, j], [i + 1, j], [i - 1, j - 1], [i + 1, j - 1],
                           [i - 1, j + 1], [i + 1, j + 1]]:
                if not (0 <= di < n and 0 <= dj < n and grid[di][dj] == 0): continue
                grid[di][dj] = 1
                queue += [(di, dj, levels + 1)]
        return -1

    #  773(hard)
    def slidingPuzzle(self, board: List[List[int]]) -> int:
        moves = [[1, 3], [0, 2, 4], [1, 5], [0, 4], [1, 3, 5], [2, 4]]
        board = board[0] + board[1]
        s = "".join(str(char) for char in board)
        queue, levels, visited = [board], 0, set([s])
        while queue:
            nextQueue = []
            for board in queue:
                if board == [1, 2, 3, 4, 5, 0]: return levels
                index = board.index(0)
                for move in moves[index]:
                    newBoard = board[:]
                    newBoard[index], newBoard[move] = newBoard[move], newBoard[index]
                    if "".join(str(char) for char in newBoard) in visited: continue
                    visited.add("".join(str(char) for char in newBoard))
                    nextQueue += [newBoard]
            queue = nextQueue
            levels += 1
        return -1

    #  191(easy)
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n:
            n = n & (n - 1)
            count += 1
        return count

    #  231(easy)
    def isPowerOfTwo(self, n: int) -> bool:
        return n and not (n & (n - 1))

    #  190(easy)
    def reverseBits(self, n: int) -> int:
        ans = 0
        for _ in range(32):
            ans = (ans << 1) + (n & 1)
            n = n >> 1
        return ans

    #  52(hard)
    def totalNQueens(self, n: int) -> int:
        def helper(i=0, cols=0, diags=0, backDiags=0):
            if i == n:
                self.count += 1
                return
            bits = ~(cols | diags | backDiags) & (1 << n) - 1
            while bits:
                p = bits & -bits
                bits = bits & (bits - 1)
                helper(i + 1, cols | p, (diags | p) >> 1, (backDiags | p) << 1)
        self.count = 0
        helper()
        return self.count

    #  338(easy)
    def countBits(self, n: int) -> List[int]:
        ans = [0] * (n + 1)
        for i in range(1, n + 1):
            ans[i] = ans[i >> 1] + (i & 1)
        return ans

    #  1122(easy)
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        ans, remain, hashmap = [], [], {}
        for num in arr2: hashmap[num] = 0
        for num in arr1:
            if num in hashmap:
                hashmap[num] += 1
            else:
                remain.append(num)
        for num in arr2:
            ans += [num] * hashmap[num]
        remain.sort()
        return ans + remain

    #  1244(medium)
    class Leaderboard:

        def __init__(self):
            self.hashmap = {}

        def addScore(self, playerId: int, score: int) -> None:
            self.hashmap[playerId] = self.hashmap.get(playerId, 0) + score

        def top(self, K: int) -> int:
            return sum(sorted(self.hashmap.values(), reverse=True)[:K])

        def reset(self, playerId: int) -> None:
            self.hashmap.pop(playerId)

    #  56(medium)
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        #  Step1. sort the intervals using their left boundaries.
        intervals = sorted(intervals, key=lambda x: x[0])
        #  Step2. Look through the sorted intervals and merge overlapped intervals.
        left, right, ans = intervals[0][0], intervals[0][1], []
        for i in range(1, len(intervals)):
            #  Update the right boundary.
            if intervals[i][0] <= right and intervals[i][1] > right:
                right = intervals[i][1]
            elif intervals[i][0] > right:
                ans += [[left, right]]
                left, right = intervals[i][0], intervals[i][1]
        return ans + [[left, right]]

    #  493(hard)
    def reversePairs(self, nums: List[int]) -> int:
        def _getReverses(low, high):
            ans = 0
            if low < high:
                mid = (low + high) >> 1
                ans += _getReverses(low, mid)
                ans += _getReverses(mid + 1, high)
                ans += _crossReverses(low, mid, high)
            return ans

        def _crossReverses(low, mid, high):
            l = [nums[i] for i in range(low, mid + 1)]
            r = [nums[i] for i in range(mid + 1, high + 1)]

            #  Count the reverses
            i, j, counts = 0, 0, 0
            while i < len(l) and j < len(r):
                while i < len(l) and l[i] <= 2 * r[j]:
                    i += 1
                else:
                    if i == len(l): break
                    counts += len(l) - i
                    j += 1

            #  Merge l and r
            l.append(math.inf)
            r.append(math.inf)
            i, j = 0, 0
            for k in range(low, high + 1):
                if l[i] <= r[j]:
                    nums[k] = l[i]
                    i += 1
                else:
                    nums[k] = r[j]
                    j += 1
            return counts

        return _getReverses(0, len(nums) - 1)

    #  746(easy)
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        @cache
        def helper(j):
            if j == 0 or j == 1: return 0
            return min(helper(j - 1) + cost[j - 1], helper(j - 2) + cost[j - 2])

        return helper(len(cost))

    #  300(medium)
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = []
        for num in nums:
            index = bisect_left(dp, num)
            if index == len(dp):
                dp.append(num)
            else:
                dp[index] = num
        return len(dp)

    #  115(hard)
    def numDistinct(self, s: str, t: str) -> int:
        rows, cols = len(s) + 1, len(t) + 1
        dp = [[0] * cols for _ in range(rows)]
        for i in range(rows): dp[i][0] = 1
        for j in range(1, cols):
            for i in range(j, rows):
                dp[i][j] = dp[i - 1][j]
                if s[i - 1] == t[j - 1]:
                    dp[i][j] += dp[i - 1][j - 1]
        return dp[-1][-1]

    #  818(hard)
    def racecar(self, target: int) -> int:
        queue, levels = [(0, 1)], 0
        while queue:
            nextQueue = []
            for pos, speed in queue:
                if pos == target: return levels
                nextQueue.append((pos + speed, speed * 2))
                if (pos + speed > target and speed > 0) or (pos + speed < target and speed < 0):
                    nextQueue.append((pos, int(-speed/abs(speed))))
            queue = nextQueue
            levels += 1

    #  709(easy)
    def toLowerCase(self, s: str) -> str:
        return s.lower()

    #  58(easy)
    def lengthOfLastWord(self, s: str) -> int:
        return len(s.split()[-1])

    #  771(easy)
    def numJewelsInStones(self, jewels: str, stones: str) -> int:
        hashmap, count = {}, 0
        for jewel in jewels:
            hashmap[jewel] = 1
        for stone in stones:
            if stone in hashmap.keys():
                count += 1
        return count

    #  387(easy)
    def firstUniqChar(self, s: str) -> int:
        hashmap = collections.OrderedDict()
        for c in s:
            hashmap[c] = hashmap.get(c, 0) + 1
        for key, val in hashmap.items():
            if val == 1: return s.index(key)
        return -1

    #  8(medium)
    #  经常出错，注意各种极端输入
    def myAtoi(self, s: str) -> int:
        s = s.split()
        if not s: return 0
        #  Step1. 得到第一个字符串
        chars = s[0]

        #  Step2.
        flag = 0  # 表示不是数字
        if chars[0] == "-":
            flag = -1
        elif chars[0] == "+" or chars[0].isdecimal():
            flag = 1
        left = 1 if chars[0] == "+" or chars[0] == "-" else 0
        right = len(chars) - 1

        for i in range(left, len(chars)):
            if not chars[i].isdecimal():
                right = i - 1
                break
        if right < left:
            return 0
        else:
            return sorted([-2 ** 31, int(chars[left: right + 1]) * flag, 2 ** 31 - 1])[1]

    #  14(easy)
    def longestCommonPrefix(self, strs: List[str]) -> str:
        #  Build the Trie
        #  当读入空字符串时，Trie会直接存入"#": "#"
        trie = {}
        for word in strs:
            node = trie
            for char in word:
                node[char] = node.get(char, {})
                node = node[char]
            node["#"] = "#"

        # Find the longest common path.
        ans, node = 0, trie
        for char in strs[0]:
            if len(node.values()) > 1: return strs[0][:ans]
            ans += 1
            node = node[char]
        #  如果输入数组中每个字符串都一样的情况
        return strs[0][:ans]

    #  344(easy)
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        i, j = 0, len(s) - 1
        while i < j:
            s[i], s[j] = s[j], s[i]
            i += 1
            j -= 1

    #  541(easy)
    def reverseStr(self, s: str, k: int) -> str:
        s = list(s)
        for i in range(0, len(s), 2 * k):
            s[i: i + k] = s[i: i + k][::-1]
        return "".join(s)

    #  557(easy)
    def reverseWords(self, s: str) -> str:
        return " ".join([word[::-1] for word in s.split()])

    #  917(easy)
    def reverseOnlyLetters(self, s: str) -> str:
        stack = [char for char in s if char.isalpha()]
        return "".join([char if not char.isalpha() else stack.pop() for char in s])

    #  151(medium)
    def reverseWords(self, s: str) -> str:
        return " ".join(s.split()[::-1])

    #  438(medium)
    def findAnagrams(self, s: str, p: str) -> List[int]:
        target, count, i, j, ans = Counter(p), len(p), 0, -1, []
        while i <= len(s) - len(p) and j < len(s):
            #  When window is full
            if j >= i and j - i + 1 == len(p):
                #  Indicate this is an answer
                if count == 0:
                    ans += [i]
                # . Move the left pointer
                target[s[i]] += 1
                if target[s[i]] > 0:
                    count += 1
                i += 1
            #  When the window is not full
            else:
                #  Move the right pointer
                j += 1
                #  If s[j] is valid
                if s[j] in target:
                    target[s[j]] -= 1
                    if target[s[j]] >= 0:
                        count -= 1
                #  If s[j] is not valid, we make a jump.
                else:
                    i = j + 1
                    target, count = Counter(p), len(p)
        return ans

    #  680(easy)
    def validPalindrome(self, s: str) -> bool:
        i, j = 0, len(s) - 1
        while i < j:
            if s[i] != s[j]:
                return s[i + 1: j + 1] == s[i + 1: j + 1][::-1] or s[i: j] == s[i: j][::-1]
            i += 1
            j -= 1
        return True


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
    S.moveZeroes(nums01)
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

    #  16 (medium)
    #  206 (easy)
    print(linkedlist2Array(S.reverseList(array2Linkedlist([1, 2, 3, 4, 5]))))

    #  141
    #  66 (easy)
    print(S.plusOne([1, 2, 3]))
    print(S.plusOne([4, 3, 2, 1]))
    print(S.plusOne([9]))

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

    #  24 (medium)
    print(linkedlist2Array(S.swapPairs(array2Linkedlist([1, 2, 3, 4]))))
    print(linkedlist2Array(S.swapPairs(array2Linkedlist([1]))))
    print(linkedlist2Array(S.swapPairs(array2Linkedlist([]))))

    #  189 (medium)
    nums = [1, 2, 3, 4, 5, 6, 7]
    S.rotate(nums, 3)
    print(nums)
    nums = [-1, -100, 3, 99]
    S.rotate(nums, 2)
    print(nums)
    nums = [-1]
    S.rotate(nums, 2)
    print(nums)

    #  142 (medium)
    #  25 (hard)
    print(linkedlist2Array(S.reverseKGroup(array2Linkedlist([1, 2, 3, 4, 5]), 2)))
    print(linkedlist2Array(S.reverseKGroup(array2Linkedlist([1, 2, 3, 4, 5]), 3)))

    #  20 (easy)
    print(S.isValid("()"))
    print(S.isValid("()[]{}"))
    print(S.isValid("(]"))
    print(S.isValid("]"))

    #  155(medium)
    #  84(hard)
    print(S.largestRectangleArea([2, 1, 5, 6, 2, 3]))
    print(S.largestRectangleArea([2, 4]))

    #  239(hard)
    print(S.maxSlidingWindow([1, 3, -1, -3, 5, 3, 6, 7], 3))
    print(S.maxSlidingWindow([1], 1))

    #  641(medium)
    #  42(hard)
    print(S.trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))
    print(S.trap([4, 2, 0, 3, 2, 5]))

    #  242(easy)
    print(S.isAnagram("anagram", "nagaram"))
    print(S.isAnagram("rat", "cat"))

    #  49(medium)
    print(S.groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
    print(S.groupAnagrams([""]))
    print(S.groupAnagrams(["a"]))

    #  94(easy)
    #  144(easy)
    #  589(easy)
    #  590(easy)
    #  429(medium)
    #  257(easy)
    #  783(easy)
    print(S.minDiffInBST(deserialize("[4, 2, 6, 1, 3]")))

    #  938(easy)
    print(S.rangeSumBST(deserialize("[10,5,15,3,7,null,18]"), 7, 15))

    #  98(easy)
    #  530(easy)
    #  105(medium)
    #  106(medium)

    #  230(medium)
    S.kthSmallest(deserialize("[3,1,4,null,2]"), 1)

    #  687(medium)
    #  543(easy)
    #  22(medium)
    print(S.generateParenthesis(1))
    print(S.generateParenthesis(3))

    #  226(easy)
    #  104(easy)
    #  111(easy)
    #  297(hard)
    #  236(medium)

    #  77(medium)
    print(S.combine(1, 1))
    print(S.combine(4, 2))

    #  46(medium)
    print(S.permute([1, 2, 3]))
    print(S.permute([0, 1]))

    #  47(medium)
    #  78(medium)
    print(S.subsets([1, 2, 3]))
    print(S.subsets([0]))

    #  90(medium)
    print(S.subsetsWithDup([1, 2, 2]))
    print(S.subsetsWithDup([0]))
    print(S.subsetsWithDup([4, 4, 4, 1, 4]))

    #  50(medium)
    #  169(easy)
    print(S.majorityElement([3, 2, 3]))
    print(S.majorityElement([2, 2, 1, 1, 1, 2, 2]))
    print(S.majorityElement([3, 3, 4]))

    #  17(medium)
    print(S.letterCombinations("23"))
    print(S.letterCombinations(""))
    print(S.letterCombinations("2"))

    #  51(hard)
    print(S.solveNQueens(1))
    print(S.solveNQueens(4))

    #  102(medium)
    print(S.levelOrder102(deserialize("[3,9,20,null,null,15,7]")))
    print(S.levelOrder102(deserialize("[1]")))

    #  515(medium)
    #  200(medium)
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

    #  529(medium)
    print(S.updateBoard([
        ["E", "E", "E", "E", "E"],
        ["E", "E", "M", "E", "E"],
        ["E", "E", "E", "E", "E"],
        ["E", "E", "E", "E", "E"]
    ], [3, 0]))
    print(S.updateBoard([
        ["B", "1", "E", "1", "B"],
        ["B", "1", "M", "1", "B"],
        ["B", "1", "1", "1", "B"],
        ["B", "B", "B", "B", "B"]
    ], [1, 2]))

    #  433(median)
    print(S.minMutation("AACCGGTT", "AACCGGTA", ["AACCGGTA"]))
    print(S.minMutation("AACCGGTT", "AAACGGTA", ["AACCGGTA", "AACCGCTA", "AAACGGTA"]))
    print(S.minMutation("AAAAACCC", "AACCCCCC", ["AAAACCCC", "AAACCCCC", "AACCCCCC"]))

    #  127(hard)
    print(S.ladderLength("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]))
    print(S.ladderLength("hit", "cog", ["hot", "dot", "dog", "lot", "log"]))
    print(S.ladderLength("red", "tax", ["ted", "tex", "red", "tax", "tad", "den", "rex", "pee"]))

    #  126(hard)
    print(S.findLadders("a", "c", ["a", "b", "c"]))
    print(S.findLadders("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]))
    print(S.findLadders("hit", "cog", ["hot", "dot", "dog", "lot", "log"]))
    print(S.findLadders("red", "tax", ["ted", "tex", "red", "tax", "tad", "den", "rex", "pee"]))

    #  860(easy)
    print(S.lemonadeChange([5, 5, 5, 10, 20]))
    print(S.lemonadeChange([5, 5, 10, 20, 5, 5, 5, 5, 5, 5, 5, 5, 5, 10, 5, 5, 20, 5, 20, 5]))

    #  322(medium)
    print(S.coinChange([1, 2, 5], 11))
    print(S.coinChange([2], 3))
    print(S.coinChange([1], 0))

    #  55(medium)
    print(S.canJump([2, 3, 1, 1, 4]))
    print(S.canJump([3, 2, 1, 0, 4]))

    #  45(medium)
    print(S.jump([2, 3, 1, 1, 4]))
    print(S.jump([2, 3, 0, 1, 4]))
    print(S.jump([1, 2, 1, 1, 1]))

    #  455(easy)
    print(S.findContentChildren([1, 2, 3], [1, 1]))
    print(S.findContentChildren([1, 2], [1, 2, 3]))

    #  122(medium)
    print(S.maxProfit([7, 1, 5, 3, 6, 4]))
    print(S.maxProfit([1, 2, 3, 4, 5]))
    print(S.maxProfit([7, 6, 4, 3, 1]))

    #  874(medium)
    print(S.robotSim([4, -1, 3], []))
    print(S.robotSim([4, -1, 4, -2, 4], [[2, 4]]))
    print(S.robotSim([6, -1, -1, 6], []))

    #  69(easy)
    print(S.mySqrt(4))
    print(S.mySqrt(8))

    #  33(medium)
    print(S.search([3, 5, 1], 3))
    print(S.search([4, 5, 6, 7, 0, 1, 2], 0))

    #  367(medium)
    print(S.isPerfectSquare(16))
    print(S.isPerfectSquare(14))

    #  74(medium)
    print(S.searchMatrix([[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]], 3))
    print(S.searchMatrix([[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]], 13))

    #  153(medium)
    print(S.findMin([3, 4, 5, 1, 2]))
    print(S.findMin([4, 5, 6, 7, 0, 1, 2]))
    print(S.findMin([11, 13, 15, 17]))

    #  62(medium)
    print(S.uniquePaths(3, 7))
    print(S.uniquePaths(3, 2))

    #  63(medium)
    print(S.uniquePathsWithObstacles([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))
    print(S.uniquePathsWithObstacles([[0, 1], [0, 0]]))
    print(S.uniquePathsWithObstacles([[0, 0], [1, 1], [0, 0]]))

    #  1143(medium)
    print(S.longestCommonSubsequence("bl", "yby"))
    print(S.longestCommonSubsequence("ace", "abcde"))
    print(S.longestCommonSubsequence("abc", "abc"))
    print(S.longestCommonSubsequence("abc", "def"))

    #  120(medium)
    print(S.minimumTotal([[2], [3, 4], [6, 5, 7], [4, 1, 8, 3]]))
    print(S.minimumTotal([[-10]]))

    #  53(medium)
    print(S.maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))
    print(S.maxSubArray([1]))
    print(S.maxSubArray([5, 4, -1, 7, 8]))

    #  152(medium)
    print(S.maxProduct([2, 3, -2, 4]))
    print(S.maxProduct([-2, 0, -1]))

    #  121(medium)
    print(S.maxProfit121([7, 1, 5, 3, 6, 4]))
    print(S.maxProfit121([7, 6, 4, 3, 1]))

    #  123(hard)
    print(S.maxProfit123([1, 2, 3, 4, 5]))

    #  188(hard)
    print(S.maxProfit188(2, [2, 4, 1]))
    print(S.maxProfit188(2, [3, 3, 5, 0, 0, 3, 1, 4]))

    #  714(medium)
    print(S.maxProfit714([1, 3, 2, 8, 4, 9], 2))
    print(S.maxProfit714([1, 3, 7, 5, 10, 3], 3))

    #  309(medium)
    print(S.maxProfit309([1, 2, 3, 0, 2]))
    print(S.maxProfit309([1]))

    #  198(medium)
    print(S.rob([1, 2, 3, 1]))
    print(S.rob([2, 7, 9, 3, 1]))

    #  213(medium)
    print(S.rob213([2, 3, 2]))
    print(S.rob213([1, 2, 3, 1]))
    print(S.rob213([1, 2, 3]))

    #  279(medium)
    print(S.numSquares(12))
    print(S.numSquares(13))

    #  72(hard)
    print(S.minDistance("horse", "ros"))
    print(S.minDistance("intention", "execution"))

    #  980(hard)
    print(S.uniquePathsIII([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, -1]]))
    print(S.uniquePathsIII([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]]))
    print(S.uniquePathsIII([[0, 1], [2, 0]]))

    #  518(medium)
    print(S.change(5, [1, 2, 5]))
    print(S.change(3, [2]))
    print(S.change(10, [10]))

    #  64(medium)
    #  91(medium)
    print(S.numDecodings("12"))
    print(S.numDecodings("226"))
    print(S.numDecodings("10"))
    print(S.numDecodings("06"))

    #  303(easy)
    #  304(medium)
    #  560(medium)
    #  363(hard)
    print(S.maxSumSubmatrix([[1, 0, 1], [0, -2, 3]], 2))
    print(S.maxSumSubmatrix([[2, 2, -1]], 3))

    #  403(hard)
    #  621(medium)
    print(S.leastInterval(["A", "A", "A", "B", "B", "B"], 2))
    print(S.leastInterval(["A", "A", "A", "B", "B", "B"], 0))
    print(S.leastInterval(["A", "A", "A", "A", "A", "A", "B", "C", "D", "E", "F", "G"], 2))

    #  647(medium)
    #  5(medium)
    #  410(hard)
    print(S.splitArray([7, 2, 5, 10, 8], 2))
    print(S.splitArray([1, 2, 3, 4, 5], 2))
    print(S.splitArray([1, 4, 4], 3))

    #  312(hard)
    print(S.maxCoins([3, 1, 5, 8]))
    print(S.maxCoins([1, 5]))

    #  76(hard)
    #  552(hard)
    print(S.checkRecord(2))
    print(S.checkRecord(1))
    print(S.checkRecord(10101))

    #  208(medium)
    #  79(medium)
    print(S.exist([["A", "B", "C", "E"], ["S", "F", "E", "S"], ["A", "D", "E", "E"]],
    "ABCESEEEFS"))

    #  212(hard)
    #  547(medium)
    #  130(medium)
    #  36(medium)
    #  37(hard)
    #  1091（medium)
    #  773（hard)
    #  191(easy)
    #  231(easy)
    #  190(easy)
    #  52(hard)
    #  338(easy)

    #  Bloom Filter
    bf = BloomFilter(500000, 7)
    bf.add("BianLong")
    print(bf.lookup("BianLong"))
    print(bf.lookup("LiuYing"))

    #  146(medium)
    lRUCache = LRUCache2(2)
    lRUCache.put(1, 1)
    lRUCache.put(2, 2)
    print(lRUCache.get(1))
    lRUCache.put(3, 3)
    print(lRUCache.get(2))
    lRUCache.put(4, 4)
    print(lRUCache.get(1))
    print(lRUCache.get(3))
    print(lRUCache.get(4))

    x = Sort([5,2,7,1,4,2,0,9,8,10])
    x.heapSort()
    print(x.nums)

    #  1122(easy)
    #  242(easy)
    #  1244(medium)
    #  56(medium)
    #  493(hard)
    #  746(easy)
    #  85(hard)
    #  300(medium)
    #  115(hard)
    #  818(hard)
    #  709(easy)
    #  58(easy)
    #  771(easy)
    #  387(easy)
    #  8(medium)
    #  14(easy)
    #  344(easy)
    #  541(easy)
    #  557(easy)
    #  917(easy)
    #  151(medium)
    #  680(easy)
    print("-------------------------------------------------------------")



