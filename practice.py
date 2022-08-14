from functools import cache
from typing import List, Optional
from Solution import deserialize, drawtree
import math
import collections


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
        prev, cur, next, p = ListNode(), head, head.next, k
        while p != 0:
            next = cur.next
            cur.next = prev
            prev, cur = cur, next
            p -= 1

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

    #  127(hard)
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
        def printpath(curWord, curPath):
            if path.get(curWord) is None:
                return
            if curWord == beginWord:
                ans.append(curPath[::-1])
                return
            for prevWord in path[curWord]:
                curPath.append(prevWord)
                printpath(prevWord, curPath)
                curPath.pop()

        def _findLadders(wordList):
            queue, wordList, path = [beginWord], set(wordList), {beginWord: [beginWord]}
            if beginWord in wordList: wordList.remove(beginWord)
            while queue:
                nextQueue, localVisited = set(), set()
                for word in queue:
                    for i in range(len(word)):
                        for char in "abcdefghijklmnopqrstuvwxyz":
                            newWord = word[:i] + char + word[i + 1:]
                            if newWord not in wordList:
                                continue
                            path[newWord] = path[newWord] + [word] if newWord in path.keys() else [word]
                            localVisited.add(newWord)
                            nextQueue.add(newWord)
                queue = list(nextQueue)
                for word in localVisited:
                    wordList.remove(word)
            return path

        ans = []
        path = _findLadders(wordList)
        printpath(endWord, [endWord])
        return ans

    #  860(easy)
    def lemonadeChange(self, bills: List[int]) -> bool:
        five, ten = 0, 0
        for paid in bills:
            if paid == 5: five += 1
            elif paid == 10: five, ten = five - 1, ten + 1
            elif ten > 0: five, ten = five - 1, ten - 1
            else: five -= 3
            if five < 0: return False
        return True

    #  322(medium)
    def coinChange(self, coins: List[int], amount: int) -> int:
        @cache
        def helper( amount ):
            if amount < 0: return float("inf")
            if amount == 0: return 0
            ans = float("inf")
            for coin in coins:
                ans = min(ans, helper( amount - coin ))
            return ans + 1

        ans = helper(amount)
        return ans if ans != float("inf") else -1



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
    print(S.findLadders("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]))
    print(S.findLadders("hit", "cog", ["hot", "dot", "dog", "lot", "log"]))
    print(S.findLadders("red", "tax", ["ted", "tex", "red", "tax", "tad", "den", "rex", "pee"]))

    #  860(easy)
    print(S.lemonadeChange([5, 5, 5, 10, 20]))
    print(S.lemonadeChange([5, 5, 10, 20, 5, 5, 5, 5, 5, 5, 5, 5, 5, 10, 5, 5, 20, 5, 20, 5]))

    print("------------------------------------------------------------------------")
    #  322(medium)
    print(S.coinChange([1,2,5],11))
    print(S.coinChange([2], 3))
    print(S.coinChange([1], 0))
    print("------------------------------------------------------------------------")
