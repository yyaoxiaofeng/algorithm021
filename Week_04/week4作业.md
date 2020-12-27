

#### [102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

```python
#1.BFS
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        quere = collections.deque()
        quere.append(root)
        res =[]
        while quere:
            level =[]
            size = len(quere)            
            for i in range(size):
                cur = quere.popleft()
                if not cur:
                    continue
                level.append(cur.val)
                quere.append(cur.left)
                quere.append(cur.right)
            if level:                  #注意这里的判断，否则最终就俄国会多一个空数组
                res.append(level)
        return res

#2.DFS

```

#### [433. 最小基因变化](https://leetcode-cn.com/problems/minimum-genetic-mutation/)

```python
#广度优先
class Solution:
    def minMutation(self, start: str, end: str, bank: List[str]) -> int:
        possible = ['A','C','G','T']
        queue = [(start,0)]
        while queue:
            (word,step) = queue.pop(0)
            if word  == end:
                return step
            for i in range(len(start)):
                for j in possible:
                    temp = word[:i]+j+word[i+1:]
                    if temp in bank:
                        bank.remove(temp)
                        queue.append((temp,step+1))
        return -1
```

#### [22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/)

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        ans =[]
        def backstack(S,left,right):
            if len(S)  == 2*n:
                ans.append(''.join(S))
            if left < n:
                S.append('(')
                backstack(S,left+1,right)
                S.pop()
            if right <left:
                S.append(')')
                backstack(S,left,right+1)
                S.pop()
        backstack([],0,0)
        return ans

```

515

```


```

#### [69. x 的平方根](https://leetcode-cn.com/problems/sqrtx/)

```python
#二分法
class Solution:
    def mySqrt(self, x: int) -> int:
        if x == 0 or x == 1:
            return x
        left,right,mid = 1,x+1,1    #为什么right初始值要是x+1?
        while left <=right:
            mid = left +(right-left)//2   #为什么这里要取整？
            if mid*mid >x:
                right =mid-1
            else:
                left = mid+1
        return int(right)

#牛顿迭代法
class Solution:
    def mySqrt(self, x: int) -> int:
        r = x
        while r*r >x:
            r = (r + x/r) /2

        return r


```

367

```

```



#### [860. 柠檬水找零](https://leetcode-cn.com/problems/lemonade-change/)

```python
#模拟+贪心
class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        five = 0
        ten = 0
        for bill in bills:
            if bill == 5:
                five += 1
            elif bill == 10:
                if five >0:
                    five -=1
                    ten +=1
                else:
                    return False
            else:
                if five >0 and ten >0:   #贪心的体现，当出现20时，优先使用十元找零
                    five -=1
                    ten -=1
                elif five >3:
                    five -=3
                else:
                    return False
        return True

```

#### [122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

```python
#一次遍历
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        sum = 0
        for i in range(1,len(prices)):
            if prices[i] > prices[i-1]:   #一次遍历，每次都将今天的价格和昨天的价格座椅对比，
                sum += prices[i] -prices[i-1]   #如果今天大，就将利润加到最终利润，如果小，就不做任何处理
        return sum

```

#### [455. 分发饼干](https://leetcode-cn.com/problems/assign-cookies/)

```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()
        s_leng = len(s)
        g_leng = len(g)
        res =0
        i,j =0,0
        while i < g_leng and j < s_leng:
            if g[i] <= s[j]:
                res +=1
                i +=1
                j +=1
            else:
                j +=1
        return res
```

874

```

```

#### [127. 单词接龙](https://leetcode-cn.com/problems/word-ladder/)

```python
#BFS
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        st = set(wordList)
        if endWord not in st:
            return 0
        queue = collections.deque()
        queue.append((beginWord,1))
        visited = set()
        visited.add(beginWord)
        m = len(beginWord)

        while queue:
            cul,step = queue.popleft()
            if cul == endWord:
                return step
            for i in range(m):
                for j in range(26):
                    temp = cul[:i]+chr(97+j)+cul[i+1:]
                    if temp in st and temp not in visited:
                        queue.append((temp,step+1))
                        visited.add(temp)
        return 0
```

#### [200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

```python
1.BFS

2.DFS
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        
        def dfsmarking(i,j):
            if i<0 or j<0 or i>=n or j>=m or grid[i][j] != '1':
                return      #i代表行，j代表列，如果i,j越界或者不为1就跳出
            
            grid[i][j]  = '0'    #将找到的1置为0
            dfsmarking(i+1,j)
            dfsmarking(i-1,j)
            dfsmarking(i,j+1)
            dfsmarking(i,j-1)  #使用递归将1周围的1也都置为0

        count = 0
        if not grid:
            return 0
        n = len(grid)
        m = len(grid[0])
        for i in range(n):
            for j in range(m):
                if grid[i][j] == '1':
                    count +=1
                    print(count)
                    dfsmarking(i,j)  #找到一个一个1后，将其以及周围的1都置为0
        return count
    
3.并查集


```

#### [529. 扫雷游戏](https://leetcode-cn.com/problems/minesweeper/)

```python
#DFS
class Solution:
        def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
            direction = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1))
            if board[click[0]][click[1]] == 'M':
                board[click[0]][click[1]] = 'X'
                return board
            self.m,self.n = len(board), len(board[0])
            def check(i, j):
                cnt = 0
                for x,y in direction:
                    x, y = x + i, y + j
                    if 0 <= x < self.m and 0 <= y < self.n and board[x][y]=='M':
                        cnt += 1
                return cnt    
            def dfs(i, j):
                cnt = check(i, j)
                if not cnt:
                    board[i][j] = 'B'
                    for x, y in direction:
                        x, y = x + i, y + j
                        if  0 <= x < self.m and 0 <= y < self.n and board[x][y]=='E': dfs(x, y)
                else: board[i][j] = str(cnt)
            dfs(click[0],click[1])
            return board
```

#### [55. 跳跃游戏](https://leetcode-cn.com/problems/jump-game/)

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        if len(nums) == 0:
            return False
        endreachable = len(nums)-1
        
        for i in range(endreachable,-1,-1):

            if nums[i]+i >= endreachable:
                endreachable = i 

        return endreachable == 0
```

33

```

```

74

```

```

153

```

```

126

```

```

45

```

```

[387. 字符串中的第一个唯一字符](https://leetcode-cn.com/problems/first-unique-character-in-a-string/)

```python
#map统计频次
class Solution:
    def firstUniqChar(self, s: str) -> int:
        frequence = collections.Counter(s)   #使用函数统计频次
        for i,ch in enumerate(s):            #返回第一个频次为1的字母
            if frequence[ch] == 1:
                return i
        return -1
#map统计频次
class Solution:
    def firstUniqChar(self, s: str) -> int:
        result = {}
        for i in s:
            if result.get(i) == None:
                result[i] =1
            else:
                result[i] +=1      ##使用map手写统计字母频次

        for i,ch in enumerate(s):
            if result[ch] == 1:
                return i
        return -1
 #map统计索引：
class Solution:
    def firstUniqChar(self, s: str) -> int:
        map ={}
        for i,ch in enumerate(s):     #如果这个字母不在map中，就记录这个字母的索引
            if map.get(ch) == None:   #如果这个字母已经在map中，就将这个字母的索引置为-1
                map[ch] = i
            else:
                map[ch] = -1
        for j in map:         
            if map[j] != -1:
                return map[j]          #遍历map，第一个索引不为-1的就是第一个非重复元素
        return -1

    
#队列
class Solution:
    def firstUniqChar(self, s: str) -> int:
        posion = {}
        q = collections.deque()
        n = len(s)
        for i,ch in enumerate(s):
            if posion.get(ch) == None:   #或者写成if ch not in posion
                posion[ch] =i
                q.append((s[i],i))
            else:
                posion[ch] = -1
                while q and posion[q[0][0]] == -1:
                    q.popleft()
        return -1 if not q else q[0][1]


```

