#### [208. 实现 Trie (前缀树)](https://leetcode-cn.com/problems/implement-trie-prefix-tree/)

```python
#python比较简洁的一个写法
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root ={}
        self.end_of_word = '#'
        

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        node = self.root
        for char in word:
            node =node.setdefault(char,{})  #setdefault实现put和get两个功能，如果key为空的话，就拿出来，不然的话就可以直接取，循环走完，就把所有单词全部加到字典树node里面
        node[self.end_of_word] =self.end_of_word   #最后再加一个标志位，end_of_word,否则无法区分是只包含前缀还是包含整个单词

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        node = self.root  #node是一个实例化的字典树
        for char in word:
            if char not in node:
                return False   #如果不在字典树中，就返回false
            node = node[char]  #如果存在就走他下一个char所对应的下一个节点，找出来之后放到新的node中
        return self.end_of_word in node


    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        node = self.root
        for char in prefix:
            if char not in node:
                return False
            node = node[char]
        return True   #与seach的区别是不用关心end_of_word


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```

#### [212. 单词搜索 II](https://leetcode-cn.com/problems/word-search-ii/)

```python
#典型的搜索题目
1.words遍历，时间复杂度O(N*m*m*4^k),N为words长度，m为数组长度/宽度，4是每个单词的字符要去四个方向找，k是单词的长度


2.Trie
a.所有的words放到trie中，构建prefix
b.对board进行DFS,他的起点是遍历每一个字符，DFS产生的每一个字符串然后去trie中查，是不是他的字串
第一种写法：
dx= [-1,1,0,0]
dy = [0,0,-1,1]  #四联通遍历
END_OF_WORD ='#'
class Solution(object):
    def findWords(self,board,words):
        if not board or not board[0]:
            return []
        if not words:
            return []   #判断参数
        self.result = set() #用result放成一个set,这里使用list也是可以的
    
        #构建trie(创建trie，把单词插进trie中)
        root = collections.defaultdict()  #root是一个dict
        for word in words:
            node =root
            for char in word:
                node = node.setdefault(char,collections.defaultdict())
            node[END_OF_WORD] = END_OF_WORD

        self.m,self.n = len(board),len(board[0])
        for i in range(self.m):
            for j in range(self.n):
                if board[i][j] in root:  #这里是一个剪枝，如果字符本身不是trie中任何单词的任何字母就不
                    self._dfs(board,i,j,"",root)    #用管了，只有字符在trie中的单词中了，再dfs
        return list(self.result)  #将result转化为list输出

    def _dfs(self,board,i,j,cur_word,cur_dict):
        cur_word +=board[i][j]   #
        cur_dict = cur_dict[board[i][j]]
        if END_OF_WORD in cur_dict:
            self.result.add(cur_word)      ######以上为递归终止条件
        tmp,board[i][j] =board[i][j],'@'  #'@'字符是为了避免重复使用一个字符，字符用过之后就将其置为@
        for k in range(4):    ######上面两行是处理当前逻辑
            x,y = i+dx[k],j+dy[k]
            if 0 <= x < self.m and 0 <= y <self.n \
                and board[x][y] != '@' and board[x][y] in cur_dict:
                self._dfs(board,x,y,cur_word,cur_dict)     ######以上两行是下探到下一层
        board[i][j] = tmp  ######恢复board[i][j]
        
        
第二种写法（非常显示代码功底的一段代码）：
def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        trie = {}  # 构造字典树
        for word in words:
            node = trie
            for char in word:
                node = node.setdefault(char, {})
            node['#'] = True

        def search(i, j, node, pre, visited):  # (i,j)当前坐标，node当前trie树结点，pre前面的字符串，visited已访问坐标
            if '#' in node:  # 已有字典树结束
                res.add(pre)  # 添加答案
            for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                _i, _j = i+di, j+dj
                if -1 < _i < h and -1 < _j < w and board[_i][_j] in node and (_i, _j) not in visited:  # 可继续搜索
                    search(_i, _j, node[board[_i][_j]], pre+board[_i][_j], visited | {(_i, _j)})  # dfs搜索

        res, h, w = set(), len(board), len(board[0])
        for i in range(h):
            for j in range(w):
                if board[i][j] in trie:  # 可继续搜索
                    search(i, j, trie[board[i][j]], board[i][j], {(i, j)})  # dfs搜索
        return list(res)

```

#### [547. 省份数量](https://leetcode-cn.com/problems/number-of-provinces/)

```python
1.暴力
2.搜索
3.DFS
找到一个省份i,然后对他相邻且没有访问过的省份使用dfs查找。
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        nums = len(isConnected)
        visited = set()
        result = 0

        def dfs(i):
            for j in range(nums):
                if isConnected[i][j] ==1 and j not in visited:
                    visited.add(j)
                    dfs(j)

        for i in range(nums):
            if i not in visited:
                dfs(i)
                result +=1

        return result
     
 4.BFS:
  class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        privence =len(isConnected)
        visited = set()
        result = 0
        

        for i in range(privence):
            if i not in visited:
                q = collections.deque([i])
                while q:
                    j = q.popleft()
                    visited.add(j)
                    for k in range(privence):
                        if isConnected[j][k] == 1 and k not in visited:
                            q.append(k)
                result +=1
        return result 
5.并查集
class UnionFind:
    def __init__(self):
        self.father = {}
        # 额外记录集合的数量
        self.num_of_sets = 0
    
    def find(self,x):
        root = x
        
        while self.father[root] != None:
            root = self.father[root]
        
        while x != root:
            original_father = self.father[x]
            self.father[x] = root
            x = original_father
        
        return root
    
    def merge(self,x,y):
        root_x,root_y = self.find(x),self.find(y)
        
        if root_x != root_y:
            self.father[root_x] = root_y
            # 集合的数量-1
            self.num_of_sets -= 1
    
    def add(self,x):
        if x not in self.father:
            self.father[x] = None
            # 集合的数量+1
            self.num_of_sets += 1

class Solution:
    def findCircleNum(self, M: List[List[int]]) -> int:
        uf = UnionFind()
        for i in range(len(M)):
            uf.add(i)
            for j in range(i):
                if M[i][j]:
                    uf.merge(i,j)
        
        return uf.num_of_sets


```

200

```python

```

130

```python

```

70

```python
#动态规划
#递归+去重
#转换为零钱兑换问题
#
```

#### [22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/)

```python
#剪枝
#DP(https://leetcode-cn.com/problems/generate-parentheses/solution/zui-jian-dan-yi-dong-de-dong-tai-gui-hua-bu-lun-da/)
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        if n == 0:
            return []
        total_l = []
        total_l.append([None])    # 0组括号时记为None
        total_l.append(["()"])    # 1组括号只有一种情况
        for i in range(2,n+1):    # 开始计算i组括号时的括号组合
            l = []        
            for j in range(i):    # 开始遍历 p q ，其中p+q=i-1 , j 作为索引
                now_list1 = total_l[j]    # p = j 时的括号组合情况
                now_list2 = total_l[i-1-j]    # q = (i-1) - j 时的括号组合情况
                for k1 in now_list1:  
                    for k2 in now_list2:
                        if k1 == None:
                            k1 = ""
                        if k2 == None:
                            k2 = ""
                        el = "(" + k1 + ")" + k2
                        l.append(el)    # 把所有可能的情况添加到 l 中
            total_l.append(l)    # l这个list就是i组括号的所有情况，添加到total_l中，继续求解i=i+1的情况
        return total_l[n]

#另一种解法
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        def generate(p, left, right):
            if right >= left >= 0:
                if not right:
                    yield p
                for q in generate(p + '(', left-1, right): yield q
                for q in generate(p + ')', left, right-1): yield q
        return list(generate('', n, n))
```

#### [51. N 皇后](https://leetcode-cn.com/problems/n-queens/)

```python
#非位运算解决N皇后问题最漂亮的代码
class Solution:
    def solveNQueens(self, n):
        def DFS(queens, xy_dif, xy_sum):
            p = len(queens)
            if p==n:
                result.append(queens)
                return None
            for q in range(n):
                if q not in queens and p-q not in xy_dif and p+q not in xy_sum: 
                    DFS(queens+[q], xy_dif+[p-q], xy_sum+[p+q])   #将queens,p-q,p+q放在数组里面，然后不断循环
        result = []
        DFS([],[],[])
        return [ ["."*i + "Q" + "."*(n-i-1) for i in sol] for sol in result]   #一行代码生成棋盘
以上代码可以优化的地方是：可以把dfs写出去，

#位运算（更加简洁和快速）
```

#### [36. 有效的数独](https://leetcode-cn.com/problems/valid-sudoku/)

```python
#没有回溯和剪枝，只通过循环来进行判断（https://leetcode-cn.com/problems/valid-sudoku/solution/you-xiao-de-shu-du-by-leetcode/）
class Solution:
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        # init data
        rows = [{} for i in range(9)]
        columns = [{} for i in range(9)]
        boxes = [{} for i in range(9)]

        # validate a board
        for i in range(9):
            for j in range(9):
                num = board[i][j]
                if num != '.':
                    num = int(num)
                    box_index = (i // 3 ) * 3 + j // 3
                    
                    # keep the current cell value
                    rows[i][num] = rows[i].get(num, 0) + 1
                    columns[j][num] = columns[j].get(num, 0) + 1
                    boxes[box_index][num] = boxes[box_index].get(num, 0) + 1
                    
                    # check if this value has been already seen before
                    if rows[i][num] > 1 or columns[j][num] > 1 or boxes[box_index][num] > 1:
                        return False         
        return True
```

#### [37. 解数独](https://leetcode-cn.com/problems/sudoku-solver/)

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        row = [set(range(1,10)) for i_ in range(9)] #行剩余可用数字
        col = [set(range(1,10)) for i_ in range(9)] #列剩余可用数字
        block = [set(range(1,10)) for i_ in range(9)] #块剩余可用数字
        ##以上处理类似于八皇后问题的横撇捺，有点是不用再后面多一个循环

        empty = []  #收集需要填数位置
        for i in range(9):      #预扫码，对整个棋盘进行遍历，如果有数字就去更新三个set
            for j in range(9):
                if board[i][j] != '.':   #更新可用数字
                    val = int(board[i][j])
                    row[i].remove(val)
                    col[j].remove(val)
                    block[(i//3)*3+j//3].remove(val)
                else:
                    empty.append((i,j))   #把空格子放到empty中
                    
        #接下来就是直接调用backtrack

        def backtrack(iter=0):
            if iter == len(empty):   #处理完empty代表找到了答案
                return True
            i,j = empty[iter]
            b = (i//3)*3+j//3
            for val in row[i] & col[j] & block[b]:
                row[i].remove(val)
                col[j].remove(val)
                block[b].remove(val)
                board[i][j] =str(val)  #将val加进去
                if backtrack(iter+1):  #然后进行下一层递归调用，最终加进去如果能够解决，就返回True
                    return True
                row[i].add(val)    #回溯,如果不能够解决，就恢复这一层的状态，然后再return false出去
                col[j].add(val)
                block[b].add(val)
            return False
        backtrack()

#解决这个题之后，可以结合安卓手机和OpenCV截图，写一个手机外挂，解决手机数独的游戏
```

#### [127. 单词接龙](https://leetcode-cn.com/problems/word-ladder/)

```python
#BFS
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        word_set = set(wordList)
        if len(word_set) == 0 or endWord not in word_set:
            return 0

        if beginWord in word_set:
            word_set.remove(beginWord)

        queue = deque()
        queue.append(beginWord)

        visited = set(beginWord)

        word_len = len(beginWord)
        step = 1
        while queue:
            current_size = len(queue)
            for i in range(current_size):
                word = queue.popleft()

                word_list = list(word)
                for j in range(word_len):
                    origin_char = word_list[j]

                    for k in range(26):
                        word_list[j] = chr(ord('a') + k)
                        next_word = ''.join(word_list)
                        if next_word in word_set:
                            if next_word == endWord:
                                return step + 1
                            if next_word not in visited:
                                queue.append(next_word)
                                visited.add(next_word)
                    word_list[j] = origin_char
            step += 1
        return 0
    
#DFS+剪枝

#Two_ended BFS
import string
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in wordList:
            return 0

        front = {beginWord}    #besign set
        back = {endWord}       #end set
        dist = 1               #distance,走的步数
        wordList = set(wordList)   #wordlist也做一个set，方便查找，setO(1),listO(n)
        word_len = len(beginWord)

        #开始BFS
        while front:   #等价于 while front and back
            dist +=1
            next_front = set()    #存放后面扩散出来的点
            for word in front:
                for i in range(word_len):
                    for c in string.ascii_lowercase:   #遍历'a'~'z'
                        if c != word[i]:
                            new_word =word[:i]+c+word[i+1:]
                            if new_word in back:  #从front中扩散出来的单词在back中，说明相交了
                                return dist
                            if new_word in wordList:
                                next_front.add(new_word)
                                wordList.remove(new_word)
            front = next_front
            if len(back)<len(front):
                front,back =back,front
        return 0



```

433

```python
#BFS
#双向BFS
```

#### [1091. 二进制矩阵中的最短路径](https://leetcode-cn.com/problems/shortest-path-in-binary-matrix/)

```python
#dp
#bfs
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        q,n = [(0,0,2)],len(grid)    #[0,0,2]代表起点，终点，步数
        if grid[0][0] or grid[-1][-1]:   #判断左上和右下
            return -1
        elif n <=2:
            return n
        #BFS start
        for i,j,d in q:
            for x,y in [(i-1,j-1),(i-1,j),(i-1,j+1),(i,j-1),(i,j+1),(i+1,j-1),(i+1,j),(i+1,j+1)]:
                if 0<= x <n and 0<= y <n and not grid[x][y]:
                    if x == n-1 and y == n-1:  #判断是否到终点了
                        return d
                    q += [(x,y,d+1)]
                    grid[x][y] =1
        return -1
#A*

```

#### [773. 滑动谜题](https://leetcode-cn.com/problems/sliding-puzzle/)

```python
#DFS
#BFS-更快的找到最优解(下面是一个比较好的代码)
##方向变换向量，类似四联通图和八连通图,
##最终2*3的矩阵转换为一维数组
moves = {
    0:[1,3],
    1:[0,2,4],
    2:[1,5],
    3:[0,4],
    4:[1,3,5],
    5:[2,4]
}
class Solution:
    def slidingPuzzle(self, board: List[List[int]]) -> int:
        used = set()   #已经访问过的
        cnt = 0
        s = "".join(str(c) for row in board for c in row)   #二维转化为一维字符串

        q = [s]   #123405
        while q:
            new =[]
            for s in q:
                used.add(s)
                if s == "123450":
                    return cnt
                arr =[c for c in s]

                #开始移动0
                zero_index = s.index("0")
                for move in moves[zero_index]:
                    new_arr = arr[:]
                    new_arr[zero_index],new_arr[move] = new_arr[move],new_arr[zero_index]
                    new_s = "".join(new_arr)
                    if new_s not in used:
                        new.append(new_s)
            cnt +=1
            q = new
        return -1
#A*

```

