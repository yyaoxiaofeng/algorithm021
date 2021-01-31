#### [191. 位1的个数](https://leetcode-cn.com/problems/number-of-1-bits/)

```python
1.位运算，n&(n-1)是清零最低位的1，每清零一次1，计数加一，当n为零时，退出循环
class Solution:
    def hammingWeight(self, n: int) -> int:
        count =0
        while n != 0:
            n = n&(n-1)
            count +=1
        return count
  
#优秀解法题解：
作者：jalan
链接：https://leetcode-cn.com/problems/number-of-1-bits/solution/python-de-si-chong-xie-fa-by-jalan/
#调用库函数count
class Solution(object):
        def hammingWeight(self, n):
            """
        :type n: int
        :rtype: int
        """
            return bin(n).count('1')
        
        
#手写循环,计数
class Solution:
    def hammingWeight(self, n: int) -> int:
        n = bin(n)
        count = 0
        for c in n:
            if c == '1':
                count += 1
        return count

 
#与2取余，如果是1，代表最后一位是1，count+1,然后就把末尾的0去掉
class Solution:
    def hammingWeight(self, n: int) -> int:
        count =0
        while n !=0:
            res = n %2
            if res ==1:
                count +=1
            n //= 2
        return count

    
#位运算
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n !=0:
            count += n&1
            n >>=1
        return count
        
            
```

#### [231. 2的幂](https://leetcode-cn.com/problems/power-of-two/)

```python
#位运算，是2的次幂意味着有且只有一个1，所以n不等于1，打掉最低为的就等于1
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        if n !=0 and n&(n-1) ==0:
            return True
        return False
```

#### [190. 颠倒二进制位](https://leetcode-cn.com/problems/reverse-bits/)

```python
-我们从右到左遍历输入整数的位字符串（即 n=n>>1）。要检索整数的最右边的位，我们应用与运算（n&1）。
-对于每个位，我们将其反转到正确的位置（即（n&1）<<power）。然后添加到最终结果。
-当 n==0 时，我们终止迭代。

class Solution:
    def reverseBits(self, n: int) -> int:
        res,power = 0,31
        while n:
            res += (n&1) << power
            n = n >>1    #n >>=1
            power -=1
        return res
```

#### [51. N 皇后](https://leetcode-cn.com/problems/n-queens/)

```python
1.基于集合的回溯
先附上回溯代码模板
result=[]
def backtrack(路径，选择列表):
    if 满足结束条件：
        result.append(路径)
        return
    for 选择 in 选择列表：
        做出选择
        递归执行backtrack
        撤销选择

N叉树的前序遍历模板
public static void preOrder(TreeNode tree) {
    if (tree == null)
        return;
    System.out.printf(tree.val + "");
    for (int i = 0; i <n ; i++) {
        preOrder("第i个子节点");
    }
}
        
 代码如下：
class Solution:
    def solveNQueens(self, n):
        def DFS(queens, xy_dif, xy_sum):   #queens，xy-dif,xy_sum是存列撇捺的数组，
            p = len(queens)                #如果使用位运算，位预算取代的就是这三个数组,用二进制位来表示相应的列撇捺有没有被占据掉
            if p==n:
                result.append(queens)
                return None
            for q in range(n):  #q.代表列，p代表行
                if q not in queens and p-q not in xy_dif and p+q not in xy_sum: 
                    DFS(queens+[q], xy_dif+[p-q], xy_sum+[p+q])  
        result = []
        DFS([],[],[])
        return [ ["."*i + "Q" + "."*(n-i-1) for i in sol] for sol in result]
    
2. A*

3.基于位运算的的回溯

def solveNQueens(self, n):
        res = []
        mask = (1 << n) - 1;
        queens = [-1] * n

        def dfs(res, ld = 0, row = 0, rd = 0, idx = 0):
            n = len(queens)
            if idx == n:
                res += ['.'*j +'Q'+ '.'*(n-j-1) for j in queens],
                return
            pos = mask & ~(ld | row | rd)
            while pos:
                p = pos & (~pos + 1)
                pos -= p
                queens[idx] = int(math.log(p, 2))
                dfs(res, (ld + p) << 1, row + p, (rd + p) >> 1, idx + 1)

        dfs(res)
        return res
```

#### [52. N皇后 II](https://leetcode-cn.com/problems/n-queens-ii/)

```python
#位运算是N皇后问题的终极解法
class Solution:
    def totalNQueens(self, n): 
        if n < 1: return [] 
        self.count = 0     #函数入口，就是最终我们需要的结果
        self.DFS(n, 0, 0, 0, 0)    
        return self.count

    def DFS(self, n, row, cols, pie, na):    #n代表皇后的个数，也就是递归有多少层，row表示当前层
        # recursion terminator               #cols,pie,na现在是三个数 
        if row >= n: 
            self.count += 1
            return
        bits = (~(cols | pie | na)) & ((1 << n) - 1) # 得到当前所有的空位（可以放皇后的位置，1代表可以放）
					      #(cols | pie | na)表示已经被皇后占据的格子，取反，代表哪些没有被占据的格子被赋予1
        while bits:       #(1 << n) - 1 代表将最高位至第n位置0
            p = bits & -bits # 取到最低位的1
            bits = bits & (bits - 1) # 表示在p位置上放入皇后
            self.DFS(n, row + 1, cols | p, (pie | p) << 1, (na | p) >> 1) 
            # 不需要revert cols, pie, na 的状态
```

#### [338. 比特位计数](https://leetcode-cn.com/problems/counting-bits/)

```python
DP+位运算
class Solution:
    def countBits(self, num: int) -> List[int]:
        bits = [0]*(num+1)
        
        for i in range(1,num+1):
            bits[i] += bits[i & (i-1)] +1  # i&(i-1)是清掉i中的最后一个1，然后再加回来就是i的对应的1数
        return bits
```

#### [146. LRU 缓存机制](https://leetcode-cn.com/problems/lru-cache/)

```python
1.使用结合了hash表和双向链表的数据结构OrderedDict
class LRUCache(object): 
	def __init__(self, capacity): 
		self.dic = collections.OrderedDict() 
		self.remain = capacity
	def get(self, key): 
		if key not in self.dic: 
			return -1
		v = self.dic.pop(key) 
		self.dic[key] = v # key as the newest one 
		return v 
	def put(self, key, value): 
		if key in self.dic: 
			self.dic.pop(key) 
		else: 
			if self.remain > 0: 
				self.remain -= 1
			else: # self.dic is full
				self.dic.popitem(last=False) 
		self.dic[key] = value


2.hashtable+双向链表
class ListNode:
    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.hashmap = {}
        # 新建两个节点 head 和 tail
        self.head = ListNode()
        self.tail = ListNode()
        # 初始化链表为 head <-> tail
        self.head.next = self.tail
        self.tail.prev = self.head

    # 因为get与put操作都可能需要将双向链表中的某个节点移到末尾，所以定义一个方法
    def move_node_to_tail(self, key):
            # 先将哈希表key指向的节点拎出来，为了简洁起名node
            #      hashmap[key]                               hashmap[key]
            #           |                                          |
            #           V              -->                         V
            # prev <-> node <-> next         pre <-> next   ...   node
            node = self.hashmap[key]
            node.prev.next = node.next
            node.next.prev = node.prev
            # 之后将node插入到尾节点前
            #                 hashmap[key]                 hashmap[key]
            #                      |                            |
            #                      V        -->                 V
            # prev <-> tail  ...  node                prev <-> node <-> tail
            node.prev = self.tail.prev
            node.next = self.tail
            self.tail.prev.next = node
            self.tail.prev = node

    def get(self, key: int) -> int:
        if key in self.hashmap:
            # 如果已经在链表中了久把它移到末尾（变成最新访问的）
            self.move_node_to_tail(key)
        res = self.hashmap.get(key, -1)
        if res == -1:
            return res
        else:
            return res.value

    def put(self, key: int, value: int) -> None:
        if key in self.hashmap:
            # 如果key本身已经在哈希表中了就不需要在链表中加入新的节点
            # 但是需要更新字典该值对应节点的value
            self.hashmap[key].value = value
            # 之后将该节点移到末尾
            self.move_node_to_tail(key)
        else:
            if len(self.hashmap) == self.capacity:
                # 去掉哈希表对应项
                self.hashmap.pop(self.head.next.key)
                # 去掉最久没有被访问过的节点，即头节点之后的节点
                self.head.next = self.head.next.next
                self.head.next.prev = self.head
            # 如果不在的话就插入到尾节点前
            new = ListNode(key, value)
            self.hashmap[key] = new
            new.prev = self.tail.prev
            new.next = self.tail
            self.tail.prev.next = new
            self.tail.prev = new

作者：liye-3
链接：https://leetcode-cn.com/problems/lru-cache/solution/shu-ju-jie-gou-fen-xi-python-ha-xi-shuang-xiang-li/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


class Node:
    def __init__(self, key, val, next=None, prev=None):
        self.key = key
        self.val = val
        self.next = next
        self.prev = prev


class DoubleList:
    def __init__(self):
        # self.head和self.tail都充当dummy节点（哨兵节点）
        self.head = Node(-1, -1)
        self.tail = Node(-1, -1)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0

    def addFirst(self, x):
        """在最前面加个节点x，注意语句顺序，很经典！"""
        x.next = self.head.next
        x.prev = self.head
        self.head.next.prev = x
        self.head.next = x
        self.size += 1

    def remove(self, x):
        """删除节点x，调用这个函数说明x一定存在"""
        x.prev.next = x.next  # 像一个顺时针
        x.next.prev = x.prev
        self.size -= 1

    def removeLast(self):
        """
        删除链表中最后一个节点，并返回该节点
        注意双向链表的删除时间复杂度是O(1)的，因为立刻能找到该删除节点的前驱
        """
        if self.size == 0:
            return None
        last_node = self.tail.prev
        self.remove(last_node)
        return last_node

    def getSize(self):
        return self.size



class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.map = {}
        self.cache = DoubleList()

    def get(self, key: int) -> int:
        if key not in self.map:
            return -1
        val = self.map[key].val
        self.put(key, val)
        return val

    def put(self, key: int, value: int) -> None:
        new_item = Node(key, value)
        if key in self.map:
            self.cache.remove(self.map[key])
            self.cache.addFirst(new_item)
            self.map[key] = new_item
        else:
            if self.capacity == self.cache.getSize():
                last_node = self.cache.removeLast()
                self.map.pop(last_node.key)
            self.cache.addFirst(new_item)
            self.map[key] = new_item
```

1122

```python
class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        res =[]
        dict1 ={}
        diff = []

        for num in arr2:     #初始化一个字典，见arr2中的数字做key
            if num not in dict1:
                dict1[num] =0

        for num in arr1:   #arr2中的词频
            if num not in dict1:
                diff.append(num)
            else:
                dict1[num] +=1

        diff.sort()

        for num in arr2:
            res.extend([num]*dict1[num])

        res.extend(diff)

        return res

```

#### [242. 有效的字母异位词](https://leetcode-cn.com/problems/valid-anagram/)

```python
1.调用系统的函数进行排序，这个过程使用的快速排序方式
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return collections.Counter(s) == collections.Counter(t) 	
    
2.手写一个计数排序

```

1244

```python

```

#### [56. 合并区间](https://leetcode-cn.com/problems/merge-intervals/)（高频题）

```python
https://leetcode-cn.com/problems/merge-intervals/solution/pai-xu-yi-ci-sao-miao-zhu-xing-jie-shi-hao-li-jie-/  ===参考连接
```

#### [493. 翻转对](https://leetcode-cn.com/problems/reverse-pairs/)

```python
1.暴力法(O（n**2）)
两层嵌套循环

2.merge-sort归并排序（O(nlogn)）
class Solution(object):
    def __init__(self):
        self.cnt = 0
    def reversePairs(self, nums):
        def msort(lst):
            # merge sort body
            L = len(lst)
            if L <= 1:                          # base case
                return lst
            else:                               # recursive case
                return merger(msort(lst[:int(L/2)]), msort(lst[int(L/2):]))
        def merger(left, right):
            # merger
            l, r = 0, 0                         # increase l and r iteratively
            while l < len(left) and r < len(right):
                if left[l] <= 2*right[r]:
                    l += 1
                else:
                    self.cnt += len(left)-l     # add here
                    r += 1
            return sorted(left+right)           # I can't avoid TLE without timsort...

        msort(nums)
        return self.cnt

3.树状数组（竞赛常用，类型并查集，代码很美，面试不用，使用二进制来罗列下标）(O(nlogn))


```

#### [989. 数组形式的整数加法](https://leetcode-cn.com/problems/add-to-array-form-of-integer/)

```python
方法1
class Solution:
    def addToArrayForm(self, A: List[int], K: int) -> List[int]:
        K =list(map(int,str(K)))   #整数转化为list,这个处理要记得
        i,j = len(A)-1,len(K)-1
        res = []
        carry = 0
        while i >=0 and j >=0:
            res.append(A[i]+K[j]+carry)
            res[-1],carry = res[-1]%10,res[-1]//10   #与10取余得个位数，与10取整得十位数
            i -=1
            j-=1
        while i>=0:
            res.append(A[i]+carry)
            res[-1],carry =res[-1]%10,res[-1]//10
            i -=1

        while j>=0:
            res.append(K[j]+carry)
            res[-1],carry =res[-1]%10,res[-1]//10
            j -=1


        if carry ==1:
            res.append(1)

        return res[::-1]
    
 方法2  
 class Solution:
    def addToArrayForm(self, A: List[int], K: int) -> List[int]:
        i = len(A) -1
        while K:
            A[i] +=K
            K,A[i] = A[i] //10,A[i]%10   #K代表高位，A[i]代表当前位
            i -=1

            if i<0 and K:    #这种情况是K的长度大于A的长度，将A前加1，继续循环
                A.insert(0,0)
                i =0
        return A
  
方法3：
class Solution:
    def addToArrayForm(self, A: List[int], K: int) -> List[int]:
        return map(int,str(int(''.join(map(str,A))) +K))

```

#### [1319. 连通网络的操作次数](https://leetcode-cn.com/problems/number-of-operations-to-make-network-connected/)

```python
#m个点连接最少需要m-1根线
#k个点连接最少需要k-1根线
#此问题就转化为求联通分量的个数
方法1：
class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        if len(connections) <n-1:
            return -1
        
        egets = collections.defaultdict(list)
        for x,y in connections:
            egets[x].append(y)
            egets[y].append(x)   #将每个点的联通的点以set的形式表示

        seen  =set()    #表示访问过的点
        def dfs(u):    #dfs找联通分量
            seen.add(u)
            for v in egets[u]:
                if v not in seen:
                    dfs(v)
        ans = 0

        for i in range(n):     #遍历所有的点
            if  i not in seen:
                dfs(i)
                ans +=1    #每次dfs之后，联通分量都+1
                
        return ans-1   #最后需要移动的线的个数就是联通分量数减一


方法2：
我们可以使用并查集来得到图中的连通分量数。
class UnionFind:
    def __init__(self,n):
        self.parent  =list(range(n))
        self.n =n 
        self.setCount = n
        self.size = [1]*n
    def findset(self,x):
        if self.parent[x] == x:
            return x
        self.parent[x] = self.findset(self.parent[x])
        return self.parent[x]
    
    def unite(self,x,y):
        x, y = self.findset(x),self.findset(y)
        if x == y:
            return False
        if self.size[x] < self.size[y]:
            x,y = y,x
        self.parent[y] =x
        self.size[x] += self.size[y]
        self.setCount -=1
        return True
class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        if len(connections) <n-1:
            return -1
        uf = UnionFind(n)
        for x, y in connections:
            uf.unite(x, y)	
        
        return uf.setCount - 1
并查集本身就是用来维护连通性的数据结构。如果其包含 n 个节点，那么初始时连通分量数为 n，每成功进行一次合并操作，连通分量数就会减少 1

```

