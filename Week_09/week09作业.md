#### [70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)

```python
#方法1：傻递归，时间是O(2**n),超出时间限制
class Solution:
    def climbStairs(self, n: int) -> int:
        def f(n):
            if n<=1:return 1
            return(f(n-1)+f(n-2))
        return f(n)
 #方法2：分治+记忆化缓存（动态规划）时间是O(n),空间是O(n)
 #用一个mem来存访问过的数，如果没有访问就访问一次存起来
 #如果访问过了就直接返回mem[n]
 class Solution:
    def climbStairs(self, n: int) -> int:
        mem = [1]*(n+1)
        def f(n):
            if n<=1:return 1
            if n not in mem:
                mem[n] = f(n-1)+f(n-2)
            return mem[n]    
        return f(n)
 #方法3：递归转化为顺推,时间O(n),空间O（n）
 class Solution:
    def climbStairs(self, n: int) -> int:
        def f(n):
            dp = [1]*(n+1)
            for i in range(2,n+1):
                dp[i] = dp[i-1]+dp[i-2]
            return dp[n]
        return f(n)
    
#方法4：优化空间到O(1),不使用数组存储，只使用两个变量不停的往前迭代
class Solution:
    def climbStairs(self, n: int) -> int:
        x,y =1,1
        for i in range(2,n+1):
            x,y = y,x+y
        return y
        
```

#### [62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)

```python
方法1：傻递归,O（2**（m+n）），超出时间限制
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        def f(x,y):
            if x<=0 or y <=0:return 0
            if x == 1 and y == 1:return 1
            return f(x-1,y)+f(x,y-1)
        return f(m,n)
方法2：加入缓存的递归 O（nm）超出时间限制
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        mem = [[0]*(n+1) for _ in range(m+1)]
        def f(x,y):
            if x<=0 or y <=0:return 0
            if x == 1 and y == 1:return 1
            if (x,y) not in mem:
                mem[x][y]=f(x-1,y)+f(x,y-1)
            return mem[x][y]
        return f(m,n)

方法3：动态规划
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[0]*n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = dp[i-1][j]+dp[i][j-1]
        return dp[-1][-1]

更常用下面这种写法：
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[0]*(n+1) for _ in range(m+1)]
        
        for i in range(1,m+1):
            for j in range(1,n+1):
                if i == 1 and j == 1:
                    dp[i][j] = 1
                else:
                    dp[i][j] = dp[i-1][j]+dp[i][j-1]
                
        return dp[-1][-1]
```

#### [198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

```python
方法1：一维DP
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) ==0:
            return 0
        if len(nums) ==1:
            return nums[0]
        
        dp = [0]*len(nums)
        dp[0]= nums[0]
        print(dp[0])
        dp[1] =max(nums[0],nums[1]) 
        for i in range(2,len(nums)):
            dp[i] = max(dp[i-1],dp[i-2]+nums[i])
            print(dp[i])
        return dp[-1]
    
方法2：二维DP
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) ==0:
            return 0
        if len(nums) ==1:
            return nums[0]
        dp=[[0]*2 for _ in range(len(nums))]
        dp[0][0] = 0
        dp[0][1] = nums[0]
        for i in range(1,len(nums)):
            dp[i][0] = max(dp[i-1][0],dp[i-1][1])
            dp[i][1] = dp[i-1][0]+nums[i]
        return max(dp[-1][0],dp[-1][1])
```

64

```python

```

121

```python

```

746

```python

```

#### [72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)

```python
方法一：BFS

方法二：two-ended BFS

方法三：DP
定义状态方程dp[i][j]  word1.substr(0-i)与word2.substr(0-j)之间的编辑距离

思考过程：

第一种情况，两个都为空，编辑距离为0
w1 = ''
w2 = ''
第二种情况，其中一个为空，编辑距离为非空字符串的长度
w1 = ''
w2 = '....w'
第三种情况，i,j位置相同
w1 = '....f'
w2 = '...f'
edit_dist(i,j) = edit_dist(i-1,j-1)
第四种情况，i,j位置不相等
edit_dist(i,j) = min(edit_dist(i-1,j-1)+1,edit_dist(i-1,j)+1,edit(i,j-1)+1)

代码：
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n1 = len(word1)
        n2 = len(word2)
        dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
        # 第一行
        for j in range(1, n2 + 1):
            dp[0][j] = dp[0][j-1] + 1
        # 第一列
        for i in range(1, n1 + 1):
            dp[i][0] = dp[i-1][0] + 1
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1] ) + 1
        #print(dp)      
        return dp[-1][-1]

```

709

```python

```

58

```python


```

771

```python

```

#### [387. 字符串中的第一个唯一字符](https://leetcode-cn.com/problems/first-unique-character-in-a-string/)

```python
方法1：blute-force
i枚举所有字符，j枚举i后的所有字符，找重复，O(n**2)

方法2：map(hashmap O(n),treemap O(logn))
总体时间复杂度O(n),或者O(nlogn)

方法3：hash table(使用字母对应的下标来统计，计算一下字母出现了多少次)

代码：
class Solution:
    def firstUniqChar(self, s: str) -> int:
        posion = {}
        q = collections.deque()
        n = len(s)
        for i,ch in enumerate(s):
            if ch not in posion:
                posion[ch] =i
                q.append((s[i],i))
            else:
                posion[ch] = -1
                while q and posion[q[0][0]] == -1:
                    q.popleft()
        return -1 if not q else q[0][1]
```

#### [8. 字符串转换整数 (atoi)](https://leetcode-cn.com/problems/string-to-integer-atoi/)

```python
方法1：高级语言java,python都提供有专门的转换函数

方法2：考察代码基本功
class Solution:
    def myAtoi(self, s: str) -> int:

        ls =list(s.strip())
        if len(ls) ==0:
            return 0 
        sign = -1 if ls[0] == '-' else 1
        if ls[0] in ['-','+']:del ls[0]
        ret,i =0,0
        while i<len(ls) and ls[i].isdigit():
            ret = ret*10 +ord(ls[i]) -ord('0')
            i+=1
        return max(-2**31,min(sign*ret,2**31-1))
```

#### [14. 最长公共前缀](https://leetcode-cn.com/problems/longest-common-prefix/)

```python
方法1：暴力法
方法2：i遍历第一个字符串的所有字符，j遍历所有字符串
方法3：Trie
```

#### [344. 反转字符串](https://leetcode-cn.com/problems/reverse-string/)

```python
方法1：双指针
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        i,j = 0,len(s)-1
        while i<j:
            s[i],s[j] = s[j],s[i]
            i +=1
            j -=1
        return s
```

541

```python

```

#### [151. 翻转字符串里的单词](https://leetcode-cn.com/problems/reverse-words-in-a-string/)

```python
方法1：split,reverse,join

方法2：reverse整个字符串，然后再reverse每个单词，两个单独的循环，时间复杂度O（n）

```

557

```python

```

917

```python

```

242

```python

```

49

```python

```

#### [438. 找到字符串中所有字母异位词](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/) （重要）

```python
类似于滑动窗口
```

125

```python


```

680

```python

```

#### [5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

```python
方法1：暴力法，i，j代表字符串起始和结束的索引，然后来判断整个字符串是否回文，O（n**3）

方法2：枚举中间字符，从中间向两边扩张O（n**2）

方法3：动态规划,i,j代表字符串的起始和终止的位置，dp[i][j]代表从i到j是回文字符串

dp[i][j] = dp[i-1][j-1] && s[i] ==s[j],
```

#### [1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)

```python
子序列和字串的区别：子序列是可以有间隔的，字串是不能有间隔的。
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m = len(text1)
        n = len(text2)
        dp= [[0]*(n+1) for _ in range(m+1)]
        for i in range(1,m+1):
            for j in range(1,n+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] +1
                else:
                    dp[i][j] = max(dp[i-1][j],dp[i][j-1])
        return dp[-1][-1]
```

#### [10. 正则表达式匹配](https://leetcode-cn.com/problems/regular-expression-matching/)(重点)

```python
参考labuladong的算法小抄解法
```

44

```python

```

#### [115. 不同的子序列](https://leetcode-cn.com/problems/distinct-subsequences/)

```python
方法1.暴力递归
方法2.动态规划

dp[li][i]代表T前i字符串可以由s前j字符串组成最多个数。所以动态方程:
当Si]= Ti], dp[i]们] = dp[i-1][-1]+dp[i]U-1]
当S0] !=T[i], dp[i][i] = dp[i]j-1]
```

205

```python

```

300

```python

```

91

```python

```

32

```python

```

818

```python

```



**补充题目：**

#### [1128. 等价多米诺骨牌对的数量](https://leetcode-cn.com/problems/number-of-equivalent-domino-pairs/)

```
方法1：因为纸牌数字小于10，所以大数乘以10+小数如果相等，则两个一样，并且两者相加不会超过100，所以可以使用一个长度为100的数组来表示val出现的次数，出现的次数只可能是0，1，统计1的个数就是最终的结果
class Solution:
    def numEquivDominoPairs(self, dominoes: List[List[int]]) -> int:
        num = [0]*100
        count =0
        for x,y in dominoes:
            val = (x*10+y if x>y else y*10+x)   
            count += num[val]
            num[val] +=1
        return count
```

