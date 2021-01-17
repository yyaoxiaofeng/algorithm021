#### [62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)

```python
#使用函数（排列组合）：对于一个m*n的矩阵，左上角移动到右下角有m+n-2步，向下m-1,向右n-1,这个题就是从m+n-2步中选择min(m-1,n-1) O（m）或者O(n)
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        return comb(m+n-2,m-1)
     
#动态规划
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1]*n]+[[1]+[0]*(n-1) for _ in range(m-1)]   #将第一行和第一列都置1，
        for i in range(1,m):                               #到达第一行和第一列的方法只有一种                         
            for j in range(1,n):
                dp[i][j] = dp[i-1][j]+dp[i][j-1]      
        return dp[-1][-1]

    
 或者：
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1]*n]*m    #只是二维数组初始化有点改变
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = dp[i-1][j]+dp[i][j-1]
        return dp[-1][-1]

```

#### [63. 不同路径 II](https://leetcode-cn.com/problems/unique-paths-ii/)

```python
#动态规划
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        height,width = len(obstacleGrid),len(obstacleGrid[0])
        for i in range(height):
            for j in range(width):
                if obstacleGrid[i][j] ==1:   #对故障点的处理
                    obstacleGrid[i][j] =0
                else:
                    if i == j  ==0:    #对[0][0]点置1
                        obstacleGrid[i][j] = 1
                    else:
                        a = obstacleGrid[i-1][j] if i !=0 else 0   #这里的处理非常巧妙，没有对第一行或者第一列进行特殊处理
                        b = obstacleGrid[i][j-1] if j !=0 else 0
                        obstacleGrid[i][j] = a+b 
        return obstacleGrid[-1][-1]
```

#### [1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)

```python
#动规划
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m = len(text1)
        n = len(text2)
        dp= [[0]*(n+1) for _ in range(m+1)]   #因为dp从1开始，最后dp[1]对应str[0],所以dp要多一行一列
        for i in range(1,m+1):     #因为涉及到dp[i-1],所以i从1开始，
            for j in range(1,n+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] =1 +dp[i-1][j-1]
                else:
                    dp[i][j] = max(dp[i-1][j],dp[i][j-1])
        return dp[-1][-1]	
```

#### [70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        a,b,c =0,1,0
        for i in range(n):
            c = a+b
            a = b
            b = c
        return c
            
```

#### [120. 三角形最小路径和](https://leetcode-cn.com/problems/triangle/)

```python
#动态规划
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        m = len(triangle)
        
        dp = [[0]*(m+1)]*(m+1)   #相比tringle,dp多开了一行一列，这样就防止了dp[i+1][j+1]越界
        for i in range(m-1,-1,-1):
            for j in range(i+1):
                dp[i][j] = min(dp[i+1][j],dp[i+1][j+1])+triangle[i][j]
        return dp[0][0]
```

#### [53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

```python
#动态规划;
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        ans= nums[0]
        sum =0
        for num in nums:
            sum = max(sum+num,num)
            ans = max(ans,sum)
        return ans
#另外一种写法
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        ans = nums[0]    #最终结果
        sum =0           #分段求sum
        for num in nums:
            if sum>=0:
                sum += num   #如果前一步sum大于0，就继续相加
            else:
                sum =num     #如果sum小于0，就抛弃sum，重新开一个模块，将当前值做为这个模块的开头
            ans = max(ans,sum)    #sum有增减波动，ans只取这个过程中的最大值，可以理解sum是一个临时和
        return ans

```

#### [152. 乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/)

```python
#动态规划
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        imin,imax =1,1
        max1 = -inf
        for num in nums:
            if num<0:   #与最大自序和不同的是，需要维护一个最小值，遇到负数就交换当前最大最小值
                tem =imin
                imin = imax
                imax =tem
            imin=min(imin*num,num)
            imax = max(imax*num,num)
            max1 = max(max1,imax)
        return max1
    
    
#·由于存在负数，那么会导致最大的变最小的，最小的变最大的。因此还需要维护当前最小值imin，imin = min(imin * nums[i], nums[i])
#·当负数出现时则imax与imin进行交换再进行下一步计算


```

#### [322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)

```python
#动态规划
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float(inf)]*(amount+1)
        dp[0] =0
        for i in range(1,amount+1):
            for con in coins:
                if i >=con:
                    dp[i]=min(dp[i],dp[i-con]+1)    
        return dp[-1] if dp[-1] != float(inf) else -1
```

#### [198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0
        dp =[0]*(len(nums))
        if len(nums) ==1:
            return nums[0]
        dp[0]=nums[0]
        dp[1] =max(nums[0],nums[1])
        for i in range(2,len(nums)):
            dp[i] = max(dp[i-2]+nums[i],dp[i-1])
        return dp[-1]

#优化为两个变量
class Solution:
    def rob(self, nums: List[int]) -> int:
        pre,now = 0,0
        for i in nums:
            pre,now = now,max(pre+i,now)
        return now
#有两种方案选择方案：
当房间数k>=3时：
1，选择了第k次偷窃，这个时候最大的的钱为第k-2次的加上第k次的
2.选择了第k-1次偷窃，这个时候最大的钱数为第k-1次的钱数
然后再在以上两种方案中选择更大的
当房间数为1时，直接选择这个房间的钱
当房间数为2时，选择两间房子里更大的钱，这里房间数为2时的结果正好可以做为k>=3时初始化的内容

```

#### [213. 打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/)

```python
#将上一道题分成两步：选择nums[0],不选择nums[0],然后将两者结果再求最大值
class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0
        if len(nums) ==1:
            return nums[0]
        if len(nums) ==2:
            return  max(nums[0],nums[1])   #对原数组长度为2的也要做特殊处理
        #取第一个元素
        dp1 =[0]*(len(nums)-1)
        
        dp1[0]=nums[0]
        dp1[1] =max(nums[0],nums[1])
        for i in range(2,len(nums)-1):
            dp1[i] = max(dp1[i-2]+nums[i],dp1[i-1])
        first= dp1[-1]

        #不取第一个元素
        dp2 =[0]*(len(nums)-1)
        
        dp2[0]=nums[1]
        dp2[1] =max(nums[1],nums[2])
        for i in range(2,len(nums)-1):
            dp2[i] = max(dp2[i-2]+nums[i+1],dp2[i-1])  #注意num[k]的索引都需要+1
            print(dp2[i])
        second= dp2[-1]
        print(first,second)
        return(max(first,second))
    
 #双指针法：
class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0
        if len(nums) ==1:
            return nums[0]
        #取第一个元素
        pre1,now1 = 0,0
        for i in range(len(nums)-1):
            pre1,now1 = now1,max(pre1+nums[i],now1)
        ans1 = now1 
        

        #不去第一个元素
        pre2,now2 = 0,0
        for i in range(1,len(nums)):
            pre2,now2 = now2,max(pre2+nums[i],now2)
        ans2 = now2

        return max(ans1,ans2) 

```

#### [121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

```python
#一次遍历（其实是动态规划的优化）
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        
        min_price = float('inf')
        max_pro  = 0
        
        for price in prices:
            min_price = min(min_price,price)
            max_pro = max(max_pro,price-min_price)
        return max_pro
      
#动态规划
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        if n <=1:
            return 0
        
        min_price = float('inf')
        dp = [0]*(len(prices))
        
        for i  in range(len(prices)):
            min_price = min(min_price,prices[i])
            dp[i] = max(dp[i-1],prices[i]-min_price)
        return dp[-1]
```

#### [122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

```python
#贪心算法
只要后一天比前一天大，就卖掉，否则不做任何处理，保证能拿到所有的利润
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        sum = 0
        for i in range(1,len(prices)):
            if prices[i] >prices[i-1]:
                sum +=(prices[i]-prices[i-1])
        return sum
        
        
```

123

```

```

309

```

```

188

```

```

714

```

```

279

```


```

72

```

```

55

```


```

45

```

```

980

```

```

#### [518. 零钱兑换 II](https://leetcode-cn.com/problems/coin-change-2/)

```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [0]*(amount+1)   #初始化数组为0
        dp[0] =1
        for con in coins:   #枚举硬币
            for i in range(1,amount+1):   #枚举金额
                if i <con:
                    continue     #金额必须大于corn
                dp[i] = dp[i]+dp[i-con]   
                
        return dp[-1]

  #这是一个组合问题，需要先枚举硬币，起到限制硬币顺序的作用，注意内外层循环不要搞反
```

#### [64. 最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/)

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if i ==0 and j ==0:
                    grid[i][j] =grid[i][j]
                elif i == 0 and j !=0:
                    grid[i][j] =grid[i][j-1]+grid[i][j]
                elif i !=0 and j == 0:
                    grid[i][j] = grid[i-1][j]+grid[i][j]
                else:
                    grid[i][j] = min(grid[i-1][j],grid[i][j-1])+grid[i][j]
        return grid[-1][-1]
```

#### [91. 解码方法](https://leetcode-cn.com/problems/decode-ways/)

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        
        if s.startswith('0'):
            return 0
        n = len(s)
        dp = [1]*(n+1)     #因为有i-2,所以这里i+1,dp[0]dp[1]都置为1
        for i in range(2,n+1):
            if s[i-1] == '0' and s[i-2] not in '12':  #当s[i-1]出现前导‘0’，无法编码
                return 0
            if s[i-2:i] in ['10','20']:      #当出现10.20，s[i-1]这个位置的编码方式和s[i-2]一样
                dp[i] =dp[i-2]               #s[i-1]对应dp[i],也就是只能组合编码
            elif '10'< s[i-2:i] <='26':      #等于s[i-1]自身的编码方式+和s[i-2]组合后的编码方式
                dp[i] = dp[i-1]+dp[i-2]
            else:
                dp[i] = dp[i-1]              #01~09，>26的情况，s[i-1]位置上的数只能自己编码
            print(dp[n])
        return dp[n]

```

221

```

```

621

```

```

647

```


```

32

```

```

363

```


```

403

```

```

410

```

```

552

```

```

76

```

```

#### [312. 戳气球](https://leetcode-cn.com/problems/burst-balloons/)

```python
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        nums.insert(0,1)
        nums.insert(len(nums),1)
        n = len(nums)
        stort = [[0]*(n) for _ in range(n)]    
        def range_best(i,j):      #先计算小区间i-j之间的最大值
            m = 0
            for k in range(i+1,j):   #k是i-j之间最后戳破的一个气球
                left = stort[i][k]
                right = stort[k][j]
                a = left+ nums[i]*nums[k]*nums[j] +right
                if a >m:
                    m =a
            stort[i][j] =m     #求出i-j之间的最大值，属于局部最大

        for m in range(2,n): #m是区间长度，从3到n
            for i in range(n-m):
                range_best(i,i+m) #遍历i的头部
        return stort[0][n-1]
```

5641

```

```

605

```

```

239

```

```

5642

```

```

5643

```

```

5644

```

```

