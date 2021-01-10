#### [26. 删除排序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0
        i =0    #i来做新数组中的下标
        for data in nums[1:] :   
            if nums[i] != data:
                i +=1
                nums[i] = data
        return i+1
    
    
##上面这个写成双指针法可能更好理解：
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0
        i =0
        for j in range(1,len(nums)):
            if nums[i] != nums[j]:
                i +=1
                nums[i] = nums[j]
        return i+1
```

#### [20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

```python
#预先定义一个字典和长度为1的stack,对输入字符串做循环，如果是值是左括号，做入栈操作，否则做出栈操作，将出栈元素在字典中的value与当前元素做对比，如果相等不做任何操作，如果不等，返回False,这样循环结束的时候再判断一下栈中元素是否都释放了。
#方法1
class Solution:
    def isValid(self, s: str) -> bool:
        dic = {'{': '}',  '[': ']', '(': ')', '?': '?'}
        stack = ['?']
        for c in s:
            if c in dic: 
                stack.append(c)
            elif dic[stack.pop()] != c: 
                return False 
        return len(stack) == 1
#方法2    
class Solution:
    def isValid(self, s: str) -> bool:
        dic = {')':'(',']':'[','}':'{'}   
        stack = []
        for i in s:      #i代表右括号，dic[i]代表左括号，遇到右括号要与栈顶元素相比，遇到左括号就追加到栈中
            if stack and i in dic:
                if stack[-1] == dic[i]: stack.pop()
                else: return False
            else: stack.append(i)
            
        return not stack

 #方法3   
class Solution:
    def isValid(self, s: str) -> bool:
        if len(s) <2:
            return False
        dic = {')':'(',']':'[','}':'{'}
        stack =[]
        for i in s:
            if stack and i in dic:
                if stack[-1] == dic[i]:
                    stack.pop()
                else:
                    return False
            else:
                stack.append(i)
        return True if not stack else False   

```

#### [21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1:
            return l2
        if not l2:
            return l1      #递归退出条件
        if l1.val <= l2.val:     #使用递归计算
            l1.next = self.mergeTwoLists(l1.next,l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l2.next,l1)
            return l2

```

#### [42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)



```python
有点难
```

#### [11. 盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water/)

```python
#双指针法
class Solution:
    def maxArea(self, height: List[int]) -> int:
        i,j,res = 0,len(height)-1,0
        while i < j:
            if height[i] <height[j]:
                res = max(res,height[i]*(j-i))
                i +=1
            else:
                res = max(res,height[j]*(j-i))
                j -=1
        return res
```

#### [189. 旋转数组](https://leetcode-cn.com/problems/rotate-array/)

```python
#插入法128ms：将数组最后一个元素pop出来，插入位置0.
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k %= n
        for _ in range(k):
            nums.insert(0, nums.pop())
#拼接法：
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k %= n
        nums[:] = nums[-k:] + nums[:-k]

#三次翻转法
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k %= n
        nums[:] = nums[::-1]
        nums[:k] = nums[:k][::-1]
        print(nums)
        nums[k:] = nums[k:][::-1]
        print(nums)

            

```

#### [88. 合并两个有序数组](https://leetcode-cn.com/problems/merge-sorted-array/)

```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        p1 = m-1
        p2 = n-1
        p  = m+n-1
        while p1>= 0 and p2 >=0:
            if nums1[p1] < nums2[p2]:
                nums1[p] = nums2[p2]
                p2 -= 1
            else:
                nums1[p] = nums1[p1]
                p1 -= 1
            p -=1
        nums1[:p2+1] = nums2[:p2+1]
```

#### [283. 移动零](https://leetcode-cn.com/problems/move-zeroes/)

```python
#将非零的移动到前面，后面的全部置零
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        j = 0
        for i in range(len(nums)):
            if nums[i] !=0:
                nums[j] = nums[i]
                j +=1
        while j<=i:
            nums[j] =0
            j +=1
        return nums


#双指针法，j代表数组索引（指针1），i代表j前零的个数，j-i（第二个指针）就是j前的第一个零，然后将两个值交换
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        i=0
        for j in range(len(nums)):
            if nums[j] == 0:
                i +=1
            else:
                nums[j],nums[j-i] = nums[j-i],nums[j]
        return nums
        
```

#### [70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)

```爬楼梯
#动态规划
class Solution:
    def climbStairs(self, n: int) -> int:
        i,j,k =0,0,1
        for m in range(1,n+1):
            i =j
            j =k
            k =i+j
        return k

```

#### [15. 三数之和](https://leetcode-cn.com/problems/3sum/)

```python
#排序+双指针
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        
        ans =[]
        nums.sort()
        n = len(nums)
        if not nums or n<3:
            return []
        for i in range(n):
            if i>0 and nums[i]==nums[i-1]:  #去重，注意1>0，是为了避免[0,0,0]这个特殊情况
                continue
            j =i+1   #左指针
            k =n-1   #右指针
            while j<k:
                if nums[i]+nums[j]+nums[k] == 0:
                    ans.append([nums[i],nums[j],nums[k]])
                    while j<k and nums[j] ==nums[j+1]:  #去重
                        j +=1
                    while j<k and nums[k] ==nums[k-1]:  #去重
                        k -=1
                    j +=1
                    k -=1
                elif nums[i]+nums[j]+nums[k] >0:
                    k -=1
                else:
                    j +=1
        return ans
```

#### [206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

```python
#迭代法
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        cul,pre = None,head
        while pre:
            t = pre.next
            pre.next = cul
            cul = pre
            pre = t
        return cul
 
或者双指针法（一般链表问题都可以使用双指针法来解决）：
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        pre,cul,nextnode = None,head,head.next
        while cul != None:
            nextnode = cul.next //nextNode 指向下一个节点,保存当前节点后面的链表。
            cul.next = pre     //将当前节点next域指向前一个节点   null<-1<-2<-3<-4
            pre = cul          //preNode 指针向后移动。preNode指向当前节点。
            cul = nextnode     //curNode指针向后移动。下一个节点变成当前节点
        return pre
        
        
#递归法
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        cul = self.reverseList(head.next)
        head.next.next = head
        head.next =None    #.next可以理解为。。。指针，比如：head.next=None可以理解为head的指针指向None
        return cul         #head.next.next=head可以理解为head.next的指针指向head
```

