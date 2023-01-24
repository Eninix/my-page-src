---
title: Numpy的使用
tags:
  - Python
  - Numpy
  - AI
categories: 编程语言
abbrlink: '63916680'
date: 2023-01-16 03:14:22
---


<center>Python的Numpy库的使用。拥有数据处理、科学计算、矩阵计算、深度学习等用途。</center>
<center>
总结自B站的教学视频 BV1Ex411L7oT
<!-- <a herf="https://www.bilibili.com/video/BV1Ex411L7oT">BV1Ex411L7oT</a> -->
</center>


<!--more-->

# 安装

```shell
# 可根据自身情况选择以下命令
pip install numpy
pip3 install numpy
sudo pip install numpy
sudo pip3 install numpy
```

# Numpy属性

```python
import numpy as np
array = np.array([[1, 2, 3], [4, 5, 6]])
print(array)
print("number of dim: ", array.ndim)  # 2
print("shape: ", array.shape)  # (2,3)
print("size: ", array.size)  # 6
```

![](https://cdn.jsdelivr.net/gh/Eninix/the-bed@main/20230116033857.png)

# array的创建

## 数据类型的指定

```python
# 矩阵的普通创建
a = np.array([[2, 23, 4], [2, 32, 4]])
print(a)
# [[ 2 23  4]
#  [ 2 32  4]]

# 指定矩阵中元素的数据类型
a = np.array([2, 23, 4], dtype=np.int64)
print(a.dtype)  # int64
a = np.array([2, 23, 4], dtype=np.float64)
print(a.dtype)  # float64
a = np.array([2, 23, 4], dtype=np.double)
print(a.dtype)  # float64
```

## 零矩阵 全一矩阵 空矩阵

```python
# 零矩阵 参数 (x,y) 为 x行y列
a = np.zeros((3,4))
print(a)

# 全一矩阵 参数 (x,y) 为 x行y列
a = np.ones((3,4))
print(a)

# 空矩阵 参数 (x,y) 为 x行y列
a1 = np.empty((4,4))
print(a1)
```

## 数列的生成及reshape

```python
# 随机生成0到1之间的数，2行4列
a=np.random.random((2,4))
print(a)

# 生成范围为[10,20)的数列，步长为2
a = np.arange(10,20,2)
print(a) # [10 12 14 16 18]

# 生成范围为[0,12)的数列，步长为1，并塑造为3行4列的矩阵
a = np.arange(12).reshape((3,4))
print(a)
#[[ 0  1  2  3]
# [ 4  5  6  7]
# [ 8  9 10 11]]

# 生成范围为[1,10]的数列，等分为10段
a = np.linspace(1,10,20)
print(a)
#[ 1.          1.47368421  1.94736842  2.42105263  2.89473684  3.36842105
#  3.84210526  4.31578947  4.78947368  5.26315789  5.73684211  6.21052632
#  6.68421053  7.15789474  7.63157895  8.10526316  8.57894737  9.05263158
#  9.52631579 10.        ]
```


# 基础运算1

```python
import numpy as np

a = np.arange(10, 50, 10)
b = np.arange(4)
print(a, b)  # [10 20 30 40] [0 1 2 3]

c = a + b
print(c)  # [10 21 32 43]

c = a - b
print(c)  # [10 19 28 37]

print(a**2)  # [ 100  400  900 1600]

c = 10 * a
print(c)  # [100 200 300 400]
print(c < 111)  # [ True False False False]
print(c == 200)  # [ False True False False]

c = np.sin(a)
print(c)  # [-0.54402111  0.91294525 -0.98803162  0.74511316]


# ------

a = np.array([[1, 2], [2, 3]])
b = np.arange(4).reshape((2, 2))
print(a, "\n", b)
# [[1 2]
#  [2 3]] 
#  [[0 1]
#  [2 3]]

print(a * b)
# [[0 2]
#  [4 9]]

print(a @ b)
# print(np.dot(a, b))
# print(a.dot(b))
# [[ 4  7]
#  [ 6 11]]


#-----
a=np.random.random((2,4))
print(a)
print(np.sum(a))
print(np.max(a))
print(np.min(a))
# axis=0,1 为 行，列
print(np.sum(a,axis=1))
print(np.max(a,axis=0))
print(np.min(a,axis=1))

```


# 基础运算2

```python
import numpy as np

a = np.array([1, 8, 15, 6, 55, 99, 125, 10, 23, 15, 89, 52]).reshape((3, 4))
print(a)

# 输出最大最小值的索引
print(np.argmin(a))  # 0
print(np.argmax(a))  # 6

# 平均值
print(np.average(a))  # 41.5
print(a.mean())  # 41.5
# 中位数
print(np.median(a))  # 19

# 前N项和数列; 累加数列
print(np.cumsum(a))  # [  1   9  24  30  85 184 309 319 342 357 446 498]

# 累差数列
print(np.diff(a))
# [[   7    7   -9]
#  [  44   26 -115]
#  [  -8   74  -37]]

# 找出非零元素
print(np.nonzero(a))
# (array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]), array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]))
# 获得非零元素的行索引矩阵和列索引矩阵

# 排序
print(np.sort(a))  # 逐行排序

# 转置矩阵
print(np.transpose(a))
print(a.T)

# clip 限制矩阵的范围 使不超出min max
min = 5
max = 50
print(np.clip(a, min, max))
# [[ 5  8 15  6]
#  [50 50 50 10]
#  [23 15 50 50]]
```


# Numpy的索引

```python
import numpy as np

a = np.array([1, 8, 15, 6, 55, 99, 125, 10, 23, 15, 89, 52]).reshape((3, 4))
print(a)
# [[  1   8  15   6]
#  [ 55  99 125  10]
#  [ 23  15  89  52]]

print(a[2])  # [23 15 89 52]
print(a[:, 1])  # [ 8 99 15]
print(a[2][1])  # 15
print(a[2, 1])  # 15
print(a[2, 1:3])  # [15 89]


# 迭代 iteration
for row in a:  # 迭代行
    print(row)
# [15 89]
# [ 1  8 15  6]
# [ 55  99 125  10]
# [23 15 89 52]

for column in a.T:  # 通过迭代转置矩阵迭代列
    print(column)
# [ 1 55 23]
# [ 8 99 15]
# [ 15 125  89]
# [ 6 10 52]

# 扁平化为一行,并遍历
for item in a.flat:
    print(item)

```


# array的合并

```python
import numpy as np

a = np.array([1, 1, 1])
b = np.array([2, 2, 2])
print(a,b)

# 上下合并
c = np.vstack((a, b))
print(c)
print(a.shape, c.shape)  # (3,) (2, 3)
# 左右合并
c = np.hstack((a, b))
print(a.shape, c.shape)  # (3,) (6,)

# 将序列变为列矩阵 增维
a = a[:, np.newaxis]
b = b[:, np.newaxis]
c = np.hstack((a, b))
print(a)
# [[1]
#  [1]
#  [1]]
print(a.shape, c.shape)  # (3,1) (3,2)

# 合并 axis 0 1 为 纵向 横向
c = np.concatenate((a, b, b, a), axis=0)
print(c)
c = np.concatenate((a, b, b, a), axis=1)
print(c)
```


# array的分割

```python
import numpy as np

a = np.arange(12).reshape((3,4))
print(a)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# np.split(array, section, axis)

print(np.split(a,3,axis=0))
print(np.vsplit(a,3))
# [array([[0, 1, 2,  3]]), 
#  array([[4, 5, 6,  7]]), 
#  array([[8, 9, 10, 11]])]

print(np.split(a,2,axis=1))
print(np.hsplit(a,2))
# [array([[0, 1],
#        [4, 5],
#        [8, 9]]), 
#  array([[ 2,  3],
#        [ 6,  7],
#        [10, 11]])]

# 不等份分割
print(np.array_split(a,3,axis=1))
# [array([[0, 1],
#        [4, 5],
#        [8, 9]]), 
#  array([[ 2],
#        [ 6],
#        [10]]),
#  array([[ 3],
#        [ 7],
#        [11]])]

```


# array的copy

![](https://cdn.jsdelivr.net/gh/Eninix/the-bed@main/20230116053906.png)


# array的deep copy

![](https://cdn.jsdelivr.net/gh/Eninix/the-bed@main/20230116054035.png)

