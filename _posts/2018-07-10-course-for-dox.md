---
layout:     post
title:     python course for dox
subtitle:   
date:       2018-07-10
author:     paradox
header-img: img/post-bg-coffee.jpeg
catalog: true
mathjax: true
tags:
---

# Project 2

$paradox\ 2018-7-10$

## 保留字符

保留字符就是python中不能被当作变量的字符，因为这些字符在python中已经被赋予了特殊的意义，例如：`and`是和的意思，`def`是定义函数的意思。

## 注释

#### 单行缩进

```python
a = 3 # 对a这个变量进行赋值
```

`#`后面的就是注释内容，注释内容不会对代码造成任何影像，完全是为了是的你的程序有更好的可读性，更容易让别人和自己理解，这点对于初学者来说，尤其重要。

#### 多行注释

```python
'''
paradox is pretty good.
how are you?
why not?
'''

"""
paradox is pretty good.
how are you?
why not?
"""
```

`'''`或`"""`，由三个单引或三个双引号号组成，将我们多行注释内容封装起来。

## 行和缩进

python对行和缩进有严格的要求，因为它不像其他语言一样，有`{}`这样的符号表示一段程序块，所以在python需要用缩进来表示代码与代码之间的层级关系。每个缩进需要用四个空格来表示，也可以用一个`tab`键来表示。例如

```python
if jinze is handsome:
    print("yes")
    if zhouyuxiang is beautiful:
        if jinze is not want_to_die:
            print("yes")
        else:
            print("no")
    
```

（以上代码是不能运行的，我们将在后续章节中解释，上述代码只是用来演示缩进）

## 多行语句

多行语句就是本来一个完整的句子太长，放在一行及不仅破坏了代码的美观性，还破坏了可读性。

整改前：
```python
if paradox is not handsome and dox is not beautiful or paradox is handsome and dox is not beautiful or paradox is not handsome and dox is beautiful or paradox is handsome and dox is beautiful:
```
多行整改后：
```python
if paradox is not handsome and dox is not beautiful 
    or paradox is handsome and dox is not beautiful 
    or paradox is not handsome and dox is beautiful 
    or paradox is handsome and dox is beautiful:
```
整改前：
```python
total = item_one + item_two + item_three
```
（这句话其实不算长，不用整改也行，只是举个例子）
整改后：
```python
total = item_one + \
		item_two + \
		item_three
```

## 引号

python中可以使用单引号`'`，双引号`"`，三引号`'''`或者`"""`，来表示字符串，其中三引号可以由多行组成。

```python
paradox = "jinze"
dox = "zhouyuxiang"
```

## 标识符

标识符有字母、数字、下划线组成，但是开头不能是数字，可以是下划线、字母，但是以下划线开头的表示符都有特殊的意义。
标识符对大小写敏感，例如`paradox`和`Paradox`是不同的标识符。

## 常量

例如自然底数$e$，圆周率$π$。

## 变量

变量存贮内存中的值，因此创建变量时，就会在内存中开辟一段内存空间，用来保存我们的值。

#### 变量赋值

```python
a = 100 # 这句话被执行之后，以后a就代表100了
name = "dox" # 以后name这个变量名就表示dox了
dox = "zhouyuxiang" # 以后dox这个变量名就表示zhouyuxiang了
```
那么上面的`dox`和`name`有什么区别呢？
#### 多变量赋值

```python
a = b = c = 1 # 以后a,b,c都将表示1
```

## 标准数据类型

### 数字

- 整型`(int)`：顾名思义就是整数，可以表示的范围是：$[-2^{31} , 2^{31} -1]$
- 浮点数`(float)`：就是小数
- 长整型`(long)`：可以表示更大范围的整数
- 复数`(complex)`

### 字符串

```
"i'm string"
"我是字符串"
'我是常量，我和1、2、3这样的数字的地位是一样的'
'我可以被双引号、单引号包裹，之前讲过的哦'
```

现在你知道之前的`dox`和`name`有什么区别了么？虽然`name`表示的是`"dox"`看起来一样，但其实就是`"dox"`和`dox`之间的区别。一个是常量，一个是变量，`"dox"`不能被赋值，而`dox`可以被赋值。

#### 字符串的操作

- 索引：

  ```python
  >>> a = "abcdefg"
  >>> print(a[0])
  >>> print(a[-1])
  ```

  ```pythton
  a
  ```

- 分割：

  ```python
  >>> dox = "abc def ghi jkl"
  >>> dox_list = dox.split() # 默认按空格分隔
  >>> print(dox_list)
  ```

  ```python
  ["abc", "def", "ghi", "jkl"]
  ```

- 合并：

  ```python
  >>> dox = "dox"
  >>> paradox = "paradox"
  >>> friends = dox + " and " + paradox
  >>> print(friends)
  ```

  ```
  dox and paradox
  ```

一个字符串的结构其实类似于一个列表结构，所有列表可以进行的操作，在字符串上基本上都可以进行。

### 列表

列表是一种数据结构，就像表格的某一列，我们告诉别人某个数据在表格的这一列的那个位置，通常都会告诉他，在这一列的第几行。列表也是一个道理。在python中列表的每一个位置的数据类型可以是任意数据类型，可以是整型，浮点型，字符串甚至是列表。我们看下面一个例子：

```python
dox_list = ['paradox', 1, 1.2222, ['dox', 2]]
```

怎么访问列表中的数据呢？就像表格一样我们告诉程序我们访问的数据在那个位置：

```python
>>> dox_list[0]
paradox
>>> dox_list[1]
1
```

记住，在任何编程语言中，索引的第一个数字都是$0$。

你会发现`dox_list[2]`同样是一个列表啊，要怎么访问这个列表中的元素呢？看下面的操作：

```python
>>> dox_list[2]
['dox', 2]
>>> dox_list[2][0]
dox
>>> dox_list[2][1]
2
```

那我们怎么修改列表中的元素呢，比如我要把`dox_list[0]`变成`jinze`。我们执行以下操作：

```python
>>> dox_list[0] = 'jinze'
>>> dox_list[0]
jinze
```

### 元组

元组类似于列表，不同的地方元组的值不能被改变，也就是说，元组是一个只读的数据类型，一旦被创建就不会被修改。

元组的表示形式如下：

```python
dox_tuple = ('paradox', 1, 2, 2.33333, ['dox', 1111], ('ppppp', 12312), 2222)
```

### 字典

我们想象一下，平时我们查字典的时候，是不是每个单词下面都有自己对应的解释？在程序中我们称前面的单词为：key，称其中的解释为：value。我们来看一个例子：

```python
dox_dic = {'jinze' : 'paradox', 'zhouyuxiang' : 'dox'}
```

再比如：

```python
phone_book = {'jinze' : 156****9980, 'dox' : 158*********}
```

怎么访问呢？

```python
>>> dox_dic['jinze']
paradox
>>> dox_dic['zhouyuxiang']
dox
```

这样我们的索引，和索引表示的内容都能有了特殊的意义，当然这还不是它主要的作用，他是一个集合类型，也就是说，元素之间会有互异性，只能有一个`jinze`也只能有一个`zhouyuxinag`，你会在以后的运用中，感受到字典的强大。

## 运算符

### 算数运算符

```python
>>> 1 + 2 
3
>>> 4 / 3
1.333333333333333
>>> 4 // 3
1
>>> 4 % 3
1
```

就是不同的加减乘除，余数，注意看上面`/`，和`//`的区别，后者是向下取整。

### 比较运算符

| 运算符 | 描述                             | 实例                              |
| ------ | -------------------------------- | --------------------------------- |
| `==`   | 判等符：比较两个对象是不是相等的 | `'paradox' == 'dox'`，返回`False` |
| `!=`   | 比较两个对象是不是不相等         | `'paradox' != 'dox'`，返回`True`  |
| `>`    | 大于                             | `2 > 1`，返回`True`               |
| `<`    | 小于                             | `1 < 0`，返回`False`              |
| `<=`   | 小于等于                         | `2 <= 2`， 返回`True`             |
| `>=`   | 大于等于                         | `1 >= 2`，返回`False`             |

不用在这纠结什么是返回`True`,`False`。

### 赋值运算符

| 运算符 | 描述             | 实例                                  |
| ------ | ---------------- | ------------------------------------- |
| `=`    | 简单的赋值运算符 | c = a + b 将 a + b 的运算结果赋值为 c |
| `+=`   | 加法赋值运算符   | c += a 等效于 c = c + a               |
| `-=`   | 减法赋值运算符   | c -= a 等效于 c = c - a               |
| `*=`   | 乘法赋值运算符   | c *= a 等效于 c = c * a               |
| `/=`   | 除法赋值运算符   | c /= a 等效于 c = c / a               |
| `%=`   | 取模赋值运算符   | c %= a 等效于 c = c % a               |
| `**=`  | 幂赋值运算符     | c \**= a 等效于 c = c\*\* a           |
| `//=`  | 取整除赋值运算符 | c //= a 等效于 c = c // a             |

### 逻辑运算符

Python语言支持逻辑运算符，以下假设变量 a 为 10, b为 20;

| 运算符 | 逻辑表达式 | 描述                                                         | 实例                      |
| ------ | ---------- | ------------------------------------------------------------ | ------------------------- |
| `and`  | `x and y`  | 布尔"与" - 如果 x 为 False，x and y 返回 False，否则它返回 y 的计算值。 | (a and b) 返回 20。       |
| `or`   | `x or y`   | 布尔"或"	- 如果 x 是非 0，它返回 x 的值，否则它返回 y 的计算值。 | (a or b) 返回 10。        |
| `not`  | `not x`    | 布尔"非" - 如果 x 为 True，返回 False 。如果 x 为 False，它返回 True。 | not(a and b) 返回 `False` |

### 成员运算符

| 运算符   | 描述                                                    | 实例                                                |
| -------- | ------------------------------------------------------- | --------------------------------------------------- |
| `in`     | 如果在指定的序列中找到值返回 True，否则返回 False。     | x 在 y 序列中 , 如果 x 在 y 序列中返回 `True`。     |
| `not in` | 如果在指定的序列中没有找到值返回 True，否则返回 False。 | x 不在 y 序列中 , 如果 x 不在 y 序列中返回 `True`。 |

### 身份运算符

| 运算符   | 描述                                        | 实例                                                         |
| -------- | ------------------------------------------- | ------------------------------------------------------------ |
| `is`     | is 是判断两个标识符是不是引用自一个对象     | **x is y**, 类似 **id(x) == id(y)** , 如果引用的是同一个对象则返回 True，否则返回 `False` |
| `is not` | is not 是判断两个标识符是不是引用自不同对象 | **x is not y** ， 类似 **id(a) != id(b)**。如果引用的不是同一个对象则返回结果 `True`，否则返回 `False`。 |

### 运算符优先级

| 运算符                     | 描述                                                   |
| -------------------------- | ------------------------------------------------------ |
| `**`                       | 指数 (最高优先级)                                      |
| `~ + -`                    | 按位翻转, 一元加号和减号 (最后两个的方法名为 +@ 和 -@) |
| `* / % //`                 | 乘，除，取模和取整除                                   |
| `+ -`                      | 加法减法                                               |
| `>> <<`                    | 右移，左移运算符                                       |
| `&`                        | 位 'AND'                                               |
| `^ |`                      | 位运算符                                               |
| `<= < > >=`                | 比较运算符                                             |
| `<> == !=`                 | 等于运算符                                             |
| `= %= /= //= -= += *= **=` | 赋值运算符                                             |
| `is is not`                | 身份运算符                                             |
| `in not in`                | 成员运算符                                             |
| `not or and`               | 逻辑运算符                                             |

# Project 3

## 认识语法

### 输入输出

```python
n = input()
print(n)
```

运行后的效果是：

```python
>>> 10
10
```

```python
>>> dox
dox
```

---

```python
in = input("请输入内容:")
print(in)
```

运行后的效果是：

```python
>>> 请输入内容:我今天偷偷回家吓了我爸妈一跳哈哈哈哈！
我今天偷偷回家吓了我爸妈一跳哈哈哈哈！
```

上面两者的区别就是，我们调用`input()`方法时，有没有给这个方法传参数，比如第一个没有传参数，运行时就没有提示信息，但是第二个我们传入了”请输入内容“的参数，就会在程序运行后先出现”请输入内容“这样的提示。

关于输出我们只要调用一下`print()`方法就可以了，参入的参数就是我们要输出的内容。

那我们要怎么格式化输出我们的东西的呢？比如我们先输入一个数字3并保存在n这个变量中，我们要输出的内容是：dox比paradox大n个月。我们可以用下面的输出方式:

```python
n = int(input())
print("dox比paradox大%d个月" % n)
print("dox比paradox大" + str(n) + "个月")
```

这里有三个小问题留给你：

- 为什么我要使用`int(input())`而不是`input()`？
- `%d`是什么意思？为什么不会输出"dox比paradox大%d个月"，而是输出了"dox比paradox大3个月"？
- 为什么第二种输出方式`n`要用`str(n)`的形式？

请通过微信或者微信语音聊天的方式告诉我你的答案或者疑问。

#### 小练习

[实验任务1-1~1-4](https://admin.buaacoding.cn/problem/index)

## 条件语句

条件语句，顾名思义就是：如果怎么样就会怎么样，如果是另外怎么样，就会怎么样，否则就会怎么样。

在python中的结构就是：

```python
if 怎么样:
    就会怎么样
elif 怎么样：
    就会怎么样
elif 怎么样：
    就会怎么样
...
else:
    就会怎么样
```

我们来看一个具体的例子：

```python
a = 10
b = 10
if a == 10:
    print("a + 1 = %d" %(a + 1))
elif b == 10:
    print("b = %d" %b)
else:
    print("lalala")
```

执行后的结果应该是：

```python
a + 1 = 11
```

#### 问题

为什么第二个条件同样满足却不会被执行呢？

我们再看：

```python
a = 10
b = 10
if a == 10:
    print("a == %d" %a)
if b == 10:
    print("b == %d" %b)
```

运行后会是什么样的结果？你自己运行试一下，并解释为什么包括上一个问题。

#### 小练习

在三个数求最大数值的案例中，还要输出是哪个数最大，应该如何设计算法？ 请给出代码。

## 循环语句

### `while`循环

`while`循环语句的结构是这样的：

```python
a = 0
while a != 10:
    a += 1
```

什么意思呢？就是`while`后面的条件`a != 10`成立时，就执行下面的语句`a += 1`，直到`a`被加到了10这个循环就会被终止。

### `for`循环

`for`循环语句的结构是这样的：

```python
for i in range(0, 11):
	print(i)
```

这句话等价与c语言中的如下`for`循环：

```c
int i;
for(i = 0; i < 11; i ++){
    printf("%d", i);
}
```

也就是我们现在有一个变量`i`（你可以叫变量`j,k,blaba`随便什么不是保留字的就行），要用这个`i`遍历范围$[0,11)$，左闭右开，每次执行完一次循环体里面的语句，就会执行此`i += 1`操作，这在c语言里面是显示的，但是在python中以迭代的方式进行。

#### 小练习

- 完成1-1000的加法

### 嵌套循环

我们看下面几个简单的嵌套循环：

```python
for i in range(0, 11):
	for j in range(0, 11):
        print(i + j)
```

```python
for i in range(0, 11):
    j = 0
    while j != 11:
        j += 1
        print(i + j)
```

```python
i = 0
while i != 11:
    i += 1
    for j in range(0, 11):
        print(i + j)
```

上述三种嵌套循环的输出是一样的吗？如果一样为什么？如果不一样为什么？

#### 小练习

用两层循环打印出一个形如下的$3*3$的坐标矩阵

```python
(1,1) (1,2) (1,3)
(2,1) (2,2) (2,3)
(3,1) (3,2) (3,3)
```

**Hint**：`print`函数输出时是默认会换行的比如：

```python
print("dox")
print("paradox")
```

输出为：

```python
dox
paradox
```

那要怎么使它不换行呢？就用如下方式：

```python
print("dox", end="")
print("paradox", end="")
print("dxxx")
```

输出为：

```python
doxparadox
```

这里面`end`是给`print`这个方法传的另一个参数，表示以什么作为结尾。

### 循环控制语句

#### `break`

`break`命令一执行就会**跳出循环**，我们看下面几个例子：

```python
a = 0
while True: # True 是python中的保留字 while True，则condition永远为True，所以我们得在内部停止它
    if a == 10:
        break
	a += 1
```

```python
n = int(input())
for i in range(2, n):
    if n % i == 0:
        print("%d is not a Prime Number." %n)
        break
```

```python
n = int(input())
for i in range(0, n):
    m = int(input())
    for j in range(0, m):
        if j % 7 == 0:
            break
```

#### 问题

在第三个样例中，`break`结束的是哪个循环？

##### `continue`

`continue`比较简单，就是什么都不做，继续循环体下一步：

```python
for i in range(0, 101):
    if i % 2 != 0:
        continue
    print(i)
```

虽然上述可以被简化成：

```python
for i in range(0, 101):
    if i % 2 == 0:
        print(i)
```

但我只是作为一个例子，我们再看下面的一个例子：

```python
for i in range(0, 101):
    if i % 2 == 0:
        continue
        print(i)
    print(i)
```

上面这个例子程序会输出什么呢？

#### `pass`

`pass`不想讲了，和`continue`类似，所用在写完一个函数的定义或者循环体之后，内容没写，可以用这个填充

```python
def dox: # 定义一个函数
    pass # 啥都没干 先pass一下，可以过编译
```
#### 小练习

[实验任务2-1~2-5](https://admin.buaacoding.cn/problem/index)

## 列表操作

我们知道列表的结构:

```python
dox_list = ["A", "B", 111, 222, 1.11, "Dox", [1111, 2222], (1,2)]
```

我们现在有一个空的列表：

```python
dox_list = []
```

### 加入元素

现在我们要按照我们自己的想法往列表里面加入元素，比如我们要把0-10中所有偶数加到列表中，我们使用我们调用列表在python中给我提供的方法`append`，如下：

```python
for i in range(0, 11):
    if i % 2 == 0:
        dox_list.append(i)
print(dox_list)
```

```python
[0, 2, 4, 6, 8, 10]
```

### 更新元素

我们想把第二个偶数换成3：

```python
dox_list[1] = 3
print(dox_list)
```

```python
[0, 3, 4, 6, 8, 10]
```

### 删除列标元素

我们把3这个不是偶数的数删去

```python
del(dox_list[2])
print(dox_list)
```

```python
[0, 4, 6, 8, 10]
```

### 列表的排序

我们现在有一个纯数字的列表：

```python
dox_list = [2, 3, 1, 4, 24, 5, 56]
```

然后我们对其升序排序：

```python
dox_list.sort()
pritn(dox_list)
```

就可以得到：

```python
[1, 2, 3, 4, 5, 24, 56]
```

---

如果我们这样：

```python
dox_list.sort(reverse=True)
print(dox_list)
```

就会得到：

```python
[56, 24, 5, 4, 3, 2, 1]
```

降序输出。

### 列表脚本操作符

列表对 + 和 * 的操作符与字符串相似。+ 号用于组合列表，* 号用于重复列表。

如下所示：

| Python 表达式                           | 结果                         | 描述                 |
| --------------------------------------- | ---------------------------- | -------------------- |
| `len([1, 2, 3])`                        | 3                            | 长度                 |
| `[1, 2, 3] + [4, 5, 6]`                 | [1, 2, 3, 4, 5, 6]           | 组合                 |
| `['Hi!'] * 4`                           | ['Hi!', 'Hi!', 'Hi!', 'Hi!'] | 重复                 |
| `3 in [1, 2, 3]`                        | True                         | 元素是否存在于列表中 |
| `for x in [1, 2, 3]: print(x, end=" ")` | 1 2 3                        | 迭代                 |

### 列表截取与拼接

列表截取与字符串操作类型，如下所示：

```python
L=['Google', 'Runoob', 'Taobao']
```

操作：

| Python 表达式 | 结果                 | 描述                                               |
| ------------- | -------------------- | -------------------------------------------------- |
| `L[2]`        | 'Taobao'             | 读取第三个元素                                     |
| `L[-2]`       | 'Runoob'             | 从右侧开始读取倒数第二个元素: count from the right |
| `L[1:]`       | ['Runoob', 'Taobao'] | 输出从第二个元素开始后的所有元素                   |

列表还支持拼接操作：

```python
>>> squares = [1, 4, 9, 16, 25]
>>> squares += [36, 49, 64, 81, 100]
>>> squares
[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
```

### 列表函数&方法

| 序号 | 函数                           |
| ---- | ------------------------------ |
| 1    | `len(list)`列表元素个数        |
| 2    | `max(list)` 返回列表元素最大值 |
| 3    | `min(list)`返回列表元素最小值  |
| 4    | `list(seq)` 将元组转换为列表   |

| 序号 | 方法                                                         |
| ---- | ------------------------------------------------------------ |
| 1    | `list.append(obj)` 在列表末尾添加新的对象                    |
| 2    | `list.count(obj)`统计某个元素在列表中出现的次数              |
| 3    | `list.extend(seq)` 在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表） |
| 4    | `list.index(obj)`从列表中找出某个值第一个匹配项的索引位置    |
| 5    | `list.insert(index, obj)` 将对象插入列表                     |
| 6    | `list.pop([index=-1\]])`移除列表中的一个元素（默认最后一个元素），并且返回该元素的值 |
| 7    | `list.remove(obj)` 移除列表中某个值的第一个匹配项            |
| 8    | `list.reverse()`反向列表中元素                               |
| 9    | `list.sort(cmp=None, key=None, reverse=False)` 对原列表进行排序 |
| 10   | `list.clear()` 清空列表                                      |
| 11   | `list.copy()` 复制列表                                       |

## 字典操作

### 字典的访问

```python
dict = {'Alice': '2341', 'Beth': '9102', 'Cecil': '3258'}
```

```python
>>> dict['Alice']
'2341'
>>> dict['Cecil']
'3248'
```

### 字典的修改

```python
dict = {'Name': 'Zara', 'Age': 7, 'Class': 'First'};
dict['Age'] = 8 # modify the Age
dict['School'] = 'DPF School' # Add new entry
```

### 删除字典元素

```python
dict = {'Name': 'Zara', 'Age': 7, 'Class': 'First'};
del dict['Name'] # 删除一个条目
dict.clear() # 清空整个字典，但是dict这个字典还在，只是里面什么都没有了
del dict # 删除这个字典
```

### 字典内置函数方法

| 序号 | 函数及描述                                                   |
| ---- | ------------------------------------------------------------ |
| 1    | `cmp(dict1, dict2)` 比较两个字典元素。                       |
| 2    | `len(dict)` 计算字典元素个数，即键的总数。                   |
| 3    | `str(dict)` 输出字典可打印的字符串表示。                     |
| 4    | `type(variable)` 返回输入的变量类型，如果变量是字典就返回字典类型。 |

| 序号 | 函数及描述                                                   |
| ---- | ------------------------------------------------------------ |
| 1    | `dict.clear()`删除字典内所有元素                             |
| 2    | `dict.copy()`返回一个字典的浅复制                            |
| 3    | `dict.fromkeys(seq[, val\])`创建一个新字典，以序列 seq 中元素做字典的键，`val` 为字典所有键对应的初始值 |
| 4    | `dict.get(key, default=None)` 返回指定键的值，如果值不在字典中返回default值 |
| 5    | ~~`dict.has_key(key)` 如果键在字典dict里返回`True`，否则返回`False`~~(python2用法，我们是python3)，可以用`key in dict.keys()`来表示 |
| 6    | `dict.items()`以列表返回可遍历的(键, 值) 元组数组            |
| 7    | `dict.keys()`以列表返回一个字典所有的键                      |
| 8    | `dict.setdefault(key, default=None)` 和`get()`类似, 但如果键不存在于字典中，将会添加键并将值设为default |
| 9    | `dict.update(dict2)` 把字典dict2的键/值对更新到dict里        |
| 10   | `dict.values()` 以列表返回字典中的所有值                     |
| 11   | `pop(key[,default\])` 删除字典给定键 `key` 所对应的值，返回值为被删除的值。key值必须给出。 否则，返回`default`值。 |
| 12   | `popitem()` 随机返回并删除字典中的一对键和值。               |

#### 小练习

[知难而上](https://admin.buaacoding.cn/problem/1326/index)

## Project 4

### 函数

#### 函数的定义

- 函数代码块以单词`define`的缩写`def`开头，`def`是一个关键字，即保留字；
- `def`后接函数的名字，也就是你给这个函数取的名字，名字后**紧**跟的是`()`；
- `()`内是我们传入的参数，`()`之后紧跟的就是`:`；
- 注意函数块内的缩进，块内代码首先是从属于上面函数的定义的，其他缩进规则个之前所说一样；

```python
# exampel
def get_sum(n):
    sum = 0
    for i in range(1, n + 1):
        sum += i
    print(sum)
```

上面这个函数的作用就是计算$1$到$n$的值，然后打印出来，因此我们传入的参数就是$n$。

我们要怎么使用这个函数呢？

```python
def get_sum(n):
    sum = 0
    for i in range(1, n + 1):
        sum += i
    print(sum)
    
# 直接传入一个具体的参数    
get_sum(10000)
# 可以传入我们定义的变量
n = int(input())
get_sum(n)
```

### 变量的生命周期

#### 全局变量

全局变量就是在整个代码的任意地方都可以被调用的变量。

#### 私有变量

只能被定义它的代码块内部访问，其他地方均不能访问。

#### 实例

```python
a = 99
def print_some():
    b = 100
    print(a)
    print(b)
print_some()
print(b)
```

运行后：

```python
99
100
-----------------------------------------------------------------------
NameError                             Traceback (most recent call last)
<ipython-input-11-6fad8591f56a> in <module>()
      5     print(b)
      6 print_some()
----> 7 print(b)

NameError: name 'b' is not defined
```

`NameError`那一行就很明确的指出了你的`b`没有被定义，因为`b`的生命周期就只有一个函数那么短，它只在函数中定义了，并在函数结束时，结束了生命，因此其它地方会认为`b`这个名字并没有被定义。

### 参数

```python
def sum1(a, b):
    print("%d + %d = %d" %(a ,b ,a + b))
    
sum1(1, 2)
```

我们为函数`sum1`设置了两个参数`a,b`，我们按顺序将`1,2`传入函数。

输出就是：

```python
1 + 2 = 3
```

我们调换以下传入参数`1,2`的顺序就是：

```python
2 + 1 = 3
```

我们换一种方式：

```python
def sum1(a, b):
    print("%d + %d = %d" %(a ,b ,a + b))
    
sum1(b=2, a=1)
```

输出就是：

```python
1 + 2 = 3
```

这里的意思就是我们指定了哪个参数的是什么。

#### 函数的返回值

