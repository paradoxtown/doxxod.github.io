---
layout:     post
title:      Beginning of Machine Learning
subtitle:   Learn from Tensorflow
date:       2017-11-11
author:     paradox
header-img: img/post_2.jpg
catalog: true
mathjax: true
tags:
    - Machine Learning 
    - Tensorflow
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
## 0.开始MNIST

$Tensorflow$使用图$(graph)$来表示计算的编程系统，图中的节点被称为 $op(operation)$.一个$op$获得零个或多个张量$(tensor)$执行计算，产生零个或多个张量.<br>
<b>张量</b>是一个按<b>类型</b>划分的多维数组.
图是一种对计算的抽象描述.在计算之前，图必须在<b>会话</b>$(Session())$中被启动. 会话将图的$op$分发到如$CPU$或$GPU$之类的设备上，同时提供执行$op$的方法.这些方法执行之后，将产生张量返回。
$Tensorflow$编程分为两个阶段：构建阶段和执行阶段.第一个阶段用于组织计算图，而后者利用$session$中执行计算图中的$op$操作.
### 认识几个重要的函数：
#### $softmax$回归：
$$softmax\ regression: evidence_i = \sum\limits_j W_{i,j}x_j + b_i$$
$$y = softmax(evidence)$$
$softmax$回归分两步：首先对输入被分类的对象属于某个类$(i)$的“证据”相加求和，然后将这个“证据”的和转化为概率$y$.
上述两个公式中，$W_i$代表权重，$b_i$代表第$i$类的偏置量（它代表了与所有输入向量无关的判断证据），$j$代表给定的图片的<b>像素</b>所引用于像素求和.
#### 交叉熵$(cross-entropy)$：
$$H_{y'}(y) = - \sum\limits_i {y'}_i\log(y_i)$$
$y$是我们预测的概率分布，$y'$是实际分布(我们输入的$one\_hot\ vector$).该函数用来及计算成本，来评估我们的模型是好是坏.<br>
```py
    y = tf.matmul(x, W) + b
    y_= tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(lables = y_, logits = y))
```
### 模型评估：
通过上面的代码运行，我们的到了`y[None, 10]`表示每张图，都有了其相对应的独热码$(one\_hot)$.
`correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))`
其中`tf.argmax`是一个非常有用的函数.它会返回一个张量某个维度中的最大值的索引.例如，`tf.argmax(y, 1)`表示我们模型对每个输入的最大概率分类的分类值.而`tf.argmax(y_, 1)`表示真实的分类值.我们可以用`tf.equal`来判断我们的预测是否与真实分类一致.这里返回的是一个布尔数组.
`accuracy=tf.reduce_mean(tf.cast(correct.prediction, "float"))`
## 1.深入MNIST
为了得到更好的精度，我们将采用卷积$\to$池化的方式，构建多层卷积网络模型.
完整的卷积网络训练过程如下：

 - <b>第一步：</b>我们初始化所有的滤波器，使用随机值设置参数/权重.

 - <b>第二步：</b>网络接收一张训练图像作为输入，通过前向传播过程（卷积、ReLU和池化操作，以及全连接层的前向传播），找到各个类型的输出概率.

 - <b>第三步：</b>在输出层计算总误差.

 - <b>第四步：</b>使用反向传播算法，根据神经网络的权重计算误差的梯度，并使用梯度下降算法更新所有滤波器的值/权重以及参数的值，是输出误差最小化.
 - <b>第五步：</b>对训练集中所有的图像重复步骤1~4.

### 什么是反向传播算法？
通常缩写为$BackProp$是人工神经网络$(ANN)$可以被训练的几种方法之一.这是一个有监督的训练方案.即：“从错误中学习”.
#### BackProp算法：
最初所有的边权重都是随分配的.对于训练数据的每一个输入，$ANN$都被激活并观察其输出，这个输出与我们已经知道的所需输出进行比较，并且“错误”被传回前一层.这个错误被记录下来，并且<b>相应的调整权重</b>.重复该过程，直到输出误差低于预定阈值.
### 什么是全连接层？
[全连接层介绍](https://ujjwalkarn.me/2016/08/09/quick-intro-neural-networks/)
### 对上述五个步骤的解释：
上述五个步骤可以简单的归纳为：1、卷积；2、非线性处理；3、池化或者采样；4、分类（全连接层）；5、反向传播.
#### 卷积：
在$CNN$术语中，$n * n$的矩阵叫做“滤波器$(filter)$”，或者“核$(kernel)$”或者“特征检测器$(feature detector)$”,通过在图像上滑动滤波器，并计算点乘得到矩阵叫做“卷积特征$(convolved Feature)$”，或者“激活图$(Activation Map)$或者“特征图$(Feature Map)$”.记住滤波器在原始输入图像上的作用是特征检测器.我们需要在进行卷积前确定：深度（对应的是卷积操作所需的滤波器个数）、步长$(stride)$、零填充$(zero-padding)$.
#### 非线性处理$(ReLU)$：
$ReLU$表示修正线性单元$(Rectified Linear Unit)$， 是一个非线性操作。
$ReLU$是一个元素级别的操作（应用到个个像素），并将特征图中的所有小于0的像素设置为零.这样做的原因是我们希望$ConvNet$学习的实际数据是非线性的（卷积是一个线性操作）.
#### 池化操作：
空间池化$(Spatial Pooling)$，也叫做亚采样或者下采样，降低了各个特征图的维度，但可以保证大部分重要的信息.空间池化可分为下面几种方式：最大化、平均化、加和等等.
对于最大池化$(Max Pooling)$，我们定义一个空间邻域（比如，2x2 的窗口），并从窗口内的修正特征图中取出最大的元素。除了取最大元素，我们也可以取平均$(Average Pooling)$或者对窗口内的元素求和。在实际中，最大池化被证明效果更好一些。
#### 分类：
<b>全连接层$(Fully Connected)$</b>是传统的多层感知器，表示前面层的所有神经元都与下一层的所有神经元连接.
卷积和池化层的输出表示了输入图像的高级特征。全连接层的目的是为了使用这些特征把输入图像基于训练数据集进行分类。比如，在下面图中我们进行的图像分类有四个可能的输出结果（注意下图并没有显示全连接层的节点连接）。

