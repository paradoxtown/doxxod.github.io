---
layout:     post
title:      Research Note
subtitle:   NELIST
date:       2018-07-12
author:     paradox
header-img: img/post-bg-unix-linux.jpg
catalog: true
mathjax: true
tags: 
    - note
---

## 词包模型$(bag\ of\ words)$

用整个语料库构建一个$dictionary$，这个$dictionary$中记录的是整个语料库中互不相同的单词（相对于词干化后的），$dictionary$中有多少个单词，就是我们接下来为每一篇$text$构建的向量的维度，向量的每一维表示的是，该单词在这篇$text$中出现的次数。

## $word2vec$

$word2vec$是词嵌入$(word embedding)$的一种，词嵌入就是将词转化成数值形式，嵌入到一个数字空间。

- $ski-gram$：用一个词作为输入，来预测它的上下文；
- $cbow$：用一个词的上下文作为输入，来预测这个词的本身。

$word2vec$模型的原始输入是$one-hot​$数据。

## 语料预处理

- 删除标点符号
- 删除停用词
- 词干化
- 记录低频词，删除低频词

## TF-IDF

- **词频**：
  $$
  tf_{i, j} = \frac{n_{i, j}}{\sum _k n_{k, j}}
  $$

- **逆向文件频率$(inverse\ document\ frequency, idf)$**：是一个词语普遍重要性的度量。某一特定的词语的$idf$里可以由总文件数目除以包含该词语的文件数目，再将得到的商取以$10$为底的对数得到：
  $$
  idf_i = log\frac{|D|}{|\{j : t_i \in d_j\}|}
  $$

- 最后两者相乘：
  $$
  tf-idf_{i,j} = tf_{i.j} * idf_i
  $$
  



## Graph Convolutional Network$(GCN)$

离散卷积的本质就是一种加权求和。

如下图所示，$CNN$中的卷积本质上就是利用一个共享参数的过滤器$(Kernel)$，通过计算中心像素点以及相邻像素点的加权和来构成$feature\ map$实现空间特征的提取，当然加权系数就是卷积核的权重系数。

卷积核的系数是由随机化初值，然后根据误差函数通过反向传播梯度下降进行迭代优化的，这是一个调参的过程。卷积核通过优化求出才能实现特征提取的作用，$GCN$的理论很大一部分工作就是**引入可优化的卷机参数。**

![preview](https://github.com/paradoxtown/paradoxtown.github.io/blob/master/img/note1.jpg?raw=true) 

$CNN$是$Computer\ Vision$中要的模型，它可以有效的提取空间特征。但是有一点值得注意的就是他提取的数据类型往往是**排列整齐的像素矩阵**，也就是很多论文中提到的$Euclidean\ Structure$。

但是在科学研究中还有很多$Non\ Euclidean\ Structure$数据，如社交网络、信息网络中就有很多类似的结构。这样的网络结构就是图论中的拓扑图。

### 提取拓扑空间特征的$Spectral\ Domain$方式

$spectral\ domain$是$GCN$的理论基础，这种思路就是希望借助图谱的理论来实现拓扑图上的卷积操作。

$spectral\ domain\ theory$简单的概括就是借助**图的拉普拉斯矩阵的特征值和特征向量**来研究图的性质。

### 拉普拉斯矩阵

对于图$G = (V, E)$， 其中$Laplacian$矩阵的定义为$L = D - A$，其中$L$是$Laplacian$矩阵，$D$是顶点的度矩阵（对角矩阵），对角线上的元素依次为各个顶点的度，$A$是图的邻接矩阵。

![preview](https://github.com/paradoxtown/paradoxtown.github.io/blob/master/img/note2.jpg?raw=true) 

常用的拉普拉斯矩阵实际上有三种：

- $Combinatorial\ Laplacian: L = D - A​$
- $Symmetric\ Normalized\ Laplacian: L^{sys} = D^{-1/2}LD^{-1/2}$
- $Random\ Walk\ Normalized\ Laplacian: L^{rw} = D^{-1}A$

#### 问题

- 什么是$logits?$

  $logits$和$softmax$都是属于咋i输出层的内容
  $$
  logits = tf.matmul(X, W) + bias
  $$
  在通过归一化处理就是$softmax$。可以这样理解，$logits$是未进入$softmax$的概率，一般是全连接层的输出，是$softmax$的输入。

  在我们的代码中他是**推理阶段**的输出：

  ```python
  op_logits = self.inference(self.ph_data1, self.ph_data2, self.ph_dropout)
  ```

- `out_map`的作用是什么？是映射什么东西？为什么传入的参数是文本的测试数据和图片的训练数据？

- `out_tsne`的作用是什么？

- 关于构建$graph\ structure$，我构建的图是这样的：

  - 提取出`train`文本中所有的词干化后的名词；
  - 将这些名词按照字典序排序；
  - 构建矩阵，当某两个名词出现在同一个句子中的时候，连一条边。

  那么问题就是，我要在代码中如何运用我的这个$graph$？

## Embedding

一个$Embedding$是从离散空对象（如单词）到实数向量的映射。例如，英文单词（成千上万）的$300$维$Embedding$可以包括：

```
blue:  (0.01359, 0.00075997, 0.24608, ..., -0.2524, 1.0048, 0.06259)
blues:  (0.01396, 0.11887, -0.48963, ..., 0.033483, -0.10007, 0.1158) 
orange:  (-0.24776, -0.12359, 0.20986, ..., 0.079717, 0.23865, -0.014213) 
oranges:  (-0.35609, 0.21854, 0.080944, ..., -0.35413, 0.38511, -0.070976) 
```

这些向量中的各个维度没有固定的意义。机器学习要利用的是这些向量之间的位置和距离的整体模式。

### TensorFlow Embedding

- 将文本分成单词，然后为词汇表中的每个单词分配一个整数($id$)；

- 创建`Embedding`变量，并使用`tf.nn.embedding_lookup`函数：

  ```python
  word_embeddings = tf.get_variable(“word_embeddings”,
      [vocabulary_size, embedding_size])
  embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, word_ids)
  ```



---

## wordnet

>$wordnet$包含了语义信息，所以有别于通常意义上的字典。
>
>$wordnet$根据词条的意义将他们分组，每一个具有相同意义的词条组成一个$synset$（同义词集合）。wosdnet为每一个$synset$提供了简短，概要的定义，并记录不同$synset$之间的语义关系。
>
>- 它既是一个字典，有时一个词典，它比单纯的词典或者词典都更加易于使用；
>- 支持自动的文本分析以及人工智能的应用。

## KNN

$k$近邻算法。

## DNN tips

- $train$上表现好而$test$上表现不好往往是$over fitting$造成的，但也不一定是其造成的。
- $test$上表现不好，要先检查$train$上表现好不好。
- 假设$test$上表现不好：
  - 换$activation function$：$network$叠得很深的时候会出现$vanishing\ gradient\ problem$现象。前几层的$gradient$比较小，后几层的gradient比较大，前几层比较慢，后记层比较快，当后几层已经收敛了，前几层还没有收敛，参数就停止更新了，而后几层的input都是由前几层来的，因此收敛的地方很大概率是错的（$sigmoid$)。理论上可以设计动态的$learning rate$。现在主要可以使用$relu$，还有就是让$network$自己学出一个$activation function$，用$maxout$。
  - 改变优化方法：$RMSProp$，因为我们有时候需要不一样的learning rate。把惯性加入到$gradient decent$中去，$Momentum$：前一次所移动的方向当作惯性的方向。$RMSProp + Momentum$ 就是 $adam$。
  - Early Stopping：理论上loss是会越来越小，除非learning rate设置错误。又时候我们的train上面，loss是不断减小的，但是test的loss有可能会先减小后增大，因此这时候我们加一个$validation set$。
  - $Regularization$：还要是参数的值越小（越接近零宇越好）越好，也叫$weight\ decay$。$ L1\ regularization\ L2\ Regularization$。
  - $Dropout$：$update$参数之前，我们都对每一个$neuron$做$ensemble$。有一些neuron就会以一定的几率消失掉。所以每一次的$network$都会不一样。好像在做一个$function$的平均。$Dropout$是$ensemble$的终极版本。在做Dropout之后test的weight得是train的weight的$(1-p\%)$，其中$p%$是$ensemble$的几率。

## RNN

### LSTM

我们之前把前面运算出来的$output$存到$memory$中以期达到记忆的能力，而现在比较流行的$memory$是$long\ short-term\ memory$（比较长的短期记忆）。

![1537284081307](https://github.com/paradoxtown/paradoxtown.github.io/blob/master/img/simple_lstm_cell.png?raw=true)

$LSTM$总共有$3$个$gate$，公式$input$，$forget$和$output$，所以总共有四个$input$，一个$output$。

我们要做$LSTM$的模型，只需要将原来的$neuron$替换成$LSTM\ cell$。 

![1537285420474](https://github.com/paradoxtown/paradoxtown.github.io/blob/master/img/complex_lstm_cell.png?raw=true)

比较复杂版本的$LSTM$。

现在有简化版的$LSTM$是$GRU$，只有两个$gate$，参数减少了，但是效果差不多。

## transE

知识库通常可以看作是三元组的集合，所谓三元组，也就是（头实体，关系，尾实体）的形式，头实体和尾实体统称为实体，简化为$(\vec h, \vec r, \vec t)$。

$transE$模型认为一个正确的三元组的$embedding(\vec h, \vec r, \vec t)$会满足$\vec h + \vec r = \vec t$，也就是说，头实体$embedding$加上关系$embedding$会等于尾实体$embedding$。

定义势能函数：
$$
f(h, r, t) = ||\vec h + \vec r - \vec t||_2
$$
也即欧几里得范数，表示两点之间的欧氏距离。

对于一个三元组来说，我们希望是能越低越好，而对于一个错误的三元组，我们希望势能越高越好。这样我们得到目标函数：
$$
min\sum_{(\vec h, \vec r, \vec t) \in \Delta}\sum_{(\vec h', \vec r', \vec t')\in \Delta'}[\gamma + f(\vec h, \vec r, \vec t) - f(\vec h', \vec r',\vec t')]_+
$$
其中：

- $\Delta$表示正确的三元组集合；
- $\Delta'$表示错误的三元组集合；
- $\gamma$表示正负样本之间的距离，是一个常数；
- $[x]_+$表示$max(0, x)$。

为了方便训练，避免**过拟合**，通常会进行$normalization$：
$$
||h|| \le1,||r|| \le1,||t|| \le1
$$

## mAP

$mAP(Mean\ Average\ Precision)$，$P(precision)$精度，正确率。和正确率一块出现的是召回率$R(recall)$，对于一个查询，返回了一系列的文档，正确率指的是返回的结果中相关的文档所占的比例，召回率指的是返回结果中相关文档占所有相关文档的比例。

正确率只考虑了返回的相关文档的个数，却没有考虑文档之间的序。为了体现越相关的文档应该越靠前的评估效果，于是有了$AP$的概念，对于一个有序的列表，计算$AP$的时候要先求出每个位置上的$precision$，然后对所有的位置的$precision$再做一个$average$。$precision$的定义前面已经给出。

$mAP$即对这多个查询的平均。

## AUC

$AUC(Area\ Under\ Curve)$，主要用于二分类问题的评估，表示曲线$ROC$下面的面积。

$ROC$曲线的$x$轴是伪阳性率$(false\ positive\ rate)$，$y$轴是真阳性率$(true\ positive\ rate)$。对于二分类问题我们一个样本的类别只有两种，我们用$0,1$分别表示两种类别，$0$和$1$也可以分别叫做阴性和阳性。当我们用一个分类器进行概率预测时，对于真是为$0$的样本，我们可能预测为$0$，也可能预测为$1$，真是为$1$同理。

真阳性率=（真阳性的数量）/（真阳性的数量+伪阴性的数量）
伪阳性率=（伪阳性的数量）/（伪阳性的数量+真阴性的数量）

## Comprehension or Translation of Papers

### *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*

这篇论文主要向我们展示了一种新颖的$RNN$模型：$RNN\ Encoder-Decoder$，这个模型有两个$RNN$神经网络构成。其中一个$RNN$将一个序列进行编码，使其变成一个固定长的向量；另一层为解码器，将之前的向量解码成另一个符号的序列。该模型的编码器和解码器将会共同训练以期达到由`source sequence`得到`target sequence`最大的概率值，用该条件概率来作为的评估值。

$RNN$是一个由一个在可变序列$X = (x_1, x_2, ..., x_T)$上运行的隐藏状态$h$和一个可选输出$y$组成。在每一个时间步长上，状态$h$的更新函数如下：
$$
h_{<t>} = f(h_{<t-1>}, x_t) \tag{1}
$$
其中$f$是一个非线性的激活函数，可能如同$sigmoid$般简单，也可能如同$LSTM$般复杂（后面应该都会使用$LSTM$）。

$RNN$可以获得一个预测下一个符号的概率。在这种情况下，**在每一个时间步长$t$的输出是条件分布**：$p(x_t | x_{t-1}, \dots, x_1)$.举个例子，一个多项分布可以用一个$softmax$激活函数来输出：
$$
p(x_{t,j}=1|x_{t-1},\dots,x_1) = \frac{\exp(w_jh_{<t>})}{\sum \limits _{j'=1} ^K \exp(w_{j'}h_{<t>})} \tag{2}
$$
$j$是所有$K$种可能中的一种，$w_j$表示的是$weight\ matrix\ W$的第$j$行。通过加和操作我们可以得到序列$x$的可能性：
$$
p(x) = \prod \limits ^T _{t = 1} p(x_t|x_{t-1}, \dots, x_1) \tag{3}
$$
以概率的视角来看，这个新的模型相当于是学习一个从一个可变长度序列到另一个可变长度序列的条件分布：$p(y_1, \dots, y_{T'}|x_1, \dots, x_T)$.

![1540227537670](https://github.com/paradoxtown/paradoxtown.github.io/blob/master/img/seq2seq.png?raw=true)

$Encoder$每次读取一个`symbol`，$RNN$中间的隐藏状态根据$(1)$来改变状态。当读到`EOS`的时候，隐藏状态就是整个序列的摘要$c$。

$Decoder$是被训练以用来输出的`output sequence`另一层$RNN$。在隐藏层$h_{<t>}$的基础上预测下一个`symbol`也就是$y_t$来产生`output sequence`。

$y_t, h_{<t>}$均是由$y_{t-1}$和$c$来决定的。因此，隐藏状态$h_{<t>}$由以下公式产生：
$$
h_{<t>} = f(h_{<t-1>}, y_{t-1}, c).
$$

下一个`symbol`的条件分布：

$$
P(y_t|y_{t-1}, y_{t-2},\dots, y_1, c)=g(h_{<t>}, y_{t-1}, c).
$$

$f$和$g$均是激活函数，其中后者必须是可以产生有效的概率的函数，例如$softmax$。

$RNN\ Encoder-Decoder$的两个部分被共同训练以期达到最大的对数似然数。

$$
\max \limits _\theta \frac{1}{N} \sum \limits _{n=1} ^N \log p_\theta(y_n|x_n) \tag{4}
$$


