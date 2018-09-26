---
layout:     post
title:      科研学习笔记
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



## 代码解构

```python
x_x0,c_x0,x_x1,c_x1,x_y0,c_y0,x_y1,c_y1 = utils.prepair_data()
```

（这里面prepair是不是拼错了）我们去看`utils`库中的`prepair_data()`函数

```python
def prepair_data():

    x0_train = np.load('./data/text_train_x.npy')
    x0_test = np.load('./data/text_val_x.npy')
    y0_train = np.load('./data/text_train_y.npy')
    y0_test = np.load('./data/text_val_y.npy')

    x1_train = np.load('./data/clipart_train_x.npy')
    x1_test = np.load('./data/clipart_val_x.npy')
    y1_train = np.load('./data/clipart_train_y.npy')
    y1_test = np.load('./data/clipart_val_y.npy')
    return np.expand_dims(x0_train,2),np.expand_dims(x0_test,2),np.expand_dims(x1_train,2)\
            ,np.expand_dims(x1_test,2),y0_train,y0_test,y1_train,y1_test
```

可得:

| 变量名 | 函数返回值                   |
| ------ | ---------------------------- |
| `x_x0` | `np.expand_dims(x0_train,2)` |
| `c_x0` | `np.expand_dims(x0_test,2)`  |
| `x_x1` | `np.expand_dims(x1_train,2)` |
| `c_x1` | `np.expand_dims(x1_test,2)`  |

### 构造正负样本

```python
train_index = utils.new_index(TRAIN_NUM,TRAIN_NUM,x_y0,x_y1)
test_index = utils.new_index(TEST_NUM,TEST_NUM,c_y0,c_y1)
```

```python
def new_index(n1,n2,y1,y2):
    cate = 205
    n1 = int(n1*(cate-1) / cate)
    n2 = int(n2*(cate+1) / cate)
    ind,r1,r2 = [],[],[]
    # print(choice(np.where(y1==0)[0]))
    # print(np.where(y2==0))
    for i in range(cate):
        for _ in range(int(n1/cate)):
            if len(np.where(y1==i)[0])!=0 and len(np.where(y2==i)[0])!=0:
                r1.append(choice(np.where(y1==i)[0]))
                r2.append(choice(np.where(y2==i)[0]))
    for j in range(n2):
        r1.append(random.randint(0,len(y1)-1))
        r2.append(random.randint(0,len(y2)-1))
    ind.append(r1)
    ind.append(r2)
    return np.array(ind)
```

```python
train_index = array([[  29,   20,   10, ..., 3515, 3425, 4565],
       [  54,   25,   53, ..., 7958, 9649, 1609]], dtype=int64)
```

其中`train_index[0]`是我们从`text_train`训练数据中按一定随机机制排序出来的训练数据的索引。`train_index[1]`同理。

我们再看`len(train_index[0])`为$75309$，而`max(train_index[0])`是$9751$，这个$9751$刚好是我们`x_x0`列表的长度，因此可以知道`train_index[0]`里面有很多重复的索引，这就说明我们一个文本数据可以对应多个相同或者不同的图片索引，图片索引可以对应多个不同的或者相同的文本索引。

```python
x0_train = x_x0[train_index[0],:,:]
x1_train = x_x1[train_index[1],:,:]
y0_train = x_y0[train_index[0]]
y1_train = x_y1[train_index[1]]
y_train = np.ones([len(train_index[0])])
y_train[y0_train != y1_train ] = 0
```

我们将文本的训练数据按照刚刚的`train_index[0]`的顺序重新组织，再将图片的训练数据按照刚刚的`train_index[1]`的顺序重新组织，文本的`label`和图片的`label`也按照相同的方式重新组织，这样在我们一一对应的时候，我们按照我们构造的`train_index`就可以刚好使我们的负样本数据达到$40000$的量，将我们的训练数据进行了扩充。`test`过程同理。

### 构建$graph$

```python
g0=sparse.csr_matrix(utils.build_tx_graph()).astype(np.float32)
```

```python
def build_tx_graph():
    start = time.clock()
    file=open('./data/txt_graph.txt','r+')
    line=file.readline().strip(' \n')
    graph=np.zeros((10055,10055),dtype=np.int)
    y=0
    while line:
        new_col=np.array(line.split(' '))
        for i in new_col:
            graph[y][int(i)]=1
        y=y+1
        line=file.readline().strip('\n')
    end=time.clock()
    print('build txt graph cost time:',(end-start)*1000,'ms')
    return graph
```

其中我们的`txt_graph.txt`中的内容是以下这种形式：

```
0 5816 4729 3713 6751 1990 9782 9907 7909
1 9487 5246 6622 4207 227 574 3949 8913
2 8291 1262 2887 4640 5541 8918 7410 9340
3 165 8515 2681 9124 736 2403 3895 3417
4 3765 5937 9324 112 958 2657 8719 7043
5 1816 697 654 9184 1040 6101 2617 1173
6 1177 8919 1446 5540 1394 2482 7781 9168
7 6666 2556 6823 4317 1440 7112 3093 566
8 3629 2880 9035 6654 7935 1738 5402 1742
9 9241 1394 1900 8919 1544 1620 1241 7300
10 2974 1613 153 4125 4566 3845 1980 7834
......
10046 4427 7420 9154 9750 7534 9449 7476 4044
10047 6583 837 8504 6014 3055 5437 9741 5586
10048 2842 9931 9666 4584 9381 3022 1746 3490
10049 7596 9253 9250 6712 8423 66 7280 8272
10050 6115 527 82 4372 1085 9355 7508 8469
10051 4487 6779 3670 2232 1932 6290 2279 6941
10052 4280 5767 9366 5373 7002 1392 4347 9657
10053 8269 9257 159 9355 1085 7277 9862 7588
10054 2795 920 3506 9676 9245 9057 7346 5263
```

## 运行模型

```python
model = models_siamese.siamese_cgcnn_cor(L0,L1, **params)
```

我们来看`model_siamese`中的`siamese_cgcnn_cor`类（`siamese_cgcnn_cor`是继承自`cgcnn`这个类，`cgcnn`这个类又是继承自`base_model`这个类）。

### 创建图表

```python
 def build_graph(self, M_0,M_1):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Inputs.
            with tf.name_scope('inputs'):
                self.ph_data1 = tf.placeholder(tf.float32, (self.batch_size, 4120,self.input_features0), 'data1')
                self.ph_data2 = tf.placeholder(tf.float32, (self.batch_size, M_1, self.input_features1), 'data2')
                self.ph_labels = tf.placeholder(tf.int32, (self.batch_size), 'labels')
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')

            # Model.
            op_logits  = self.inference(self.ph_data1,self.ph_data2,self.ph_dropout)
            self.op_loss, self.op_loss_average = self.loss(op_logits,self.ph_labels,self.regularization)
            self.op_train = self.training(self.op_loss, self.learning_rate,
                                          self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = self.prediction(op_logits)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()

            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)

        self.graph.finalize()
```

此$graph$即为$tensorflow$中的$graph$。

首先我们为输入的数据设置占位符，然后在模型建立阶段，我们$logit$为我们的推理环节$inference$的返回值。

#### 问题

- 什么是$logits$？

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

我们要做$LSTM$的模型，只需要将原来的$neuron$替换成$LSTM\ cell​$。 

![1537285420474](https://github.com/paradoxtown/paradoxtown.github.io/blob/master/img/complex_lstm_cell.png?raw=true)

比较复杂版本的$LSTM$。

现在有简化版的$LSTM$是$GRU$，只有两个$gate​$，参数减少了，但是效果差不多。