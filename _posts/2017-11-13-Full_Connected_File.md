---
layout:     post
title:      Full Connected File
subtitle:   
date:       2017-11-13
author:     paradox
header-img: img/post_4.jpg
catalog: true
tags:
mathjax: true
    - Machine Learning 
    - Tensorflow
    - Full Connected File
    - Neural Networks
---

# 全连接层文件
在为数据创建占位符之后，就可以运行`mnist.py`文件，经过三个阶段的模式函数操作：`inference(), loss(), training()`，可以构建图表.

- <b>question_0:</b>所以构建的图表有什么用？
  图表是一种对计算的抽象描述，图中的节点被称为$op(operation)$，一个$op$获得零个或多个张量执行计算，并产生零个或多个张量.
- <b>question_1:</b>为什么构建多个连接层，也就是为什么构建多感知器？
  多感知器可以学习非线性函数.

##构建图表
###推理环节$(inference)$
定义内容在`mnist.py`中.
```py3
def inference(images, hidden1_units, hidden2_units):
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
                tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                    stddev = 1.0 / math.sqrt(float(IMAGE_PIXEELS))),
                name = 'weights')
        biases = tf.Variable(tf.zeros([hidden1_units]), name = 'biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    with tf.name_scope('hidden2'):
        weights = tf.Variable(
                tf.truncated_normal([hidden1_units, hidden2_units],
                                    stddev = 1.0 / math.sqrt(float(hidden_units))),
                name = 'weights')
        biases = tf.Variabel(tf.zeros([hidden2_units]), name = 'biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    with tf.name_scope('softmaz_linear'):
        weights = tf.Variable(
                tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                    stddev = 1.0 / math.sqrt(float(hidden2_units))),
                name = 'weights')
        biases = tf.Variable(tf.zero([NUM_CLASSES]), name = 'biases')

        logits = tf.matmul(hidden2, weights) + biases
    return logits                
```
上面这个函数就是`inference()`过程.

- <b>quection_2:</b>什么是`truncated_normal`?
  通过`tf.truncated_normal`函数初始化权重变量.该函数将会根据所得到的均值和标准差，生成一个随机分布.
- <b>question_3:</b>为什么生成一个随机分布？（待解决）

### 损失$(loss)$
```py3
def loss(logits, lables):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        lablels = labels, logits = logits, name = 'xentropy')
    return tf.reducy_mean(cross_entropy, name = 'xentropy_mean')
```
返回该批次的平均损失.

### 训练$(training)$
```py3
def training(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #学习效率应用梯度下降法
    global_step = tf.Variable(0, name = 'global_step', trainable = False)#初始化
    #生成变量用于保存全局训练步骤(global training step)的数值，并使用minimizer()函数
    #更新系统中的三角权重(triangle weights), 增加全局步骤的操作 
    #根据惯例, 这个操作被称为train_op, 是Tensorflow会话(session)诱发一个完整训练步骤所必须运行的操作.
    train_op = optimizer.minimize(loss, global_step = global_step)
    return train_op
```
返回训练操作输出结果的张量.
至此我们的图表已经基本构建完成.

### 评估$(evauation)$
```py3
def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    #tf.nn.in_top_k操作指的是如果在k个最有可能的预测中可以发现真的标签，
    #那么这个操作就会将模型输出标记为正确.我们将k设置为1，
    #也就是只有在预测为真的标签是，才判定它是正确的.
    return tf.reduce_sum(tf.cast(correct, tf.int32))
```

## 训练模型
```py3
def run_training():
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

    with tf.Graph().as_default():
    image_placeholder, labels_placeholder = placeholder_input(
        FLAGS.batch_size)

    #创建一个图表用来计算来自推理模型的预测值
    logits = mnist.inference(images_placeholder,
                             FLAGS.hidden1,
                             FLAGS.hidden2)
    #将计算损失的操作添加进图表
    loss = minst.loss(logits, labels_placeholder)

    #将计算和应用的梯度添加进图表
    train_op = mnist.training(loss, FLAGS.learning_rate)

    #将评估（比较得到的预测和label）操作添加到图表中去
    eval_correct = mnist.evaluation(logits, labels_placeholder)

    summary = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    #将训练的检查点保存（断点？）
    saver = tf.train.Saver()

    sess = tf.Session()

    #将一个summary和图表的summary_writer实例化
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    #先运行初始化操作节点
    sess.run(init)

    #开始进行循环训练
    for step in xrange(FLAGS.max_steps):
        start_time = time.time()

        #用真实的集合中的image和labels来填充一个feed_dict
        feed_dict = fill_feed_dict(data_sets.train,
                                    image_placeholder,
                                    labels_placeholder)

        #运行一次模型，返回值是被激活了的`train_op`（被抛弃？）和`loss`Op
        #为了检查我们的Ops值或者变量的值，我们也许需要将他们装进一个给sess.sun的list
        #而且张量将会被返回到一个元组中
        #train_op不会产生输出，所以会被抛弃
        #如果模型在训练中出现偏差，loss Tensor的值可能变成NaN，所以我们要获取它的值并记录
        #怎么检查的？
        _, loss_value = sess.run([train_op, loss], 
                                 feed_dict = feed_dict)

        duration = time.time() - start_time

        if step % 100 == 0:
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))

            summary_str = sess.run(summary, feed_dict = feed_dict)
            summary_writer.add_summary(summay, feed_dict = feed_dict)
            summary_writer.flush()

        if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_file = os.path.join(FLAGS.log_die, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step = step)

            print('Training Data Eval')
            do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder)
            print('Validation Data Eval:')
            do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.validation)
            print('Test Data Eval:')
            do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.test)
```

### 评估模型&构建评估图图表&评估图标的输出
在上面的代码中我们可以看见`do_eval`函数被调用了三次，分别使用训练数据集、验证数据集和测试数据集对模型进行评估.其中`do_eval`的代码如下:
```py3
def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
 
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  #我们创建循环，往其中添加feed_dict，并在调用sess.run()函数时传入eval_correct操作，
  #目的是用 给定 的数据集评估模型
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    #true_count变量会累加所有in_top_k操作哦安定为正确的预测之和.
    #接下来，只需要将正确测试的总和除以例子的总数，就可以算出准确率了
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))
```
<b>在进入循环前</b>，我们应该先调用mnist.py文件中的`evaluation`函数，传入的`logits`和标签参数要与`loss`函数的一致.这样做是为了先构建`Eval`操作.
`eval_correct = mnist.evaluation(logits, labels_placeholder)`中
再更复杂的场景中，要先隔绝`data_sets.test`测试数据集，使用大量超参数优化调整$(hyper\ parameter\  tuning)$之后才进行检查.

另:`fill_feed_dict`函数会索要下一个批次batch_size的图像和标签.

