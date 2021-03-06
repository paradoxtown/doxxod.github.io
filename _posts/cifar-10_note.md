---
layout:     post
title:      cifar-10 note
subtitle:   Learn from Tensorflow
date:       2017-11-15
author:     paradox
header-img: img/post_3.jpg
catalog: true
mathjax: true
tags:
    - Machine Learning 
    - Tensorflow
    - cifar-10
---

### 零碎的知识点

#### 函数总结：
- `tf.Variable()`:是用来存储模型参数，与存储数据的$tensor$不同， $tensor$一旦被使用就会消失.
- `tf.reduce_mean`:是缩减为都的计算均值.
- `tf.argmax()`:是寻找$tensor$中的最大值元素的序号，此例中用来判断类别.
- `tf.cast(, type)`:是用来数据类型转换
- `tf.slice(input_, begin, size, name=None)`:意为从`begin`(位置）开始,截取`size`(长度).
- `tf.reshape(t, [x, y])`:将数据$t$分割成$x$块，每块$y$个元素.
- `tf.concat(values, axis, name = 'concat')`: 
#### `variable_scope`和`name_scope`的区别
我们只需要搞清楚`tf.Variable()`和`tf.get_variable()`这两个函数的区别.这里指先说他们的一个区别：

```python
import tensorflow as tf

v = tf.Variable([2, 2, 3, 32], name = 'weights')
with tf.variable_scope('variable_scope'):
    v1 = tf.get_variable('weights', [2, 2, 3, 32])

with tf.name_scpoe('name_scope'):
    v2 = tf.get_variable('weights', [2, 2, 3, 32])

print v.name
print v1.name
print v2.name
```

输出结果为：

```python
weight:0
variable_scope/weights:0
weights_1:0
```

可以看出，`tf.get_variable`是不受`name_scope`影响的，而两者均受`tf.variable_scope`的影响.同时，如果我们用的`variable()`是允许同名的，他会自动给你加上下划线并且编号，但是`weight`和`wieght_1`并不是相同的变量.（PS：在调用`variable_scope()`的时候，会自动调用`name_scope()`).

### 概念：
- $Learning\ Rate:$
  学习率决定了权值更新的速度，设置的太大会使结果超过最优值，太小会使下降速度过慢.
- $Weight\ Decay:$
  权重衰减值，在实际应用中，为了避免网络的过拟合，必须对价值函数$(Cost\ Function)$加入一些正则项.权值衰减的使用并不是为了提高收敛精确度，也不是为了提高收敛速度，其最终目的是防止过拟合.在损失函数中，权值衰减是放在正则项$(Regularization)$前面的一个系数，正则项一般指示模型对的复杂度，所以权值衰减的作用是调节模型复杂度对损失函数的影响，若权值衰减很大，则复杂的模型损失函数的值也就越大.
- $Data\ Aufmentation:$
  以为数据增强，这里将32*32的图片通过各种变换得到增强后的数据.
- $LRN(Local\ Respinse\ Normalization):$
  意为局部响应归一化.具体作用就是把输出只拉回到中间的线性区，从而减轻梯度的消失.

`CIFAR-10`模型中完整的训练图中包含约765个操作，但是通过下面的模块来构造训练图可以最大限度的提高代码的复用率：

- <b>模型输入</b>：这里相较于之前主要增加了`distoreted_input()`的操作，而这里的`input()`是用来给评估函数进行输入的;
- <b>模型预测</b>:包括`inference()`等一些操作，用于进行统计计算，比如在提供的图像进行分类;
- <b>模型训练</b>：包括`loss()`和`train()`等一些操作，用于计算损失、计算梯度、进行变量更新以及呈现最终结果.

## 模型输入
```python
def read_cifar10(filename_queue):
    """
    建议：如果你想并行地从N条路读取数据，调用这个函数N次.
    这回为你提供N个独立的从不同文件和位置的Reader，以获得更大的乱序.
    参数： filename_queuw, 一个等待被读取的文件名字符串队列.
    Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
    """
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    #result 记录了存储要输出的数据.
    # input format.
    label_bytes = 1  # 2 for CIFAR-100??
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth

    record_bytes = label_bytes + image_bytes
    reader = tf.FixedLengthRecordReader(record_bytes = record_bytes)
    #生成一个阅读操作器reader.
    reader.key, value = reader.read(filename_queue)
    #给reader传入I/O类型的参数filename_queue，返回一个$tensor$

    record_bytes = tf.decode_raw(value, tf.unit8)
    #将一个字符串转换成长度为record_bytes的unit8形式的向量.

    result.label = tf.cast(
        tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
    #将张量record_bytes中额第一个字符——标签取出，转换成int32类型，并赋值.

    depth_major = tf.reshape(tf.slice(record_bytes,
    [label_bytes], [image_bytes]), [result.depth, result.hight, result.width])
    #将image数据分成三部分.

    result.unit8image = tf.tanspose(depth_major, [1, 2, 0])
    #将其转换为(height, weight, depth)的类型.
```
最终返回的`result`是一个$cifar10\ Record$对象，其中包含了：
- $height, width, depth$
- $key$:描述$filename$和$record number$的$tensor$
- $label$:一个32位整型的$tensor$
- $unit8image$:图片的数据

`_generate_image_and_label_batch()`使用队列来生成批次的.
![cifar-10](https://github.com/paradoxtown/paradoxtown.github.io/raw/master/img/cifar-10.gif)
```python
def _generate_image_and_label_batch(image, label, min_quue_examples,
                                    batch_size):
    num_preprocess_threads = 16
    #shuffle乱序读入,大致原理是，将样本的tensor按顺序压倒一个队列中（random shuffle queue），
    #知道样本个数达到capacity（容量），然后需要的时候随机从中取出batch_size个样本
    image, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size = batch_size,
            num_threads = num_preprocess_threads,
            capacity = min_queue_examples + 3 * batch_size,
            min_after_dequeue = min_queue_examples)
    
    tf.image_summary('image', image)

    return image, tf.reshape(label_batch, [batch_size])
```
以下是训练数据输入的总函数
```python
def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
               for i in xrange(1, 6)]
  for f in filenames:
    if not gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # 产生一个我们要读取的文件的队列，并传入read_cifar10函数中
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # 图片会被同意裁剪到24*24像素大小，裁剪中央区域用于评估或随机裁剪用于训练.
  distorted_image = tf.image.random_crop(reshaped_image, [height, width])

  # 随机水平翻转图片
  distorted_image = tf.image.random_flip_left_right(distorted_image)

  # Because these operations are not commutative（不可交换的）, consider randomizing
  # randomize the order their operation.
  # 随机变换图片的亮度
  distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
  # 随舰变换图片的对比度
  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(distorted_image)

  # Ensure that the random shuffling（搅浑） has good mixing properties（性能）.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size)
```

## 模型预测
![5a0d94a86194133b91000000](https://github.com/paradoxtown/paradoxtown.github.io/raw/master/img/5a0d94a86194133b91000000.png)
这一步中与之前的变化就是：由于我们这次使用的是彩色的图片，因此我们需要传入参个通道的值(RGB)，因此把传入通道数改成3，其他部分和前面训练$mnist$数据集基本是一样的.
```python
def inference(images):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    dim = 1
    for d in pool2.get_shape()[1:].as_list():
      dim *= d
    reshape = tf.reshape(pool2, [FLAGS.batch_size, dim])

    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear
```

## 模型训练
```python
def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Reshape the labels into a dense Tensor of
  # shape [batch_size, NUM_CLASSES].
  sparse_labels = tf.reshape(labels, [FLAGS.batch_size, 1])
  indices = tf.reshape(tf.range(FLAGS.batch_size), [FLAGS.batch_size, 1])
  concated = tf.concat(1, [indices, sparse_labels])
  dense_labels = tf.sparse_to_dense(concated,
                                    [FLAGS.batch_size, NUM_CLASSES],
                                    1.0, 0.0)

  # Calculate the average cross entropy loss across the batch.
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      logits, dense_labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')
```

定义`train()`操作.
```python
def train(total_loss, global_step):
    """创建一个优化并应用到每一个可以被训练的变量上，为所有可以训练的变量添加移动平均数.
    输入参数：
    total loss: from loss()
    global_step: 一个记录了训练步数的整数.
    返回：
    train_op: 一个操作
    """
    #影响学习率的变量
    num_batch_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_step = int(num_batches_per_epoch + NUM_EPOCH_PER_DECAY)

    #随着steps成倍地衰减学习率
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase = True)
    tf.scalar_summary('learning_rate', lr)

    #生成损失的平均移动，并且整合成summary
    apply_gradient_op = opt.apply_gradients(grads, global_step = global_step)

    #为可训练的变脸添加直方图
    for var in tf.trainable_variavles():
        tf.histogram_summary(var.op.name, var)

     for grad, var in grads:
    if grad:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op
```
函数`expinential_decay()`:

- `INITAIL_LEARNING_RATE`: 初始学习率；
- `global_step:`: 全局$step$数，每一个$step$对应一次$batch$
- `decay_steps`: $learning_rate$更新的$step$周期，更新一次$learning\ rate$的值
- `decay_rate`: 指数衰减参数
- `staircase`: 是否阶梯型更新$learning\ rate$， 也就是$\frac{global\_ steps}{decay\_ steps}$

$$decay\_learning\_rate = learning\_rate * decay\_rate ^{\frac{global\_steps}{decay\_steps}}$$
##开始合并操作并执行训练
```python
def train():
    #train CIFAR-10 for number of steps
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable = False)

        #从cifar10获得images，和labels
        images, labels = cifar10.distorted_inputs()

        #建立一个计算从推理模型中获得logits预测的图表
        logits = cifar10.inference(images)

        #计算损失
        loss = cifar10.loss(logits, labels)

        #创建一个训练一个批次的样例并用来更新参数的模型的图表
        train_op = cifar10.tain(loss, gobal_step)

        #创建一个saver
        saver = tf.train.Saver(tf.all_variabel())

        #创建summary
        summary_op = tf.merge_all_summaries()

        #创建一个初始化操作
        init = tf.initialize_all_variables()

        #开始在图标上运行操作
        sess = tf.Session(config = tf.ConfigProto(
            log_device_placement = FLAGS.log_device_placement))
        sess.run(init)

        #开始运行队列
        tf.train.start_queue_runners(sess = sess)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                                graph_def = sess.graph_ded)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if strp % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,
                                 examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
```
