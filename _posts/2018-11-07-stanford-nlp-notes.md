---
layout:     post
title:      stanford nlp notes
subtitle:   winter 2017
date:       2018-11-07
author:     paradox
header-img: img/stanford.jpg
catalog: true
mathjax: true
tags: 
---

## Lecture2: Word Vectors Representations

skip-gram:

for each word t = 1... T, predict surrounding words in a window of 'radius' m of every word.

objective function: maximize the **probability of any context word** given the current center word:

$$
J'(\theta) = \prod _{t = 1} ^T \prod _{- m \leq j \leq m(j\neq0)} {\rm p}(w_{t+j}|w_t; \theta) \tag{1}
$$

$$
J(\theta)=-\frac{1}{T}\sum \limits _{t = 1} ^{T} \sum \limits _{-m\leq j\leq m(j\neq0)} \log {\rm p}(w_{t + j} | w_{t}) \tag{2}
$$

Where $\theta$ represents all variables(parameters) we will optimize.

So the formula(2) is a transformation form of formula(1), and it make formula(1) be a negative log likelihood. 

loss function = cost function = objective function.

the simplest first formulation is

$$
{\rm p}(o|c) = \frac{\exp(u_o^{\rm T}v_c)}{\sum\limits _{w = 1} ^{\rm v} \exp(u_w^{\rm T} v_c)} \tag{3}
$$

o is the outside(or output) word index, c is the center word index, $v_c$ and $u_o$ are 'center' and 'outside' vectors of indices c and o. V means the scale of vocabulary.

soft max using word c to obtain probability of word o.

we start computing deflector about $v_c$ for choosing how to change this vector, minimizing the objective function(2). 
$$
\frac{\partial}{\partial v_c} \log \frac{\exp u_o^{\rm T}v_c}{\sum _{w = 1} ^{\rm V} \exp(u_w ^{\rm T} v_c)} \tag{4}
$$

by doing minus, we get

$$
\frac {\partial}{\partial v_c} \log \exp(u_o ^{\rm T} v_c) - \frac{\partial}{\partial v_c} \log\sum\limits _{w =  1} ^{\rm V}\exp(u_w ^{\rm T} v_c) \tag{5}
$$

so, we can take this formula apart into tow parts.

we can simplify the first part to the following

$$
\frac{\partial}{\partial v_c} u_o^{\rm T} v_c = u_o \tag{6}
$$

then, take the second part into consideration, we can use chain rule to simplify the formula following

$$
\begin{split}
&\frac{\partial}{\partial v_c} \log \sum\limits _{w = 1} ^{\rm V} \exp(u_w ^{\rm T} v_c)\\
&= \frac{1}{\sum\limits_{w = 1} ^{\rm V}\exp(u_w ^{\rm T}v_c)} \frac{\partial}{\partial v_c}
\sum\limits _{x=1} ^{\rm V}\exp(u_x^{\rm T}{v_c})\\
&= \frac{1}{\sum\limits_{w = 1} ^{\rm V}\exp(u_w ^{\rm T}v_c)} \sum\limits _{x=1}^{\rm V} \frac{\partial}{\partial v_c} \exp(u_x^{\rm T}v_c)\\
&= \frac{1}{\sum\limits_{w = 1} ^{\rm V}\exp(u_w ^{\rm T}v_c)} \sum\limits _{x=1} ^{\rm V} \exp(u_x^{\rm T}v_c) \frac{\partial}{\partial v_c}u_x^{\rm T}v_c\\
&= \frac{1}{\sum\limits_{w = 1} ^{\rm V}\exp(u_w ^{\rm T}v_c)} \sum\limits _{x=1} ^{\rm V} \exp(u_x^{\rm T}v_c) u_x
\end{split}
$$

this equation is so likely to soft max.

$$
\sum\limits ^{\rm V} _{x = 1} \frac{\exp(u_x^{\rm T}v_c)}{\sum\limits_{w=1}^{\rm V} \exp(u_w^{\rm T})v_c} u_x \tag{7}
$$

so, the final form of formula(5) is

$$
u_o - \sum\limits ^{\rm V} _{x = 1} {\rm p}(x|c) u_x \tag{8}
$$

which $u_o$ is what we **observed** and the second part of this formula means sum of each word's **expectation**.

We update the model by update the matrix U.

Update equation(in matrix notation):
$$
\theta^{new} = \theta^{old} - \alpha\nabla _{\theta} J(\theta) \tag{9}
$$
Update equation(for single parameter):
$$
\theta ^{new} _{j} = \theta ^{old} _{j} - \alpha \frac{\partial}{\partial \theta _{j} ^{old}}J(\theta) \tag{10}
$$
Algorithm:

```python
while True:
    theta_grad = evaluate_gradient(J, corpus, theta)
    theta = theta - alpha * theta_grad
```

However, we know the $J(\theta)$ is a function of **all windows** in corpus, so it is expensive to compute the value of $\nabla _\theta J(\theta) $.

The solution is Stochastic(random) gradient descent (SGD): repeatedly sample windows, and update after each one.

Algorithm:

```python
while True:
    window = sample_window(corpus)
    theta_grad = evaluate_gradient(J, window, theta)
    theta = theta - alpha * theta_grad
```

## Lecture3: Glove: Global Vectors for Word Representation

finish word2vec

what does word2vec capture?

how could we capture this essence more effectively?

how can we analyze word vectors?

How to reduce vectors dimension. We have a very large co-occurrence matrix.

We just use simple SVD, singular value decomposition of co-occurrence matrix.

combining the best of both worlds, we make glove. The objective function of glove is: 
$$
J(\theta) = \frac{1}{2} \sum\limits _{i, j = 1} ^W f(P_{i,j})(u_i^{\rm T}v_j - \log  P_{ij})^2 \tag{11}
$$

$P _{i, j}$ means the co-occurrence matrix;

$f(P _{i, j})$ means the weighting function, the reason why we use this function is explained in paper:

> A main drawback to this model ($w^{T} _i \tilde{w} _k + b_i + \tilde{b}_k = \log(X_{ik})$) is that it weight all co-occurrences equally, even those that happen rarely or never. Such rare co-occurrences are noisy and carry less information than the more frequent ones -- yet even just the zero entries  account for $75-95\%$ of the data in $X$, depending on the vocabulary size and corpus.

$f(x)$ should be relatively small for large values of $x$, so that frequent co-occurrences are not over weighted.
$$
f(x) = 
\begin{cases}
(x/x_{max})^{\alpha}&\text{if } x < x_{\max}\\
1&\text{  otherwise .}
\end{cases} \tag{12}
$$

## Lecture4: Word Window Classification and Neural Networks

$$
-\log p(y|x) = - \log(\frac{\exp(f_y)}{\sum _{c=1} ^C \exp(f_c)}) \tag{1}
$$

we minimize the negtive log probability of that class is equal to maximize probability.

we assuming the $p = [0,...,0,1,0,...,0]$, so the cross entropy is the following
$$
H(p,q) = -\sum \limits _{c = 1} ^C p(c)\log q(c) \tag{2}
$$
the cross entropy loss function over full dataset is
$$
J(\theta) = \frac{1}{N} \sum\limits ^{N} _{i = 1} -\log (\frac{\exp(f_{y_i})}{\sum^C_{c=1}\exp(f_{c})}) \tag{3}
$$

$$
f_y = f_y(x) = W_yx=\sum\limits ^{d} _{j = 1} W_{y_j}x_j \tag{4}
$$

so the $f_y$ is a number. we will write $f$ in matrix notation: $f = Wx$.

really full loss function over any dataset includes regularization over all parameters $\theta$
$$
J(\theta) = \frac{1}{N} \sum\limits ^{N} _{i = 1} -\log (\frac{\exp(f_{y_i})}{\sum^C_{c=1}\exp(f_{c})}) + \lambda \sum \limits _{k} \theta ^2 _{k} \tag{5}
$$
try to encourage the model to keep all the weights as small as possible and as close as possible to zero.

**Tips**:

if you only have a small training data set, don't train the word vectors, just use the pretraining vectors and remain them unchanged.

if you have had a very large dataset, it may work better to train word vectors to the task.

$\hat {y} = p(y|x)$.

we use chain rule to derive our objective function.
$$
\frac{\partial}{\partial x} - \log softmax(f(y)) = \sum \limits _{c = 1} ^C -\frac{\partial \log softmax(f_y(x))}{\partial f_c}\cdot \frac{f_c(x)}{\partial x} \tag{6}
$$
$x$ is the entire window, 5 d-dimensional word vectors.

for the softmax part of the derivative:

first take the derivative wrt  $f_c$ when $c=y$  (the correct class), then take derivative wrt $f_c$ when $c \neq y$ (all the incorrect classes)

we can get the formula as following:
$$
\frac{\partial}{\partial f} - \log softmax(f_y) = \begin{bmatrix}
\hat{y}_1\\
\hat{y}_2\\
\vdots\\
\hat{y}_y - 1\\
\vdots\\
\hat{y}_C
\end{bmatrix} = [\hat{y} - t] = \delta \tag{7}
$$

$$
\sum \limits _{c = 1} ^C -\frac{\partial \log softmax(f_y(x))}{\partial f_c}\cdot \frac{f_c(x)}{\partial x} = \sum \limits _{c = 1} ^{C} \delta _c W_c ^T \tag{8}
$$

$$
\frac{\partial}{\partial x} - \log p(y|x) = \sum \limits _{c = 1} ^C \delta _c W_c ^T = W^T \delta \tag{9}
$$

$$
\nabla _x J = W^T \delta \in \R ^{5d} \tag{10}
$$

  