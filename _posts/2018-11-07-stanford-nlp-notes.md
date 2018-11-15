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
\frac {\partial}{\partial v_c} \log \exp(u_o ^{\rm T} v_c) - \frac{\partial}{\partial v_c} \sum\limits _{w =  1} ^{\rm V}\exp(u_w ^{\rm T} v_c) \tag{5}
$$

so, we can take this formula apart into tow parts.

we can simplify the first part to the following

$$
\frac{\partial}{\partial v_c} u_o^{\rm T} v_c = u_o \tag{6}
$$

then, take the second part into consideration, we can use chain rule to simplify the formula following

$$
\begin{split}
&\frac{\partial}{\partial v_c} \sum\limits _{w = 1} ^{\rm V} \exp(u_w ^{\rm T} v_c)\\
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

which $u_o$ is what we observed and the second part of this formula means sum of each word's expectation.

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

combining the best of both worlds, we make glove. The objective of glove is: 
$$
J(\theta) = \frac{1}{2} \sum\limits _{i, j = 1} ^W f(P_{i,j})(u_i^{\rm T}v_j - \log  P_{ij})^2 \tag{11}
$$

- $P _{i, j}$ means the co-occurrence matrix;

- $f(P _{i, j})$ means the weighting function, the reason why we use this function is explained in paper:

  >a main drawback to this model $w^{T} _i \tilde{w} _k + b_i + \tilde{b}_k = \log(X_{ik})$ is that it weight all co-occurrences equally, even those that happen rarely or never. Such rare co-occurrences are noisy and carry less information than the more frequent ones -- yet even just the zero entries  account for $75-95\%$ of the data in $X$, depending on the vocabulary size and corpus.

  $f(x)$ should be relatively small for large values of $x$, so that frequent co-occurrences are not over weighted.
  $$
  f(x) = 
  \begin{cases}
  (x/x_{max})^{\alpha}&\text{if } x < x_{\max}\\
  1&\text{  otherwise .}
  \end{cases} \tag{12}
  $$


