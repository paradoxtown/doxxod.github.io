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

## lecture2: word vector representations

skip-gram:

for each word t = 1... T, predict surrounding words in a window of 'radius' m of every word.

objective function: maximize the probability of any context word given the current center word:

$$
J'(\theta) = \prod _{t = 1} ^T \prod _{- m \leq j \leq m(j\neq0)} {\rm p}(w_{t+j}|w_t; \theta) \tag{1}
$$

$$
J(\theta)=-\frac{1}{T}\sum \limits _{t = 1} ^{T} \sum \limits _{-m\leq j\leq m(j\neq0)} \log {\rm p}(w_{t + j} | w_{t}) \tag{2}
$$

Where $\theta$ represents all variables(parameters) we will optimize.

So the formula(2) is a transformation form of formula(1), and it make formula(1) be a negative log likelihood. 

loss function = cost function = objective function.

for $p(w_{t+j} |w_{t})$ the simplest first formulation is

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
\sum\limits ^{\rm V} _{x = 1} \frac{\exp(u_x^{\rm T}v_c)}{\sum\limits_{w=1}{\rm V} \exp(u_w^{\rm T})v_c} u_x \tag{7}
$$

so, the final form of formula(5) is

$$
u_o - \sum\limits ^{\rm V} _{x = 1} {\rm p}(x|c) u_x \tag{8}
$$

which $u_o$ is what we observed and the second part of this formula means sum of each word's expectation.