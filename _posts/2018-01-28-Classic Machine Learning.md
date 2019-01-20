---
title: "Classic Machine Learning"
date: 2018-01-28
tags: [machine learning, data science, neural network]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Machine Learning, Perceptron, Data Science"
mathjax: "true"
---

# H1 Heading

## H2 Heading

### H3 Heading

Here's some basic text.

And here's some *italics*

Here's some **bold** text.

What about a [link](https://github.com/dataoptimal)?

Here's a bulleted list:
* First item
+ Second item
- Third item

Here's a numbered list:
1. First
2. Second
3. Third

Machine learning algorithms build a mathematical model of sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to perform the task  

* Supervised learning - Trained on data that contains both the inputs and the desired outputs. Supervised learning algorithms include classification and regression.  
* Unsupervised learning -Unsupervised learning algorithms take a set of data that contains only inputs, and find structure in the data  

#Linear regression

This is used for linear modelling of continuous outputs

$$y ( \mathrm { x } ) = \mathrm { w } ^ { T } \mathrm { x } + \epsilon$$

Where $\mathrm { w } ^ { T } \mathrm { x }$ is the scalar product of input x and models weights W. $\epsilon$ is the residual error between the predictions and the true value.

#Maximum likelihood estimation (least squares)
Assuming that the training examples are independent and identically distributed(iid)

$$ \hat { \boldsymbol { \theta } } \triangleq \arg \max _ { \boldsymbol { \theta } } \log p ( \mathcal { D } | \boldsymbol { \theta } )$$

 we can equivalently minimize the negative log likelihood
$$\mathrm { NLL } ( \theta ) \triangleq - \sum _ { i = 1 } ^ { N } \log p \left( y _ { i } | \mathrm { x } _ { i } , \theta \right)$$

 Here MLE is the one that reduces the sum of squared errors

$$\operatorname { RSS } ( \mathbf { w } ) \triangleq \sum _ { i = 1 } ^ { N } \left( y _ { i } - \mathbf { w } ^ { T } \mathbf { x } _ { i } \right) ^ { 2 }$$

Where RSS is residual sum of squares
$$\mathrm { NLL } ( \mathbf { w } ) = \frac { 1 } { 2 } ( \mathbf { y } - \mathbf { X } \mathbf { w } ) ^ { T } ( \mathbf { y } - \mathbf { X } \mathbf { w } ) = \frac { 1 } { 2 } \mathbf { w } ^ { T } \left( \mathbf { X } ^ { T } \mathbf { X } \right) \mathbf { w } - \mathbf { w } ^ { T } \left( \mathbf { X } ^ { T } \mathbf { y } \right)$$

 Gradient of this is given by
$$\mathbf { g } ( \mathbf { w } ) = \left[ \mathbf { X } ^ { T } \mathbf { X } \mathbf { w } - \mathbf { X } ^ { T } \mathbf { y } \right] = \sum _ { i = 1 } ^ { N } \mathbf { x } _ { i } \left( \mathbf { w } ^ { T } \mathbf { x } _ { i } - y _ { i } \right)$$

Equating to zero we get
$$\mathbf { X } ^ { T } \mathbf { X } \mathbf { w } = \mathbf { X } ^ { T } \mathbf { y }$$

$$\hat { \mathbf { w } } _ { O L S } = \left( \mathbf { X } ^ { T } \mathbf { X } \right) ^ { - 1 } \mathbf { X } ^ { T } \mathbf { y }$$

#Bayesian linear regression

Sometimes we want to compute the full posterior over w and $\sigma^2$. If we assume Gaussian prior and that the X is distributed in Gaussian then;

$$ \begin{aligned} p ( \mathbf { w } | \mathbf { X } , \mathbf { y } , \sigma ^ { 2 } ) & \propto \mathcal { N } ( \mathbf { w } | \mathbf { w } _ { 0 } , \mathbf { V } _ { 0 } ) \mathcal { N } ( \mathbf { y } | \mathbf { X } \mathbf { w } , \sigma ^ { 2 } \mathbf { I } _ { N } ) = \mathcal { N } ( \mathbf { w } | \mathbf { w } _ { N } , \mathbf { V } _ { N } ) \\ \mathbf { w } _ { N } & = \mathbf { V } _ { N } \mathbf { V } _ { 0 } ^ { - 1 } \mathbf { w } _ { 0 } + \frac { 1 } { \sigma ^ { 2 } } \mathbf { V } _ { N } \mathbf { X } ^ { T } \mathbf { y } \\ \mathbf { V } _ { N } ^ { - 1 } & = \mathbf { V } _ { 0 } ^ { - 1 } + \frac { 1 } { \sigma ^ { 2 } } \mathbf { X } ^ { T } \mathbf { X } \\ \mathbf { V } _ { N } & = \sigma ^ { 2 } \left( \sigma ^ { 2 } \mathbf { V } _ { 0 } ^ { - 1 } + \mathbf { X } ^ { T } \mathbf { X } \right) ^ { - 1 } \end{aligned}$$

If $\mathbf { w } _ { 0 } = 0 \text { and } \mathbf { V } _ { 0 } = \tau ^ { 2 } \mathbf { I }$ then the posterior mean reduces to ridge regression estimate.

#Logistic regression

Logistic regression corresponds to the following binary classification model
$$p ( y | \mathbf { x } , \mathbf { w } ) = \operatorname { Ber } ( y | \operatorname { sigm } \left( \mathbf { w } ^ { T } \mathbf { x } \right) )$$

The negative log-likelihood for logistic regression is given by
$$\begin{aligned} \mathrm { NLL } ( \mathbf { w } ) & = - \sum _ { i = 1 } ^ { N } \log \left[ \mu _ { i } ^ { \mathrm { I } \left( y _ { i } = 1 \right) } \times \left( 1 - \mu _ { i } \right) ^ { \mathrm { I } \left( y _ { i } = 0 \right) } \right] \\ & = - \sum _ { i = 1 } ^ { N } \left[ y _ { i } \log \mu _ { i } + \left( 1 - y _ { i } \right) \log \left( 1 - \mu _ { i } \right) \right] \end{aligned}$$
$$N L L ( \mathbf { w } ) = \sum _ { i = 1 } ^ { N } \log \left( 1 + \exp \left( - \tilde { y } _ { i } \mathbf { w } ^ { T } \mathbf { x } _ { i } \right) \right)$$

This is also called cross entropy error function.
The gradient is given by
$$\mathrm { g } = \frac { d } { d \mathrm { w } } f ( \mathbf { w } ) = \sum _ { i } \left( \mu _ { i } - y _ { i } \right) \mathbf { x } _ { i } = \mathbf { X } ^ { T } ( \boldsymbol { \mu } - \mathbf { y } )$$

We can use gradient descent to find the parameters

# Bayesian logistic regression

If we assume a Gaussian Prior along with the Bernoulli likelihood
$p ( \theta ) = \frac { 1 } { \sqrt { 2 \pi \sigma ^ { 2 } } } \exp \left( - \frac { 1 } { 2 \sigma ^ { 2 } } ( \theta - \mu ) ^ { \prime } ( \theta - \mu ) \right)$

The posterior is given by  $P ( \theta | D ) = \frac { P ( y | x , \theta ) P ( \theta ) } { P ( y | x ) }$

Where $P ( y | x ) = \int P ( y | x , \theta ) P ( \theta ) d \theta$. This is typically intractable for most problems.

We also want to predict y given a new x and the current data
$$P \left( y _ { n + 1 } | x _ { n + 1 } , D \right) = \int P \left( y _ { n + 1 } , \theta | x _ { n + 1 } , D \right) d \theta$$
$$= \int P \left( y _ { n + 1 } | \theta _ { 1 } x _ { m , D } \right) P ( \theta | x _ { m , p } ) d \theta$$

We can drop D as $\theta$ summarizes it.
$$= \int P \left( y _ { n + 1 } | \theta _ { 1 } x _ { n + 1 } \right) P ( \theta | D ) d \theta$$

Using Monte carlo estimates
$$\approx\frac { 1 }{N} \sum_{i=1}^{N}P ( y_{n+1} |\theta^i,x_{n+1} )$$

<p align="center">
<img src="https://imgur.com/U5OdrtB.jpg">

</p>

<center>
Bayesian Logistic regression
</center>



Python code block:
```python
    import numpy as np

    def test_function(x, y):
      z = np.sum(x,y)
      return z
```

R code block:
```r
library(tidyverse)
df <- read_csv("some_file.csv")
head(df)
```

Here's some inline code `x+y`.

Here's some math:

$$z=x+y$$

Latex Math

$$ \theta H = - \sum _ { x _ { = 0 } } ^ { 1 } \theta ^ { x } ( 1 - \theta ) ^ { 1 - x } \log \left[ \theta ^ { x } ( 1 - \theta ) ^ { 1 - x } \right] $$




<p align="center">
<img src="https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif">

</p>

<center>
*Fig. 2: The minimum dominating set of a graph*
</center>
