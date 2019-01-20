---
title: "Probability"
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

#Probability

Sample space Ω: The set of all the outcomes of a random experiment  
Event space F: A set whose elements A ∈ F (called events) are subsets of Ω
$$P ( A ) \geq 0 , \text { for all } A \in \mathcal { F }$$
$$P ( \Omega ) = 1$$
$$\sum _ { i = 1 } ^ { k } P \left( A _ { k } \right) = 1$$

The conditional probability of any event A given B is defined as

$$ P ( A | B ) \triangleq \frac { P ( A \cap B ) } { P ( B ) }$$

a random variable X is a function X: $\Omega \longrightarrow \mathbb { R } ^ { 2 }$

A cumulative distribution function (CDF) $F _ { X } ( x ) \triangleq P ( X \leq x )$

<p align="center">
<img src="https://imgur.com/io2lfu2.jpg">

</p>
<center>
Cumulative distribution function(CDF)
</center>

Probability mass functions: When a random variable X is a discrete then

$$p _ { X } ( x ) \triangleq P ( X = x )$$

$$0 \leq p _ { X } ( x ) \leq 1$$

Probability density functions -  For some continuous random variables, the cumulative distribution function F(x)is differentiable everywhere.
$$ f _ { X } ( x ) \triangleq \frac { d F _ { X } ( x ) } { d x }$$
$$\int _ { - \infty } ^ { \infty } f _ { X } ( x ) = 1$$

#Expectation
If X is a discrete random variable with PMF $p _ { X } ( x ) \text { and } g : \mathbb { R } \longrightarrow \mathbb { R }$, then expected value
$$ E [ g ( X ) ] \triangleq \sum _ { x \in V a l ( X ) } g ( x ) p _ { X } ( x )$$

For continuous randon variable

$$E [ g ( X ) ] \triangleq \int _ { - \infty } ^ { \infty } g ( x ) f _ { X } ( x ) d x$$

This is also known as the mean of the random variable X

#Variance

The variance of a random variable X is a measure of how concentrated the distribution of a random variable X is around its mean

$$\operatorname { Var } [ X ] \triangleq E \left[ ( X - E ( X ) ) ^ { 2 } \right]$$

# Some common Discrete random variables

X ~ Binomial(n,p) (where $0 \leq p \leq 1$ ): the number of heads in n independent flips of a coin with heads probability p.
$$ p ( x ) = \left( \begin{array} { l } { n } \\ { x } \end{array} \right) p ^ { x } ( 1 - p ) ^ { n - x } $$

X ~ Poisson($\lambda$) (where $\lambda > 0$): a probability distribution over the nonnegative integers used for modeling the frequency of rare events
$$ p ( x ) = e ^ { - \lambda } \frac { \lambda ^ { x } } { x ! }$$

# some common continuous random variable

1. X ~ Uniform(a,b) (where a < b): equal probability density to every value between a and b on the real line.
$$ f ( x ) = \left\{ \begin{array} { l l } { \frac { 1 } { b - a } } & { \text { if } a \leq x \leq b } \\ { 0 } & { \text { otherwise } } \end{array} \right.$$
2. $X \sim \operatorname { Normal } \left( \mu , \sigma ^ { 2 } \right)$ also known as the Gaussian distribution
$$f ( x ) = \frac { 1 } { \sqrt { 2 \pi } \sigma } e ^ { - \frac { 1 } { 2 \sigma ^ { 2 } } ( x - \mu ) ^ { 2 } }$$

<p align="center">
<img src="https://imgur.com/dw7Bgj1.jpg">

</p>

<center>
Gaussian distribution
</center>
<p align="center">
<img src="https://imgur.com/3lv2qCQ.jpg">

</p>

<center>
Uniform distribution
</center>
<p align="center">
<img src="https://imgur.com/2piWiKy.jpg">

</p>

<center>
Mean and Variance of common probability functions
</center>

#Joint and marginal probability mass functions

joint probability mass function  $p _ { X Y } ( x , y ) = P ( X = x , Y = y )$

marginal probability mass function $p _ { X } ( x ) = \sum _ { y } p _ { X Y } ( x , y )$

joint probability density function $f _ { X Y } ( x , y ) = \frac { \partial ^ { 2 } F _ { X Y } ( x , y ) } { \partial x \partial y }$

marginal probability density function $f _ { X } ( x ) = \int _ { - \infty } ^ { \infty } f _ { X Y } ( x , y ) d y$

Conditional PMF $p _ { Y | X } ( y | x ) = \frac { p _ { X Y } ( x , y ) } { p _ { X } ( x ) }$

Conditional PDF $f _ { Y | X } ( y | x ) = \frac { f _ { X Y } ( x , y ) } { f _ { X } ( x ) }$

Two random variables X and Y are independent if $p _ { X Y } ( x , y ) = p _ { X } ( x ) p _ { Y } ( y )$ for discrete case and  $F _ { X Y } ( x , y ) = F _ { X } ( x ) F _ { Y } ( y )$ for continuous case

Chain Rule
$f \left( x _ { 1 } , x _ { 2 } , \ldots , x _ { n } \right)$  $= f \left( x _ { 1 } \right) \prod _ { i = 2 } ^ { n } f \left( x _ { i } | x _ { 1 } , \ldots , x _ { i - 1 } \right)$

#Bayes Rule
Discrete Case $P _ { Y | X } ( y | x ) = \frac { P _ { X Y } ( x , y ) } { P _ { X } ( x ) } = \frac { P _ { X | Y } ( x | y ) P _ { Y } ( y ) } { \sum _ { y ^ { \prime } \in V a l ( Y ) } P _ { X | Y } ( x | y ^ { \prime } ) P _ { Y } \left( y ^ { \prime } \right) }$

Continuous Case $f _ { Y | X } ( y | x ) = \frac { f _ { X Y } ( x , y ) } { f _ { X } ( x ) } = \frac { f _ { X | Y } ( x | y ) f _ { Y } ( y ) } { \int _ { - \infty } ^ { \infty } f _ { X | Y } ( x | y ^ { \prime } ) f _ { Y } \left( y ^ { \prime } \right) d y ^ { \prime } }$

To study the relationship of two random variables with each other we use the covariance
$$\operatorname { Cov } [ X , Y ] \triangleq E [ ( X - E [ X ] ) ( Y - E [ Y ] ) ]$$

Covariance matrix: is the square matrix whose entries are given by $\Sigma _ { i j } = \operatorname { Cov } \left[ X _ { i } , X _ { j } \right]$

Normalizing the covariance gives the correlation
$$ \rho ( X , Y ) = \frac { \operatorname { Cov } ( X , Y ) } { \sqrt { \operatorname { Var } ( X ) \operatorname { Var } ( Y ) } } $$

Correlation also measures the linear relationship between two variables, but unlike covariance always lies between−1 and 1

Multivariate Gaussian distribution

$$f _ { X _ { 1 } , X _ { 2 } , \ldots , X _ { n } } \left( x _ { 1 } , x _ { 2 } , \ldots , x _ { n } ; \mu , \Sigma \right) = \frac { 1 } { ( 2 \pi ) ^ { n / 2 } | \Sigma | ^ { 1 / 2 } } \exp \left( - \frac { 1 } { 2 } ( x - \mu ) ^ { T } \Sigma ^ { - 1 } ( x - \mu ) \right)$$


Maximum likelihood estimation(MLE)

MLE is to choose parameters that “explain” the data best by maximizing the probability/density of the data we’ve seen as a function of the parameters.

$$\hat { \theta } _ { \mathrm { MLE } } = \underset { \theta } { \arg \max } \mathcal { L } ( \theta )$$
where $\mathcal { L }$ is the likelihood function.
$$\mathcal { L } ( \theta ) = p \left( x _ { 1 } , \ldots , x _ { n } ; \theta \right)$$

It is usually convenient to take logs, giving rise to the log-likelihood.
$$\log \mathcal { L } ( \theta ) = \sum _ { i = 1 } ^ { n } \log p \left( x _ { i } ; \theta \right)$$

Maximum a posteriori estimation(MAP)

We  assume  that  the  parameters  are  a  random  variable,  and  we  specify  a  prior distribution p(θ).

Using Bayes rule $p ( \theta | x _ { 1 } , \ldots , x _ { n } ) \propto p ( \theta ) p \left( x _ { 1 } , \ldots , x _ { n } | \theta \right)$

We can ignore the Normalizing constant as it does not affect the MAP estimate.
$$ \hat { \theta } _ { \mathrm { MAP } } = \underset { \theta } { \arg \max } p ( \theta ) p \left( x _ { 1 } , \ldots , x _ { n } | \theta \right)$$

If the observations are iid (independent and identically distributed) then
$$\hat { \theta } _ { \mathrm { MAP } } = \underset { 0 } { \arg \max } \left( \log p ( \theta ) + \sum _ { i = 1 } ^ { n } \log p \left( x _ { i } | \theta \right) \right)$$



# Information Theory  

We can quantify the amount of uncertainty in an entire probability distribution using the Shannon entropy of an event X = x

$$ H ( \mathrm { x } ) = \mathbb { E } _ { \mathrm { x } \sim P } [ I ( x ) ] = - \mathbb { E } _ { \mathrm { x } \sim P } [ \log P ( x ) ] $$


If we have two separate probability distributions P (x) and Q (x) over the same
random variable x, we can measure how different these two distributions are using
the Kullback-Leibler (KL) divergence:

$$ D _ { \mathrm { KL } } ( P \| Q ) = \mathbb { E } _ { \mathrm { x } \sim P } \left[ \log \frac { P ( x ) } { Q ( x ) } \right] = \mathbb { E } _ { \mathrm { x } \sim P } [ \log P ( x ) - \log Q ( x ) ] $$

The KL divergence has many useful properties, most notably that it is nonnegative.
The KL divergence is 0 if and only if P and Q are the same distribution in
the case of discrete variables, or equal “almost everywhere” in the case of continuous variables

A quantity that is closely related to the KL divergence is the cross-entropy

$$ H ( P , Q ) = - \mathbb { E } _ { \mathrm { x } \sim P } \log Q ( x )$$

Minimizing the cross-entropy with respect to Q is equivalent to minimizing the
KL divergence,

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
