---
title: "Linear Algebra"
date: 2018-01-28
tags: [machine learning, data science, neural network]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Machine Learning, Perceptron, Data Science"
mathjax: "true"
---

Linear algebra provides a way of compactly representing and operating on sets of linear
equations. For example, consider the following system of equations:

$$ \begin{aligned} 4 x _ { 1 } - 5 x _ { 2 } & = - 13 \\ - 2 x _ { 1 } + 3 x _ { 2 } & = 9 \end{aligned} $$

$$A x = b $$

$$ A = \left[ \begin{array} { c c } { 4 } & { - 5 } \\ { - 2 } & { 3 } \end{array} \right] , \quad b = \left[ \begin{array} { c } { - 13 } \\ { 9 } \end{array} \right] $$

By  $A \in \mathbb { R } ^ { m \times n }$ we denote a matrix with m rows and n columns, where the entries of A are real numbers.

By $x \in \mathbb { R } ^ { n }$, we denote a vector with n entries. By convention, an n-dimensional vector is often thought of as a matrix with n rows and 1 column, known as a **column vector**.

The ith element of a vector x is denoted $x _ { i }$
$$ x = \left[ \begin{array} { c } { x _ { 1 } } \\ { x _ { 2 } } \\ { \vdots } \\ { x _ { n } } \end{array} \right] $$

We use the notation $a _ { i j } \left( \text { or } A _ { i j } , A _ { i , j } , \text { etc } \right)$ to denote the entry of A in the ith row and
jth column:

$$ A = \left[ \begin{array} { c c c c } { a _ { 11 } } & { a _ { 12 } } & { \cdots } & { a _ { 1 n } } \\ { a _ { 21 } } & { a _ { 22 } } & { \cdots } & { a _ { 2 n } } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { a _ { m 1 } } & { a _ { m 2 } } & { \cdots } & { a _ { m n } } \end{array} \right] $$

The product of two matrices $A \in \mathbb { R } ^ { m \times n } \text { and } B \in \mathbb { R } ^ { n \times p }$ is the matrix
where $C _ { i j } = \sum _ { k = 1 } ^ { n } A _ { i k } B _ { k j }$

in order for the matrix product to exist, the number of columns in A must equal the number of rows in B.

#inner product or dotproduct of the vectors

inner product or dot product of the vectors, is a real number given by

$$ x ^ { T } y \in \mathbb { R } = \left[ \begin{array} { c c c c } { x _ { 1 } } & { x _ { 2 } } & { \cdots } & { x _ { n } } \end{array} \right] \left[ \begin{array} { c } { y _ { 1 } } \\ { y _ { 2 } } \\ { \vdots } \\ { y _ { n } } \end{array} \right] = \sum _ { i = 1 } ^ { n } x _ { i } y _ { i } $$

Also $x ^ { T } y = y ^ { T } x$

The Outer product of vectors is given by
$$ \left( x y ^ { T } \right) _ { i j } = x _ { i } y _ { j } $$

In addition to this, it is useful to know a few basic properties of matrix multiplication at a higher level:

* Matrix multiplication is associative: (AB)C = A(BC)
* Matrix multiplication is distributive: A(B + C) = AB + AC
* Matrix multiplication is, in general, not commutative; that is, it can be the case that $A B \neq B A$


The identity matrix, denoted $I \in \mathbb { R } ^ { n \times n }$, is a square matrix with ones on the diagonal and zeros everywhere else.

For all $A \in \mathbb { R } ^ { m \times n }$ ;

The transpose of a matrix results from “flipping” the rows and columns.
$$ \left( A ^ { T } \right) _ { i j } = A _ { j i }$$
Transposes have the following properties
1. $\left( A ^ { T } \right) ^ { T } = A$
2. $( A B ) ^ { T } = B ^ { T } A ^ { T }$
3. $( A + B ) ^ { T } = A ^ { T } + B ^ { T }$

A square matrix $A \in \mathbb { R } ^ { n \times n }$ is symmetric if $A = A ^ { T }$

The trace of a square matrix $A \in \mathbb { R } ^ { n \times n }$ denoted tr(A) is the sum of diagonal elements in the matrix:
$$ \operatorname { tr } A = \sum _ { i = 1 } ^ { n } A _ { i i } $$

Trace has the following properties
1. $\operatorname { tr } A = \operatorname { tr } A ^ { T }$
2. $\operatorname { tr } ( A + B ) = \operatorname { tr } A + \operatorname { tr } B$
3. $\operatorname { tr } A B = \operatorname { tr } B A$
4. $\operatorname { tr } ( t A ) = t \operatorname { tr } A$

# Norms of Vector

A norm of a vector $\| x \|$ is the length of the vectors

The euclidean or the $\ell _ { 2 }$ norm is
$$\| x \| _ { 2 } = \sqrt { \sum _ { i = 1 } ^ { n } x _ { i } ^ { 2 } }$$

where $\| x \| _ { 2 } ^ { 2 } = x ^ { T } x$

In General the norm for a real number p ≥ 1 is
$$ \| x \| _ { p } = \left( \sum _ { i = 1 } ^ { n } \left| x _ { i } \right| ^ { p } \right) ^ { 1 / p }$$

Norms can also be defined for matrices, such as the Frobenius norm,
$$\| A \| _ { F } = \sqrt { \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { n } A _ { i j } ^ { 2 } } = \sqrt { \operatorname { tr } \left( A ^ { T } A \right) }$$

# Linear Independence and Rank
A set of vectors is said to be (linearly) independent if no vector can
be represented as a linear combination of the remaining vectors.
The column rank of a matrix is the size of the largest subset of columns of that constitute a linearly independent set.

For $A \in \mathbb { R } ^ { m \times n }$
$$ \operatorname { rank } ( A ) \leq \min ( m , n )$$
* If $\operatorname { rank } ( A ) = \min ( m , n )$ then A is said to be full Rank
* $\operatorname { rank } ( A ) = \operatorname { rank } \left( A ^ { T } \right)$


#The Inverse of Matrix

The inverse of a square matrix is the unique matrix such that
$$A ^ { - 1 } A = I = A A ^ { - 1 }$$

If a matrix does not have an inverse it is said to be non-invertible or singular.In order for a square matrix A to have an inverse , then A must be full rank.

Following are the properties of inverse
* $\left( A ^ { - 1 } \right) ^ { - 1 } = A$
* $( A B ) ^ { - 1 } = B ^ { - 1 } A ^ { - 1 }$
* $\left( A ^ { - 1 } \right) ^ { T } = \left( A ^ { T } \right) ^ { - 1 }$

#Orthogonal Matrices
Two vectors x, y are orthogonal if $x ^ { T } y = 0$ . A vector x i normalized if $\| x \| _ { 2 } = 1$

A square matrix U is orthogonal if all its columns are orthogonal to each other and are
normalized (the columns are then referred to as being orthonormal ).
$$U ^ { T } U = I = U U ^ { T }$$

A nice property of orthogonal matrices is that operating on a vector with an
orthogonal matrix will not change its Euclidean norm,
$$\| U x \| _ { 2 } = \| x \| _ { 2 }$$

#Null Space, Column Space and span

The span of a set of vectors is the set of all vectors that can be expressed asa linear combination of those vectors
$$\operatorname { span } \left( \left\{ x _ { 1 } , \ldots x _ { n } \right\} \right) = \left\{  \sum _ { i = 1 } ^ { n } \alpha _ { i } x _ { i } , \quad \alpha _ { i } \in \mathbb { R } \right\}$$

The nullspace of a matrix is the is the set of all vectors that equal
0 when multiplied by A,
$$\mathcal { N } ( A ) = \left\{ x \in \mathbb { R } ^ { n } : A x = 0 \right\}$$

The column space of A, denoted by C(A), is the span of the columns of A for all vectors

#The Determinant

The determinant of a square matrix det A
The  formula for determinant for an nxn matrix A is
$$ | A | = \sum _ { i = 1 } ^ { n } ( - 1 ) ^ { i + j } a _ { i j } \left| A _ { | i , j | } \right| \quad \text { (for any } j \in 1 , \ldots , n )$$

Example for  a 2x2 matrix:

$$| A | = \left| \begin{array} { l l } { a } & { b } \\ { c } & { d } \end{array} \right| = a d - b c$$  

Properties of determinants are as follows:
-   $| A | = \left| A ^ { T } \right|$
-   $| A B | = | A | | B |$
-   $| A | = 0$ if and only if A is singular
-   For non singular matrix $\left| A ^ { - 1 } \right| = 1 / | A |$

#Eigenvalues and Eigenvectors

Given a square matrix $A \in \mathbb { R } ^ { n \times n }$ ; $\lambda \in \mathbb { C }$ is an eigenvalue of A and $x \in \mathbb { C } ^ { n }$ is the eigenvector if

$$A x = \lambda x , \quad x \neq 0$$

Properties of Eigen value and Eigen Vectors:
1. $$| ( \lambda I - A ) | = 0$$
2. $$\operatorname { tr } A = \sum _ { i = 1 } ^ { n } \lambda _ { i }$$
3. $$| A | = \prod _ { i = 1 } ^ { n } \lambda _ { i }$$
4. The rank of A is equal to the number of non-zero eigenvalues of A.

We can write all the eigenvector equations simultaneously as

$$ A X = X \Lambda \quad or\quad  A = X \Lambda X ^ { - 1 }$$


where
$$ X \in \mathbb { R } ^ { n \times n } = \left[ \begin{array} { c c c c } { 1 } & { 1 } & { 1 } \\ { x _ { 1 } } & { x _ { 2 } } & { \cdots } & { x _ { n } } \\ { 1 } & { 1 } & { } & { 1 } \end{array} \right] , \Lambda = \operatorname { diag } \left( \lambda _ { 1 } , \ldots , \lambda _ { n } \right)$$







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
