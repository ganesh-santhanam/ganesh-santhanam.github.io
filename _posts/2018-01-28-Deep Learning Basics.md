---
title: "Machine Learning Project: Perceptron"
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


Deep learning (also known as deep structured learning or hierarchical learning) is part of a broader family of machine learning methods based on learning data representations, as opposed to task-specific algorithms. Learning can be supervised, semi-supervised or unsupervised

<p align="center">
<img src="https://imgur.com/aAQATco.jpg">

</p>

<center>
Deep Learning is a subset of AI
</center>

# Gradient Descent

Most deep learning algorithms involve optimization of some sort. Optimization
refers to the task of either minimizing or maximizing some function f (x) by altering x. The function we want to minimize or maximize is called the objective function or the cost function . We can reduce f (x) by moving x in small steps with opposite sign of the derivative to get to the minima (Local or global depending on the function) .This technique is called gradient descent.

For functions with multiple inputs, we must make use of the concept of partial
derivatives. The partial derivative $\frac { \partial } { \partial x _ { i } } f ( x )$ measures how f changes as only the variable $x _ { i }$ increases at point x . The gradient of f is the vector containing all of the partial derivatives, denoted by $\nabla _ { \boldsymbol { x } } f ( \boldsymbol { x } )$

The new point is proposed using the below equation where $\epsilon$ is the learning rate which determines the step size.

$$ x ^ { \prime } = x - \epsilon \nabla _ { x } f ( x ) $$
  <p align="center">


  <img src="https://upload.wikimedia.org/wikipedia/commons/f/ff/Gradient_descent.svg">

  </p>

  <center>
  Gradient Descent
  </center>  

# Mini-batch gradient descent  
 In large-scale applications the training data can have on order of millions of examples. It is computationally wasteful to compute the full loss function over the entire training set in order to perform only a single parameter update.Instead we compute the gradient over batches of the training data.A typical batch size contains 256 examples from the entire training set(Epoch) . This batch is then used to perform a parameter update.

 # Second order Optimization

  We can also think of using the second derivative as it measures curvature to achieve faster convergence.

  The Hessian matrix H(f)(x) is defined as:
$$ \boldsymbol { H } ( f ) ( \boldsymbol { x } ) _ { i , j } = \frac { \partial ^ { 2 } } { \partial x _ { i } \partial x _ { j } } f ( \boldsymbol { x } ) $$

A popular group of methods for optimization in context of deep learning is based on Newton’s method, which iterates the following update:

$$ x \leftarrow x - [ H f ( x ) ] ^ { - 1 } \nabla f ( x )$$

Multiplying by the inverse Hessian leads the optimization to take more aggressive steps in directions of shallow curvature and shorter steps in directions of steep curvature. Note, crucially, the absence of any learning rate hyperparameters in the update formula, which the proponents of these methods cite this as a large advantage over first-order methods.

However, the update above is impractical for most deep learning applications because computing (and inverting) the Hessian in its explicit form is a very costly process in both space and time. For instance, a Neural Network with one million parameters would have a Hessian matrix of size [1,000,000 x 1,000,000] which is computationally expensive.  There are algorithms such as L-BFGS, which uses the information in the gradients over time to form the approximation implicitly (i.e. the full matrix is never computed). However these do not work in mini batch settings hence are not commonly used with Deep learning.

 Instead of initializing the weights in a purely random manner, Xavier initialization enables to have initial weights that take into account characteristics that are unique to the architecture. If W is the weights gaussian distribution and N is the number of neurons.

$$\operatorname { Var } ( W ) = \frac { 1 } { n _ { \mathrm { in } } }$$

To reduce dependence on initialization we use Batch Norm . It is usually done after a fully connected/convolutional layer and before a non-linearity layer and aims at allowing higher learning rates. $\gamma ,\beta$ are the hyperparameters while $\mu , \sigma$ are the mean and variance of the batch which we want to normzlize.

$$  { x _ { i } \leftarrow \gamma \frac { x _ { i } - \mu _ { B } } { \sqrt { \sigma _ { B } ^ { 2 } + \epsilon } } + \beta } $$


There are other optimization methods with adaptive learning rates

<p align="center">
<img src="https://imgur.com/KTCmBYy.jpg">

</p>

<center>
Optimization methods
</center>


<p align="center">
<img src="http://cs231n.github.io/assets/nn3/opt2.gif">

</p>

<center>
Optimization methods comparison ( Credit Alec Radford)
</center>

# Feedforward neural networks

In this network, the information moves in only one direction, forward, from the input nodes, through the hidden nodes (if any) and to the output nodes. There are no cycles or loops in the network.
<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/en/5/54/Feed_forward_neural_net.gif">

</p>

<center>
Feedforward neural network
</center>


With an appropriate loss function on the neuron’s output, we can turn a single neuron into a linear classifier such as binary Softmax classifier. We can interpret $\sigma \left( \sum _ { i } w _ { i } x _ { i } + b \right)$ to be the probability of one of the classes $P \left( y _ { i } = 1 | x _ { i } ; w \right)$ adn the probability of the other class would be $P \left( y _ { i } = 0 | x _ { i } ; w \right) = 1 - P \left( y _ { i } = 1 | x _ { i } ; w \right)$

Some of the commonly used activation functions are Sigmoid $\sigma ( x ) = 1 / \left( 1 + e ^ { - x } \right)$ , Tanh $\tanh ( x ) = 2 \sigma ( 2 x ) - 1$ , RELU(Rectified linear unit)  $f ( x ) = \max ( 0 , x )$ , Leaky Relu $f ( x ) = 1 ( x < 0 ) ( \alpha x ) + 1 ( x > = 0 ) ( x )$ where $\alpha$ is a small constant

The universal approximation theorem (Hornik et al., 1989; Cybenko, 1989) states that a feedforward network with a linear output layer and at least one hidden layer with any “squashing” activation function can approximate any function from one finite-dimensional space to another with any desired non-zero
amount of error, provided that the network is given enough hidden units


<p align="center">
<img src="https://imgur.com/lcDCbcV.jpg">

</p>

<center>
Activation Functions (Source Staford cs229 Notes)
</center>


The universal approximation theorem means that regardless of what function
we are trying to learn, we know that a large MLP will be able to represent this
function. However, we are not guaranteed that the training algorithm will be able
to learn that function.

#Back-Propagation

When we use a feedforward neural network to accept an input x and produce an
output ˆ y, information flows forward through the network. The inputs x provide
the initial information that then propagates up to the hidden units at each layer
and finally produces yˆ. This is called forward propagation. During training,
forward propagation can continue onward until it produces a scalar cost J (θ).
The back-propagation algorithm (Rumelhart et al., 1986a), often simply called
backprop, allows the information from the cost to then flow backwards through
the network, in order to compute the gradient.

The derivative with respect to weight W is computed using chain rule and is of the following form:
$$ \frac { \partial L ( z , y ) } { \partial w } = \frac { \partial L ( z , y ) } { \partial a } \times \frac { \partial a } { \partial z } \times \frac { \partial z } { \partial w } $$

The weights are updated as follows:
$$ w \longleftarrow w - \alpha \frac { \partial L ( z , y ) } { \partial w }$$

Updating weights ― In a neural network, weights are updated as follows:
- Step 1: Take a batch of training data.
- Step 2: Perform forward propagation to obtain the corresponding loss.
- Step 3: Backpropagate the loss to get the gradients.
- Step 4: Use the gradients to update the weights of the network.

Let us see an example of a Sigmoid with a 2-dimensional neuron (with inputs x and weights w) for Back prop

$$ f ( w , x ) = \frac { 1 } { 1 + e ^ { - \left( w _ { 0 } x _ { 0 } + w _ { 1 } x _ { 1 } + w _ { 2 } \right) } }$$

We can group multiple gates into a single gate, or decompose a function into multiple gates whenever it is convenient for computation.


$$ \begin{array} { c c } { f ( x ) = \frac { 1 } { x } } & { \rightarrow } & { \frac { d f } { d x } = - 1 / x ^ { 2 } } \\ { f _ { c } ( x ) = c + x } & { \rightarrow } & { \frac { d f } { d x } = 1 } \\ { f ( x ) = e ^ { x } } & { \rightarrow } & { \frac { d f } { d x } = e ^ { x } } \\ { f _ { a } ( x ) = a x } & { \rightarrow } & { \frac { d f } { d x } = a } \end{array}$$

$$ \begin{array} { c } { \sigma ( x ) = \frac { 1 } { 1 + e ^ { - x } } } \\ { \rightarrow \quad \frac { d \sigma ( x ) } { d x } = \frac { e ^ { - x } } { \left( 1 + e ^ { - x } \right) ^ { 2 } } = \left( \frac { 1 + e ^ { - x } - 1 } { 1 + e ^ { - x } } \right) \left( \frac { 1 } { 1 + e ^ { - x } } \right) = ( 1 - \sigma ( x ) ) \sigma ( x ) } \end{array} $$



<p align="center">
<img src="https://imgur.com/e61BZgO.jpg">

</p>

<center>
Example of a Sigmoid activation function
</center>

This decomposition of the network to graphs is done automatically in modern deep learning packages. The gradients are calculated automatically using reverse mode auto differentiation.

The core ideas behind modern feedforward networks have not changed substantially
since the 1980s. The same back-propagation algorithm and the same approaches to gradient descent are still in use. Most of the improvement in neural
network performance from 1986 to now can be attributed to larger datasets and to more powerful computers.

#REGULARIZATION FOR DEEP LEARNING

We can use the L1(Lasso) and L2(Ridge) REGULARIZATION from Classic machine learning.

L1 regularization is defined as $\lambda \| \theta \| _ { 1 }$ while
L2 regularization is defined as  $\lambda \| \theta \| _ { 2 } ^ { 2 }$

Dropout is a regularization technique for reducing overfitting in neural networks by preventing complex co-adaptations on training data. It drops out neurons with probability p > 0 and forces the model to avoid relying too much on particular sets of features.

<p align="center">
<img src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/05/20073927/dropout.png">

</p>

<center>
Dropout
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
