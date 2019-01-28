---
title: "Dropout as bayesian approxiamtion"
mathjax: "true"
---

While current deep networks are widely used in multiple applications, they do have some issues:
* Neural networks compute point estimates
* Neural networks make overly confident decisions about the correct class, prediction or action
* Neural networks are prone to overfitting

While Bayesian Neural networks do address these issues they dont scale to the large scale data available today. One approach that solves this is is using stochastic Dropout as bayesian approximation.

# Dropout in Deep Learning

Dropout is a well-established procedure to regularize a neural network and limit overfitting. It is first introduced by Srivastava et al. [1] using a branch/prediction averaging analogy. The “dropout as a Bayesian Approximation” proposes a simple approach to quantify the neural network uncertainty. It employs dropout during both training and testing and this is called Monte Carlo Dropout(MC Dropout). MC Dropout offers a new and handy way to estimate uncertainty with minimal changes in most existing networks. In the simplest case, you just need to keep your dropout on at test time, then pass the data multiple times and store all the predictions. The downside is, this could be computationally expensive.

The Steps are:
Given a point x:
1. Drop units at test time
2. Repeat n times
3. Look at mean and sample variance





$$
\text { Posterior: } p ( \mathbf { w } | \mathbf { X } , y ) = \frac { p ( y | \mathbf { X } , \mathbf { w } ) p ( \mathbf { w } ) } { p ( y | \mathbf { X } ) }
$$

We compute  distribution y* for new input x*
$$
p \left( y ^ { * } | \mathbf { x } ^ { * } , \mathbf { X } , y \right) = \int p \left( y ^ { * } | \mathbf { x } ^ { * } , \mathbf { w } \right) p ( \mathbf { w } | \mathbf { X } , y ) d \mathbf { w }
$$

As exact inference is computationally intractable we try to  approximate the Posterior using variational inference by optimizing ELBO. However this can cause issues with high variance of the gradient estimates.


<p align="center">
<img src="https://imgur.com/JUKu4xj.jpg">

</p>

<center>
Dropout
</center>




$$
\begin{aligned} z _ { i } ^ { ( l + 1 ) } & = \mathbf { w } _ { i } ^ { ( l + 1 ) } \mathbf { y } ^ { l } + b _ { i } ^ { ( l + 1 ) } \\ y _ { i } ^ { ( l + 1 ) } & = f \left( z _ { i } ^ { ( l + 1 ) } \right) \end{aligned}
$$
where f is any activation function.

Dropout multiplies hidden activations by Bernoulli distributed random variables which take value 1 with probability p and 0 otherwise
With Dropout feed forward run becomes
$$
\begin{array} { r l } { r _ { j } ^ { ( l ) } } & { \sim \text { Bernoulli } ( p ) } \\ { \widetilde { \mathbf { y } } ^ { ( l ) } } & { = \mathbf { r } ^ { ( l ) } * \mathbf { y } ^ { ( l ) } } \\ { z _ { i } ^ { ( l + 1 ) } } & { = \mathbf { w } _ { i } ^ { ( l + 1 ) } \widetilde { \mathbf { y } } ^ { l } + b _ { i } ^ { ( l + 1 ) } } \\ { y _ { i } ^ { ( l + 1 ) } } & { = f \left( z _ { i } ^ { ( l + 1 ) } \right) } \end{array} $$


<p align="center">
<img src="https://imgur.com/cs2vfO8.jpg">

</p>

<center>
Dropout vs standard
</center>

With Dropout the objective function
$$
\mathcal { L } _ { \text { dropout } } = \underbrace { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } E \left( \mathbf { y } _ { i } , \hat { \mathbf { y } } _ { i } \right) } _ { \text { Loss function } } + \underbrace { \lambda \sum _ { i = 1 } ^ { L } \left( \left\| \mathbf { W } _ { i } \right\| ^ { 2 } + \left\| \mathbf { b } _ { i } \right\| ^ { 2 } \right) } _ { L _ { 2 } \text { regularization with weight decay\lambda } }
$$

Gal and Ghahramani showed that we can reparametrize  the approximate variational distribution to be non-gaussian(Bernoulli)

They showed that Variational inference and dropout objective functions are same given the choice of approximating reparametrized posterior and normal priors over network weights.

optimizing any neural network with dropout is equivalent to a form of approximate bayesian inference

Network trained with dropout is a bayesian Neural network

MC dropout requires application of dropout at every weight layer at test time
$$q \left( \mathbf { y } ^ { * } | \mathbf { x } ^ { * } \right) = \int p \left( \mathbf { y } ^ { * } | \mathbf { x } ^ { * } , \mathbf { w } \right) q ( \mathbf { w } | \mathbf { X } , \mathbf { Y } )$$

MC dropout averages over T forward passes  thru the network at test time. To estimate test mean and uncertainty by simply collecting the stochastic dropout forward passes.

<p align="center">
<img src="https://imgur.com/rKFQo3e.jpg">

</p>

<center>
Results on common datasets ( Source Andrew Rowan)
</center>


<p align="center">
<img src="https://imgur.com/RvNacsP.jpg">

</p>

<center>
Results on Synthetic dataset in Python
</center>
# References

1.Srivastava et al  Dropout:A simple way to prevent neural networks from overfitting,2014  
2.Abadi et al ,TensorFlow: Large-scale machine learning on heterogeneous systems, 2015  
3.Gal and Ghahramani: Dropout as Bayesian approximation: Representing model uncertainty in deep learning 2015  
4.Radford Neal: Monte Carlo Implementation of Gaussian Process Models for Bayesian Regression and Classification 1997







```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib.distributions import Bernoulli


class VariationalDense:
    """Variational Dense Layer Class"""
    def __init__(self, n_in, n_out, model_prob, model_lam):
        self.model_prob = model_prob
        self.model_lam = model_lam
        self.model_bern = Bernoulli(probs=self.model_prob, dtype=tf.float32)
        self.model_M = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.01))
        self.model_m = tf.Variable(tf.zeros([n_out]))
        self.model_W = tf.matmul(
            tf.diag(self.model_bern.sample((n_in, ))), self.model_M
        )

    def __call__(self, X, activation=tf.identity):
        output = activation(tf.matmul(X, self.model_W) + self.model_m)
        if self.model_M.shape[1] == 1:
            output = tf.squeeze(output)
        return output

    @property
    def regularization(self):
        return self.model_lam * (
            self.model_prob * tf.reduce_sum(tf.square(self.model_M)) +
            tf.reduce_sum(tf.square(self.model_m))
        )

# Created sample data.
n_samples = 20
X = np.random.normal(size=(n_samples, 1))
y = np.random.normal(np.cos(5.*X) / (np.abs(X) + 1.), 0.1).ravel()
X_pred = np.atleast_2d(np.linspace(-3., 3., num=100)).T
X = np.hstack((X, X**2, X**3))
X_pred = np.hstack((X_pred, X_pred**2, X_pred**3))

# Create the TensorFlow model.
n_feats = X.shape[1]
n_hidden = 100
model_prob = 0.9
model_lam = 1e-2
model_X = tf.placeholder(tf.float32, [None, n_feats])
model_y = tf.placeholder(tf.float32, [None])
model_L_1 = VariationalDense(n_feats, n_hidden, model_prob, model_lam)
model_L_2 = VariationalDense(n_hidden, n_hidden, model_prob, model_lam)
model_L_3 = VariationalDense(n_hidden, 1, model_prob, model_lam)
model_out_1 = model_L_1(model_X, tf.nn.relu)
model_out_2 = model_L_2(model_out_1, tf.nn.relu)
model_pred = model_L_3(model_out_2)
model_sse = tf.reduce_sum(tf.square(model_y - model_pred))
model_mse = model_sse / n_samples
model_loss = (
    # Negative log-likelihood.
    model_sse +
    # Regularization.
    model_L_1.regularization +
    model_L_2.regularization +
    model_L_3.regularization
) / n_samples
train_step = tf.train.AdamOptimizer(1e-3).minimize(model_loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        sess.run(train_step, {model_X: X, model_y: y})
        if i % 100 == 0:
            mse = sess.run(model_mse, {model_X: X, model_y: y})
            print("Iteration {}. Mean squared error: {:.4f}.".format(i, mse))

    # Sample from the posterior.
    n_post = 1000
    Y_post = np.zeros((n_post, X_pred.shape[0]))
    for i in range(n_post):
        Y_post[i] = sess.run(model_pred, {model_X: X_pred})

if True:
    plt.figure(figsize=(8, 6))
    for i in range(n_post):
        plt.plot(X_pred[:, 0], Y_post[i], "b-", alpha=1. / 200)
    plt.plot(X[:, 0], y, "r.")
    plt.grid()
    plt.show()
```
