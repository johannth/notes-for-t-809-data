% Notes
% Jóhann Þorvaldur Bergþórsson & Stefanía Bjarney Ólafsdóttir
% 9. October 2012

Neural Networks
=====================================

Multilayer feed forward neural network

![](img/2012-10-09-1.jpg)

$$y_k(\x, \w) = \sigma \left ( \underbrace{\sum_{j=1}^{m} w_{kj}^{(2)} \underbrace{h \left( \underbrace{\sum_{i=1}^{D}w_{ji}^{(1)} x_i + w_{j0}^{(1)}}_{a_j} \right)}_{z_j} + w_{k0}^{(2)}}_{a_k} \right )$$


Trivial extensions
-------------------------------------

1. We can add layers

![An example of a multilayer feed forward neural network with more layers.](img/2012-10-09-2.jpg)

2. We can force some/many $w_{ji}$ and $w_{kj}$ to be zere - a sparse network.

![An example of a sparse network.](img/2012-10-09-3.jpg)

3. We can skip layers.

![An example of a neural network where some layers are skipped.](img/2012-10-09-4.jpg)

Nontrivial extensions
-------------------------------------

1. Recurrent connections, a neural network which is not longer a feed forward neural network.

![An example of a nontrivial neural network with recurrent connections.](img/2012-10-09-5.jpg)

2. ... (there are more nontrivial extensions but they are nontrivial and we don't even look at them.)

Weight Space Symmetries
------------------------------------
We have a weight vector
$$\w = \begin{bmatrix}
    w_1 \\ \vdots \\ w_C
\end{bmatrix} = \begin{bmatrix}
    w_{10} \\ w_{20} \\ \vdots \\ v_{M0} \\ \vdots
\end{bmatrix}$$
We can one instance of our neural network as a point in a $C$-dimensional space.

![An example of the symmetries in the weight space.](img/2012-10-09-6.jpg)


1.5.4 Inference and Decision
===========================================

a) Use class conditional probabilities $p(\x | C_k)$. We can compute this and therefore we can get $p(C_k | \x)$ using Bayes.

b) Use posterior class probabilities directly $p(C_k | \x)$.

c) Not connected to probabilities. We define a function that makes it easier for us to make a decision (e.g. least squares, Fisher, etc.)

Probabilistic Discriminant Models
======================================
Alternative to using generative models. The functional form of the generalized linear model is used explicitly and parameters estimated using maximum likelihood. This is called _discriminative training_.

### Fixed basis functions
We interchange $\x$ with $\phi(\x)$ and proceed.

### Logistic Regression

Here we assume that the posterior probability of class $C_1$ can be expressed as
$$p(C_1| \x) = y(\phi(\x)) = \sigma(\w^T \phi(\x)) = 1 - p(C_2 | \x)$$
and
$$\sigma(a) = \frac{1}{1 + e^{-a}}$$
Notice that this model has $M$ adjustable parameters.

If we had to fit a Gaussian we would have to fit
$$2M + \frac{M(M+1)}{2}+1 = \frac{M(M+5)}{2} + 1$$

Therefore logistic regression is good if $M$ is large (and training data small).

Now assume we have a dataset $\{ \phi_n, t_n \}$ where $t_n \in \{0,  1\}$ and $\phi_n = \phi(\x_n)$, with $n=1, \dotsc, N$. Then the likelihood function is
$$p(\t| \w) = \prod_{n=1}^N y_n^{t_n}(1 - y_n)^{1 - t_n}$$
where $\t = \begin{bmatrix}
    t_1 & t_2 & \dotsc  & t_n
\end{bmatrix}^T$ and $y_n = p(C_1 | \phi_n)$.

We define the error function by taking the negative logarithm

$$E(w) = - \ln (p(\t | \w)) = - \sum_{n=1}^N \left \{ t_n \ln(y_n) + (1 - t_n) \ln(1 - y_n) \right \}$$

This is called the _cross entropy function_.

Let's remember $y = \sigma(a_n)$ and $a_n = \w^T \phi_n$ and note that $\pa{\sigma}{a} = \sigma (1 - \sigma)$. Then

$$\grad E(\w) = \sum_{n=1}^{N}(y_n - t_n)\phi_n$$
and there is no way to solve for $\w$. This can however be used in a sequential learning algorithm.

$$\w^{\tau + 1} = \w^{\tau} + \mu \grad E (\w^{\tau})$$

![Gradient descent.](img/2012-10-09-7.jpg)