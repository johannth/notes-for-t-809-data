% Notes
% Jóhann Þorvaldur Bergþórsson & Stefanía Bjarney Ólafsdóttir
% 10. October 2012

Network Training
======================================

A neural network is a nonlinear mapping of our input variables, dictated by weights $y(\x, \w)$. If we have a training set $[\x_n, t_n]$, $n=1, \dotsc, N$ then we should determine $\w$ by minimizing an error function. For example
$$E(\w) = \frac{1}{2} \sum_{n=1}^{N}\norm{y(\x_n, \w) - t_n}^2$$

Case 1: Single Output Regression
--------------------------------------
$$p(t|\x, \w) = \NormalDist(t | y(\x, \w, \beta^{-1}))$$

![\label{Figure1}](img/2012-10-10-1.jpg) \

Using the training set we get the likelihood function

$$p(\t| X, \w, \beta) = \prod_{n=1}^N p(t_n| \x_n, \w, \beta)$$

From this we can obtain the error function $E(\w)$ by taking the negative logarithm.
$$\frac{\beta}{2} \sum_{n=1}^{N} (y(\x, \w) - t_n)^2 - \frac{N}{2} \ln (\beta) + \frac{N}{2} \ln (2 \pi)$$
which can be used to learn $\w$ and $\beta$.

We define the value $\w$ which minimizes this error function as
$$\w_{ML} = \arg \min_{\w} E(\w)$$
When we have found $\w_{ML}$ we can compute
$$\frac{1}{\beta_{ML}} = \frac{1}{N} \sum_{n=1}^{N}(y(\x_n, \w_{ML}) - t_n)^2$$

Case 2: Multiple output regression
-------------------------------------

(Dæmi: verðbólga í mörgum mismunandi löndum út frá mörgum breytum.)

Case 3: Single output classification
-------------------------------------------

![](img/2012-10-10-2.jpg) \

$t=1$ denotes class $C_1$, $t=0$ class $C_2$. Let's consider a network that has a single output with activation
$$y = \sigma(a) = \frac{1}{1+ e^{-a}}$$
so $y(\x, \w) \in [0, 1]$. Let's interpret $y(\x, \w)$ as class conditional probability $p(C_1|\x)$ and $p(C_2, \x)$ is then $1 - p(C_1|\x)$[^1]. Then the conditional distribution of the targets is given by the Bernoulli distribution (see pp. 69)
$$p(t| \x, \w) = y(\x, \w)^t (1 - y(\x, \w))^{(1-t)} = \begin{cases}
    y(\x, \w) &\caseif C_1 \\
    1 - y(\x, \w), &\caseif C_2
\end{cases}$$
Now we have a training set and we get the cross entropy error function
$$E(\w) = - \sum_{n=1}^{N} \left \{ t_n \ln(y_n) + (1 - t_n) \ln (1 - y_n) \right \}$$

+ There is no analogue of the noise precision $\beta$ because we assume that $t_n$ is correct.
+ Using this error function leads to faster training.

[^1]: In single output classification we can only work with two classes because the output is always between $0$ and $1$

Case 4: Multiple Output Classification
-----------------------------------------
Leads to softmax for output activation.

![Softmax as output activation for multiple classification](img/2012-10-10-3.jpg) \


Comparison
-----------------------------

Task                     $E(\w)$                    Output activation
------------------------ -------------------------  -------------------
Regression               sum-of-squares             linear
Binary classification    cross-entropy              logistic sigmoid
Multiple classification  multiclass cross-entropy   softmax



Parameter Optimization
==============================
For neural networks, it is impossible to solve $\grad E(\w) = \0$ to optain $\w$ optimum. We must use iterative procedures
$$\w^{(\tau+1)} = \w^{(\tau)} + \delta \w^{(\tau)}$$
Most algorithms use
$$\delta \w^{(\tau)} \propto \grad E(\w^{(\tau)})$$

![Iterative method to find an optimal $\w$.](img/2012-10-10-4.jpg)

Local Quadratic Approximation
====================================

Let's consider the Taylor expansion around a point $\hat{\w}$ in weight space
$$E(\w) \approx E(\hat{\w}) + (\w - \hat{\w})^T \vec{b} + \frac{1}{2}(\w - \hat{\w})^T \vec{H} (\w - \hat{\w})$$
where
$$\vec{b} = \grad E|_{\w=\hat{\w}}$$
and
$$H_{ij} = \frac{\partial E}{\partial w_i \partial w_j} |_{\w = \hat{\w}}$$
A local approximation for the gradient is
$$\grad E \approx \vec{b} + \vec{H}(\w - \hat{\w})$$
for points $\hat{\w}$ which are sufficiently close to $\w$.

Gradient Descent
====================================
The simplest approach to using gradient information is to choose an initial value for $\w$, $\w^0$ and then travel in the direction of the negative gradient in small steps.
$$\w^{(\tau + 1)} = \w^{(\tau)} - \eta \grad E (\w^{(\tau)})$$
where $\eta > 0$ is known as the _learning rate_. Note that the error function is defined with respect to the whole dataset, a so called batch method.

At each step the weight vector is pushed in the direction of the greatest rate of decrease of $E(\w)$. This method is called _gradient descent_ or _steepest descent_.

Online version leads to _sequential gradient descent_ or _stochastic gradient descent_. Note that $E(\w) = \sum_{n=1}^{N} E_n(\w)$ so we can update $\w$ based on one data point only.
$$\w^{(\tau + 1)} = \w^{(\tau)} - \eta \grad E_n(\w^{(\tau)})$$

Error Backpropagation
=====================================
Backpropagation is a method to compute $\grad E(\w)$. Most practical error functions are
$$E(\w) = \sum_{n=1}^{N} E_n(\w)$$

![](img/2012-10-10-5.jpg)

The input into the neuron is
$$a_j = \sum_{i} w_{ji} z_i$$
$z_i$ is the output of other neurons, or inputs that send their connections to the neuron in question and $z_j = h(a_j)$ is the output of the neuron in question.
