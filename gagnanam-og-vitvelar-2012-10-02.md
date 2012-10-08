% Notes
% Jóhann Þorvaldur Bergþórsson & Stefanía Bjarney Ólafsdóttir
% 2. October 2012

# Fisher discriminant for multiple classes

For two classes we are maximizing
$$J(\w) = \frac{(m_2 - m_1)^2}{s_1^2 + s_2^2} = \frac{\w^TS_B\w}{\w^T S_W \w}$$
For multiple classes we have
$$\y = W^T \x$$
and we redefine
$$S_W = \sum_{k=1}^{K} S_k$$
and
$$S_B = \sum_{k=1}^{K} N_k(\m_k - \m)(\m_k - \m)^T$$
where $S_W$ is the _within-class covariance matrix_ and $S_B$ is the _between-class-covariance matrix_.

![](img/2012-10-02-1.jpg)

\begin{align*}
S_W^{(y)} &= W S_W W^T \\
S_B^{(y)} &= W S_B W^T \\
{S_W^{(y)}}^{-1} S_B^{(y)} &= (W S_W W^T)^{-1}(W S_B W^{-1})
\end{align*}

We can think of $S_W^{(y)}$ and $S_B^{(y)}$ as $S_W$ and $S_B$ after we have projected down into a lower dimension.

So for multiple classes we get
$$J(\w) = \Tr ((W S_W W^T)^{-1}(W S_B W^T))$$
i.e. weights are determined by eigenvectors of $S_W^{-1} S_B$

## The perceptron algorithm

Related to neural networks

![](img/2012-10-02-2.jpg)

$$y(\x) = f(\w^T \phi(\x))$$
where
$$f(a) = \begin{cases}
    1, &a \ge 0\\
    -1, &a < 0
\end{cases}$$

### Error criterion
Total number of falsely classified patterns which can be difficult, i.e. there is no simple formula.

### Perceptron criterion

Using the target coding sheme $t_n \in {-1, 1}$ we would like all patterns to have
$$\w^T \phi(\x_n) t_n > 0$$

* Zero error is associated to correctly classified patterns
* $-\w^T \phi(\x_n) t_n$ error is associated to  misclassified patterns, total error
$$E_p(\w) = \sum_{n\in\mathcal{M}} \w^T \phi(\x) t_n$$
where $\mathcal{M}$ is the set of misclassified patterns.

![Perceptron criterion can't solve the xor problem](img/2012-10-02-3.jpg)

# Probabilistic Generative Models

We approach the problem of classification from a probability theory point of view. If we are given an observation $\x$ then we want to determine the posterior probability for all the classes and choose the highest value
$$\hat{k} = \text{chosen class} = \arg \max_k p(C_k | \x)$$
Let's explore $p(C_k | \x)$ and start by assuming we have two classes.
\begin{align*}
p(C_1 | \x) &= \frac{p(\x|C_1) p(C_1)}{p(\x)} \\
    &= \frac{p(\x|C_1) p(C_1)}{p(\x|C_1)p(C_1) + p(\x |C_2)p(C_2)}
\end{align*}

If we define
$$a = \ln \frac{p(\x|C_1) p(C_1)}{p(\x|C_2)p(C_2)}$$
(which is called the _logit function_) then
$$p(C_1|\x) = \frac{1}{1 - e^a} = \sigma(a)$$
where $\sigma(a)$ is called the _logistic sigmoid_.

The shape of $\sigma(a$ forces the result to be between $0$ and $1$.

![](img/2012-10-02-4.jpg)

We will look at cases where $a(\x)$ takes a linear form. We will end up with generalized linear model where
$$y(\x) = p(C_1|\x) = \sigma(a(\x)) = \sigma(\w^T\x+ w_0)$$
So the non-linear function in the generalized linear model is the sigmoid, a smooth approximation to
$$f(b) = \begin{cases}
    1, &b \ge 0\\
    0, &b < 0
\end{cases}$$

The multiclass extension of this is
$$p(C_k|\x) = \frac{ p(\x|C_k)p(C_k) }{ \underbrace{\sum_j p(\x|C_j)p(C_j)}_{p(\x)} } = \frac{e^{a_k}}{\sum_j e^{a_j}}$$
where
$$\frac{e^{a_k}}{\sum_j e^{a_j}}$$
is the _normalized exponential_, also known as the _softmax function_. Here we have $a_k(\x) = \ln (p(\x|C_k) p(C_k) )$

If $a_k >> a_j, \forall j \neq k$ then $p(C_k|\x) \approx 1$ and $p(C_j|\x) \approx 0$.

## Gaussian input

Let's assume that
$$p(\x|C_k) = \frac{1}{(2\pi)^{D/2}} \frac{1}{\det{\mvSigma_k}^{\frac{1}{2}}} \exp \left ( -\frac{1}{2} (\x - \mvmean_k)^T){\mvSigma^{-1}}(\x - \mvmean_k) \right )$$
We then have
\begin{align*}
a_k(\x) &= \ln ( p(\x | C_k) p(C_k) ) \\
&= - \frac{1}{2} (\x - \mvmean_k)^T \mvSigma_k^{-1} (\x -\mvmean_k) - \frac{D}{2} \ln (2 \pi) - \frac{1}{2} \ln \det{\mvSigma_k} + \ln (p(C_k))
\end{align*}

Let's look at $3$ cases:

### Case 1: $\mvSigma_k = \sigma^2 \I$

The samples fall into $K$ equally hypersperical clusters centered about $\mvmean_k$.

![](img/2012-10-02-5.jpg)

The we get
$$a_k(\x) = \frac{\norm{\x - \mvmean_k}^2}{2 \sigma^2} + \ln (p(C_k))$$
where we have cancelled out term 2 and term 3 becomes the same for all classes. We can also expand
$$a_k(\x) = -\frac{1}{2\sigma^2}(\x^T\x - 2 \mvmean_k)\x + \mvmean_k \mvmean) + \ln p(C_k)$$
So we define a new $a$
$$a_k(\x) = \w_k^T \x + \w_{k_0}$$
where
$$\w_k = \frac{\mvmean_k}{\sigma^2}$$
and
$$\w_{k_0} = - \frac{1}{2\sigma^2}\mvmean_k^T \mvmean_k + \ln (p(C_k))$$

