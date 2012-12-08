% Notes
% Jóhann Þorvaldur Bergþórsson & Stefanía Bjarney Ólafsdóttir
% 11. October 2012

Error Backpropagation
=====================================

![](img/2012-10-11-1.jpg) \

\begin{align*}
    a_j &= \sum_{i} w_{ji} z_i \\
    z_j &= h(a_j)
\end{align*}

We now assume that we have computed all a's and z's in the network for $\x_n$. This computation is called _forward propagation_.

We have
$$E(\w) = \sum_{n=1}^N E_n(\w)$$
Now we we want to compute the derivative of $E_n(\w)$ with respect to $w_{ji}$. Note that $E_n(\w)$ is dependent on $\w_{ji}$ only through $a_j$. We can therefore use the chain rule:
$$\frac{\partial E_n(\w)}{\partial w_{ji}} = \frac{\partial E_n(\w)}{\partial a_j} \frac{\partial a_j}{\partial w_{ji}}$$

![](img/2012-10-11-2.jpg) \

We define
$$\delta_j = \frac{\partial E_n}{\partial a_j}$$
where $\delta_j$ is called _error related to neuron j_. We also have
$$\frac{\partial a_j}{\partial w_{ji_0}} = \frac{\partial}{\partial w_{ji_0}} \left ( \sum_i w_{ji} z_i \right ) = z_{i_0}$$
so
$$\pa{E_n}{w_{ji}} = \pa{E_n}{a_j} \pa{a_j}{w_{ji}} = \delta_j z_i$$
This means that if we compute $\delta_j$ for every neuron in the network we can obtain the derivative by using the input $z_i$.

If we use a linear activation function in the output layer we have $y_k = \sum_j w_{kj} z_j$ and if we use an error function
$$E_n = \frac{1}{2} \sum_k (y_{nk} - t_{nk})^2$$

![](img/2012-10-11-3.jpg) \

Then
$$\delta_k = \pa{E_n}{a_k} = y_k - t_k$$

Now we have obtained the $\delta$'s in the output layer, we can compute them for the hidden layers.

![](img/2012-10-11-4.jpg) \

Let's use the chain rule again:
\begin{align*}
\delta_j &= \pa{E_n}{a_j} = \sum_k = \pa{E_n}{a_k} \pa{a_k}{a_j} \\
    &= \sum_k \delta_k \pa{a_k}{a_j}
\end{align*}

\begin{align*}
a_k &= \sum_j w_{kj} z_j = \sum_j w_{kj} h(\hat{a}_j)
\pa{a_k}{a_{j_0}} &= w_{kj_0} h'(a_{j_0})
\delta_j &= \sum_k \hat{\delta}_k w_{kj} h' (a_j)
\end{align*}

The following is the error backpropagation algorithm: (pp. 244)

1) Apply the pattern $\x_n$ to the net and obtain all $a$'s and $z$'s.
2) Compute the output $\delta_k$
3) Use the backpropagation formula
$$\delta_j = h'(a_j) \sum_k \delta_k w_{kj}$$
to obtain all $\delta$'s in the net.
4) Use
$$\pa{E_n}{w_{ji}} = \delta_j z_i$$
to obtain the derivative.
5) If we are in batch mode we obtain the gradient of the total error by
$$\grad E(\w) = \pa{E}{W_{ji}} = \sum_n \pa{E_n}{w_{ji}}$$

A simple example
--------------------------
Let's use the activation function
$$h(a) = \tanh(a) = \frac{e^a - e^{-a}}{e^a + e^{-a}}$$
which has the derivative
$$h'(a) = 1 - h(a)^2$$
Consider
$$E_n = \frac{1}{2} \sum_{k=1}^{K} (y_k - t_k)^2$$

![](img/2012-10-11-5.jpg) \

1) For each pattern $\x_n$ we have
\begin{align*}
a_j &= \sum_{i= 0}^D w_{ji}^{(1)} x_i \\
z_j &= \tanh(a_j) \\
y_k &= \sum_{j=0}^M w_{kj}^{{2}} z_j
\end{align*}

2) Obtain the output $\delta$'s:
    $$\delta_k = y_{nk} - t_{nk}$$

3) Use the backprop formula:
$$\delta_j = h'(a_j) = \sum_{k=1}^K \delta_k w_{kj} = (1 - z_j^2) \sum_{k=1}^K \delta_k w_{kj}$$

4) Obtain the derivatives for the weights in layer 1:
$$\pa{E_n}{w_{ji}^{(1)}} = \delta_j x_i$$
layer 2:
$$\pa{E_n}{w_{kj}^{(2)}} = \delta_k z_j$$

After we have obtained $\grad E(\w)$ we can implement gradient decent to iterate on $\w$,
$$\w^{(\tau + 1)} = \w^{(\tau)} - \eta \grad E(\w)$$