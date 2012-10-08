% Notes
% Jóhann Þorvaldur Bergþórsson & Stefanía Bjarney Ólafsdóttir
% 3. October 2012

# Probabilistic Generative Models

Trying to use $p(\x|C_k)$. Discriminant function:

Softmax:
$$a_k(\x) = \ln ( p(\x|C_k)p(C_k))$$

## Gaussian input

In general we have
\begin{align*}
a_k(\x) &= \ln ( p(\x | C_k) p(C_k) ) \\
&= - \frac{1}{2} (\x - \mvmean_k)^T \mvSigma_k^{-1} (\x -\mvmean_k) - \frac{D}{2} \ln (2 \pi) - \frac{1}{2} \ln \det{\mvSigma_k} + \ln (p(C_k))
\end{align*}

We can look at the following     3 cases:

### Case 1: $\mvSigma_k = \sigma^2 \I$

$$a_k(\x) = \w_k^T \x + \w_{k_0}$$
where
$$\w_k = \frac{\mvmean_k}{\sigma^2}$$
and
$$\w_{k_0} = - \frac{1}{2\sigma^2}\mvmean_k^T \mvmean_k + \ln (p(C_k))$$

Let's take a look at the decision boundary between $C_k$ and $C_j$, $a_k(\x) = a_j(\x)$. From this we get
$$\w^T(\x - \x_0) = 0$$
where
$$\w = \mvmean_k - \mvmean_j$$
$$\x_0 = \frac{1}{2}(\mvmean_k - \mvmean_j) - \frac{\sigma^2}{\norm{\mvmean_k - \mvmean_j}} \ln \left( \frac{p(C_k)}{p(C_j)} \right ) (\mvmean_k - \mvmean_j)$$
This is a hyperplane through $\x_0$ and orthogonal to $\w$.

![](img/2012-10-03-1.jpg)

Such a classifier is called the *minimum distance classifier*. If the class means $\mvmean_k$ are interpreted as prototypes or templates for the class, this method is called *template matching*.

### Case 2: $mvSigma_k = \mvSigma$

The samples fall into hyper-ellipsoidal clusters of equal size and shape. Then
$$a_k(\x) = -\frac{1}{2}(\x - \mu_k)^T \mvSigma^{-1}(\x - \mu_k) + \ln(p(C_k))$$
(we leave out term 2 and 3 because they are the same for all classes). We expand this:
$$a_k(\x) = -\frac{1}{2}(\x^T\mvSigma^{-1}\x - 2 \mvmean_k \mvSigma^{-1}\x + \mvmean_k \mvSigma^{-1} \mvmean_k) + \ln(p(C_k))$$
and since $\x^T \mvSigma^{-1} \x$ does not depend on $C_k$ we can write
$$a_k(\x) = \w_k^T \x + w_{k0}$$
where
$$\w_k = \mvSigma^{-1} \mvmean_k$$
and
$$w_{k0} = - \frac{1}{2} \mvmean_k^T \mvSigma^{-1} \mvmean_k + \ln (p(C_k))$$

![](img/2012-10-03-2.jpg)

(Til þess að fá línulegan classifier fyrir Gaussíska dreifingu þá þurfum við að gera ráð fyrir að öll samvikafylkin séu eins)

### Case 3: $\mvSigma_3$ is arbitrary

Here the only term we can drop is term 2: $\frac{D}{2}\ln(2\pi)$. We get
$$a_k(\x) = \x^T \W_k \x + \w_k^T \x_k + w_{k0}$$
where
$$\W_k = -\frac{1}{2} \mvSigma_k^{-1}$$
$$\w_k = \mvSigma_k^{-1} \mvmean_k$$
$$w_{k0} = -\frac{1}{2} \mvmean_k^T \mvSigma_k^{-1} \mvmean_k - \frac{1}{2} \ln \det{\mvSigma_k} + \ln(p(C_k))$$

This leads to non-linear second order decision boundaries called hyper quadrics.

![](img/2012-10-03-3.jpg)

(Kafli 4.2.2 í Bishop tengir þetta við parameter estimation)