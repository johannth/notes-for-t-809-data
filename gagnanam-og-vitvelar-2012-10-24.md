% Notes
% Jóhann Þorvaldur Bergþórsson & Stefanía Bjarney Ólafsdóttir
% 24. October 2012

Mixture of Gaussians
======================================

We have the probability of $\x$ being in cluster $k$:
$$p(k) = p(z_k=1) = \pi_k$$
where we have defined the latent (hidden)
$$\z = \begin{bmatrix}
    z_1 \\ z_2 \\ \vdots \\ z_k
\end{bmatrix}$$
where $z_k = 1$ if $\x$ belongs to cluster $k$ but $z_i = 0$ if $i \neq k$. So the probability of $\z$ is
$$p(\z) = \prod_{i=1}^K \pi_i^{z_i} = (\pi_1)^0 (\pi_2)^0 \cdots (\pi_k)^1 \cdots (\pi_K)^0 = \pi_k$$
(Þetta eru fyrirframlíkurnar á að vera í cluster $k$)

We also have
$$p(\x | z_k = 1) = \NormalDist(\x | \mvmean_k, \mvSigma_k)$$
We can also write
$$p(\x|\z) = \prod_{k=1}^K \NormalDist(\x | \mvmean_k, \mvSigma_k)^{z_k}$$
so if we define
$$I_{jk} = \begin{cases}
    1, &j = k \\
    0, &j \neq k\end{cases}$$
\begin{align*}
p(\x) &= \sum_{\z} p(\x | \z) p(\z) \\
    &=\sum_\z \prod_{k=1}^K (\pi_k \NormalDist(\x | \mvmean_k, \mvSigma_k)^{z_k} \\
    &= \sum_{j=1}^K \prod_{k=1}^K (\pi_k \NormalDist(\x | \mvmean_k, \mvSigma_k)^{I_{jk}} \\
    &= \sum_{j=1}^K \pi_j  \NormalDist(\x | \mvmean_j, \mvSigma_j)
\end{align*}

An important definition
\begin{align*}
\gamma(z_k) &:= p(z_k = 1| \x) \\
&= \frac{p(z_k=1)p(\x|z_k=1)}{p(\x)} \\
&= \frac{\pi_k \NormalDist(\x| \mvmean_k, \mvSigma_k)}{\sum_{j=1}^K \pi_j  \NormalDist(\x | \mvmean_j, \mvSigma_j)}
\end{align*}

EM algorithm
------------------------------------------

1. Initialize $\mvmean_k$, $\mvSigma_k$ and $\pi_k$.
2. E-step: Compute
$$\gamma_k(\x_n) = \frac{\pi_k \NormalDist(\x_n| \mvmean_k, \mvSigma_k)}{\sum_{j=1}^K \pi_j  \NormalDist(\x_n | \mvmean_j, \mvSigma_j)}$$
3. M-step: Compute new estimates:
$$N_k = \sum_{n=1}^N \gamma_k(\x_n)$$
$$\mvmean_k = \frac{1}{N_k} \sum_{n=1}^N \gamma_k(\x_n) \x_n$$
$$\mvSigma_k = \frac{1}{N_k} \sum_{n=1}^N \gamma_k(\x_n) (\x_n - \mvmean_k)(\x_n - \mvmean_k)^T$$
$$\pi_k = \frac{N_k}{N}$$
4. Iterate 2-3. In each step compute the likelihood
$$\mathcal{L} = \ln(p(\X | \mvmean, \mvSigma, \pi)) = \sum_{n=1}^N \ln ( \sum_{k=1}^K \pi_k \NormalDist(\x_n| \mvmean_k, \mvSigma_k))$$
    Stopping criterion:
        $$\mathcal{L}^{(\tau + 1)} - \mathcal{L}^{(\tau)} < \epsilon$$


We have the EM algorithm for GMM. We are always trying to maximize
$$\ln p(\X| \theta) = \ln ( \sum_\z p(\X, \z| \theta) )$$
which is difficult (because we cannot simplify the logarithm and the sum)! Here $\X$ is the data set (or the *incomplete data set*), $\z$ the latent variable and $\{\X, \z \}$ the *complete data set*.

The best thing we know about $z$ is
$$p(\z | \X, \theta)$$
$$E_z(p(\X, \z| \theta)) = \sum_\z p(\z| \X, \theta_{old}) \ln(p(\X, \z | \theta)) = Q(\theta, \theta_{old})$$

We can then compute the new parameters with
$$\theta_{new} = \arg \max_{\theta} Q(\theta, \theta_{old})$$

(Hvernig ákvörðunarjaðar myndi K-means flokkari bjóða upp á? Línulega því K-means notar evklíðska fjarlægð, svipað og þegar við gerðum ráð fyrir isotrópískum, eins)

