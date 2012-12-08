% Notes
% Jóhann Þorvaldur Bergþórsson & Stefanía Bjarney Ólafsdóttir
% 14. November 2012

Algorithm for Principal Component Analysis
=======================

1. Evaluate $\bar{\x}$ and $\delta$.
2. Find $M$ eigenvectors from $M$ largest eigenvalues.
3. Project $\y = U^T \x$.

We wanted to maximize
$$\u_1^T S \u_1$$
(variance in first principal subspace)

For $M$-dimensions
$$U^T S U \rightarrow \begin{cases} \text{trace} \\ \text{det} \end{cases}$$

Minimum-error formulation
----------------------------
(See picture 12.2, pp. 561)

Alternate formulation based on projection minimization. We have a bases whose vectors satisfy
$$\u_i^T \u_j = \delta_{ij} = \begin{cases}
    1, & i = j \\
    0, & i \neq j
\end{cases}$$
(orthonormal basis).

We can write each datapoint as
$$\x_n = \sum_{i=1}^{D}\alpha_{ni} \u_i$$
where $\alpha_{ni}$ differ for each datapoint (new coordinates). $\begin{bmatrix}
    x_1, x_2, \cdots, x_D
\end{bmatrix}_n^T$ is replaced by $\begin{bmatrix}
    \alpha_1, \alpha_2, \cdots, \alpha_D
\end{bmatrix}_n^T$.

The orthonormality property gives
$$\alpha_{nj} = \x_n^T \u_j$$
so we can write
$$\x_n = \sum_{n=1}^{D} (\x_n^T \u_i) \u_i$$

Now we approximate
$$\tilde{\x}_n = \sum_{i=1}^{M} z_{ni}\u + \sum_{i=M+1}^{D} b_i \u_i$$
Note that $b_i$ do not depend on $n$ or ($\x_n$). We are free to choose $\u_i$, $z_{ni}$ and $b_i$ to minimize the distortion between $\x_n$ and $\tilde{\x}_n$
$$J = \frac{1}{N} \sum_{n=1}^{N} \norm{\x_n - \tilde{\x}_n}^2$$
Differentiating with respect to $z_{ni}$ and $b_i$ and setting to zero gives
$$z_{nj} = \x_n^T \u_j, \qquad j=1,2, \cdots, M$$
$$b_i = \bar{\x}^T \u_j, \qquad j=M+1, \dotsc, D$$
and when we plug this into the equations for $\x_n$ and $\tilde{\x}_n$ we get
$$\x_n - \tilde{\x}_n = \sum_{i=M+1}^{D} \left ((\x_n - \bar{\x})^T \u_i \right ) \u_i$$
i.e. the displacement vector lies in the space orthogonal to the principal subspace. We can rewrite the distortion measure
\begin{align*}
J &= \frac{1}{N} \sum_{n=1}^{N} \sum_{i=M+1}^{D}(\x_n^T \u_i - \bar{\x}^T \u_i)^2 \\
&= \sum_{i=M+1}^{D}\u_i^T S \u_i
\end{align*}
so
$$S= \frac{1}{N} \sum_{n=1}^{N} (\x_n - \bar{\x}) (\x_n - \bar{\x})^T$$
is a sample covariance.

When we minimize $J$ we obtain $M$ eigenvectors $\u_i$ of the covariance matrix $S$ corresponding to the $M$ largest eigenvalues. The corresponding distortion $J$ is given by
$$J = \sum_{i=M+1}^{D} \lambda_i$$

<!-- ![](img/2012-11-14-1.jpg) -->

(Mögulega spurt í smáatriðum út í 12.1. Prófar úr 12.1.3 en ekki 12.1.4.)

The purpose of all this
----------------------------

We have data $\{ \x \}$ which we can transform into another dataset $\{ \x_n \}'$ such as  $\bar{\x}' = \0$ and $U$ is an eigenvector matrix from $S$.
$$\z_n = U \x_n$$