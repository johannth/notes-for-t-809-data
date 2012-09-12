% Notes
% Jóhann Þorvaldur Bergþórsson & Stefanía Bjarney Ólafsdóttir
% 11. September 2012

# Linear Models For Classification

True goal in in classification is take an input vector $\x$ and assign it to one or $K$ discreet classes $C_k$, $k=1, \dotsc, K$. The input space is divided into regions whose boundaries are called **decision surfaces**.

![](img/2012-09-11-1.jpg)

If $\x$ is $D$ dimensional then the decision surfaces are $D-1$ dimensional.

We still have observations $(\x_n, t_n)$ but now instead of taking a continuous value $t_n$ has to indicate a class. One possible way of doing this is to have $t_n \in \{1,2, \dotsc, K \}$. Another way would be to set
$$t_3 = \begin{bmatrix}
    0 \\ 0 \\ 1  \\ 0 \\ 0 \\ 0 \\ 0
\end{bmatrix}$$

From decision theory we have three different ways of making a decision.

1. Using a discriminant function.
2. Solve inference problem by modeling $p(C_k| \x)$ directly.
3. Solve inference problem by modeling $p(\x|C_l)$ and then applying Bayes theorem.

![](img/2012-09-11-2.jpg)

Often we cannot model $p(C_k | \x)$ directly but we know that
$$p(C_k | \x) = \frac{p(\x|C_k) p(C_k)}{p(\x)}$$
![](img/2012-09-11-3.jpg)

In regression we had $y(\x,\w) = \w^T \x + w_0$. For classification we need to generalize this using an **activation function**
$$y(\x, \w) = f(\w^T\x + w_0)$$
which maps the real number $\w^T \x + w_0$ to $0$ or $1$ or a posterior probability.
![](img/2012-09-11-4.jpg)

Decision surfaces will correspond to $y(\x) = C$, where $C$ is a constant, so that $\w^T\x + w_0 = C$ implies that decision surfaces are linear function of $\x$ even though $f$ is always non linear.

