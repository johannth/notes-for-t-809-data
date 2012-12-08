% Notes
% Jóhann Þorvaldur Bergþórsson & Stefanía Bjarney Ólafsdóttir
% 26. September 2012

# Fisher's linear discriminants

We can view a linear classification model in terms of dimensionality reduction. Let's consider the case where we have two classes and we project $D$-dimensional input $\x$ to one dimension.

$$y = \w^T \x$$

If we then place a threshold on $y$ and classify:
\begin{align*}
    y &\ge - w_0 &\text{then } C_1 \\
    y &< -w_0 &\text{then } C_2
\end{align*}
we get a linear classifier.

When we project to $1$ dimension we lose information. The objective, therefore, is to adjust the weights $\w$ so that class separation in $1$-dimensional is maximized.

We have $N_1$ points from class $C_1$ and $N_2$ from $C_2$ so the sample means are given by:

$$\vec{m}_1 = \frac{1}{N_1} \sum_{n \in C_1} \x_n, \qquad \vec{m}_2 = \frac{1}{N_2} \sum_{n \in C_2} \x_n$$

We could try to adjust $\w$ so that the class means in $1$-dimension are separated.

$$m_2 - m_1 = \w^T \vec{m}_2 - \w^T \vec{m}_1 = \w^T(\vec{m}_2 - \vec{m}_1)$$

This, however, can be made arbitrarily large (by makin $\w$ large).

![](img/2012-09-26-1.jpg)

Fisher proposed to maximize a function that gives a large seperation between the projected class means and a small variance within each class to minimize class overlap. To do this we define the within-class variance of the transformed data from class $C_k$
$$s_k^2 = \sum_{n \in C_k} (y_n - m_k)^2$$

The within class variance for the whole data is simply $s_1^2 + s_2^2$.

So we want to optimize
\begin{align*}
    J(\w) &= \frac{(m_2 - m_1)^2}{s_1^2 + s_2^2} \\
     &= \frac{\w^TS_B\w}{\w^T S_W \w}
\end{align*}
where $S_B$ is the between-class covariance matrix
$$S_B = (\vec{m}_2 - \vec{m}_1)(\vec{m}_2 - \vec{m}_1)^T$$
and $S_W$ is the within-class covariance matrix
$$S_W = \sum_{n \in C_1}(\x_n - \vec{m}_1)(\x_n - \vec{m}_2)^T + \sum_{n \in C_2}(\x_n - \vec{m}_1)(\x_n - \vec{m}_2)^T$$
If we differentiate and set to $\0$ we get
$$\w = S_W^{-1}(\vec{m}_2 - \vec{m}_1)$$


# Linear models for classification

![](img/2012-09-26-2.jpg)

<!-- ![](img/2012-09-26-3.jpg) -->
Let's consider a single $k$ class discriminant comprising of $K$ linear functions:
$$y_k(\x) = \w_k^T \x + w_{k0}.$$
and then assign $\x$ to class $C_k$ if $y_k(\x) \ge y_j(\x)$ for all $j \ne k$.

We get a decision boundary between $C_k$ and $C_j$ by $y_k(\x) = y_j(\x)$ which corresponds to a $(D-1)$ dimensional hyperplane:
$$(\w_k - \w_j)^T \x + (w_{k0} - w_{j0}) = 0$$

One extra property: Decision regions of such a discriminants are always (see p 184):

1. Singly connected
2. Convex

