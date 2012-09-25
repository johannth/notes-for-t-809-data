% Notes
% Jóhann Þorvaldur Bergþórsson & Stefanía Bjarney Ólafsdóttir
% 25. September 2012

# Linear Classification

<!-- ![](img/2012-09-25-1.jpg) -->

Linear equation
$$y(\x) = \w^T \x + w_0$$

Activation function
$$f(y(\x)) \to \{0, 1 \}$$

(Kíkja vel á flokkara á bls 183 því þeir gætu komið á prófi)

# Least Squares for classification

How do we determine the weight vector in our model? One way is LMS, i.e. set an error and minimize.

So we have our linear model for class k:
$$y_k(\x) = \w_k^T \x + w_{k0}$$
when $k=1, 2, \dotsc, K$. Let's define
$$\tilde{W} = \begin{bmatrix}
    w_{10} & \hdots & w_{k0} & \hdots & w_{K0} \\
    \vdots &        &  \vdots &       & \vdots \\
    \w_1   & \hdots & \w_t    &       &  \w_K \\
    \vdots &        & \vdots  &       &  \vdots \\
\end{bmatrix}
$$
\text{and}

$$\tilde{\x} = \begin{bmatrix} 1 \\ \vdots \\ \x \\ \vdots \end{bmatrix}$$

We can then write our determinant functions in one equation:
$$\y(\x) = \tilde{W}^T \tilde{\x}$$
We want to determine $\tilde{W}$ by minimizing a sum-of-squares error function. We have a training set $\{\x_n, \t_n\}$ with $n=1, \dotsc, N$ and we define
$$T = \begin{bmatrix} \t_1^T \\ \t_2^T \\ \vdots \\ \t_N^T \end{bmatrix}$$
and
$$\tilde{X} = \begin{bmatrix} \tilde{\x}_1^T \\ \vdots \\ \tilde{\x}_N^T \end{bmatrix}$$

If $\x_n$ belongs to class $k$ then $\t_n = \begin{bmatrix} 0 \\ \vdots \\ 1 \\ \vdots \end{bmatrix}$$
This allows us to write the error of all the discriminant functions in one
$$\tilde{X} \tilde{W} - T$$
and the sum-of-squares error function can be written as
$$E_D(\tilde{W}) = \frac{1}{2} \Tr \{ (\tilde{X}\tilde{W} - T)^T(\tilde{X}\tilde{W} - T)\}$$
If we differentiate this and set to zero then we obtain:
$$\tilde{W} = \underbrace{(\tilde{X}^T \tilde{X})^{-1} \tilde{X}^T}_{\tilde{X}^\dagger \text{- pseudo inverse of } \tilde{X}} T = \tilde{X}^\dagger T$$

## Pros and cons of least squares

### Pros

* Closed form solution.
* The values of $\y(\x)$ sum up to $1$.
* It's an approximation to $E[\t| \x]$

### Cons

* Don't take values between $(0, 1)$.
* Lacks robustness to outliers and special class geometry

<!-- ![LMS lacks robustness for outliers](img/2012-09-25-2.jpg) -->

<!-- ![LMS lacks robustness for special class geometry](img/2012-09-25-3.jpg) -->

It shouldn't be surprising that LMS corresponds to maximum likelihood assumptions of Gaussian conditional distribution $p(\t|\x)$ but since $\t$ takes binary values we know that the Gaussian assumption is pretty bad.

