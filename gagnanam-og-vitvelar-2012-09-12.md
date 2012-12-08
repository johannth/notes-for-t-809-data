% Notes
% Jóhann Þorvaldur Bergþórsson & Stefanía Bjarney Ólafsdóttir
% 12. September 2012

# Discriminant Functions

Let's start with two classes, $K=2$. The simplest linear discriminant is
$$y(\x) = \underbrace{\w^T}_{\text{weight vector}} \x + \underbrace{w_0}_{\text{threshold}}$$

An input vector $\x$ is assigned to class $C_1$ if $y(\x) > 0$, else to $C_2$.

Let's consider two points on the decision boundary $\x_A$ and $\x_B$. Then we have
$$y(\x_A) = 0 = y(\x_B)$$
by definition, and therefore
$$\w^T(\x_A - \x_B) = 0$$
so that $\w$ is orthogonal to every vector on the decision boundary.

If $\x$ is on the decision boundary surface then $y(\x) = 0$ and
$$\frac{\w^T \x}{\norm{\w}} = - \frac{w_0}{\norm{\w}}$$
is the normal distance from the origin to the decision surface.

$y(\x)$ gives a signed measure of the perpendicular distance $r$ of the point $\x$ to the decision surface. To see this we have $\x_{\perp}$, the orthogonal projection onto the decision surface.

![](img/2012-09-12-4.jpg)
$$\x = \x_\perp + r \frac{\w}{\norm{\w}}$$
which is equivalent to
$$\w^T\x = \w^T \x_\perp + \frac{r \w^T\w}{\norm{\w}}$$
so that
$$\underbrace{\w^T\x + w_0}_{y(\x)} = \w^T \x_\perp + \underbrace{\frac{r \w^T \w}{\norm{\w}}}_{r\norm{\w}} + w_0$$
and finally
$$r = \frac{y(\x)}{\norm{\w}}$$

![](img/2012-09-12-5.jpg)
So instead of build a multiclass discriminant from binary classifiers, let's consider a $K$ class discriminant comprising of $k$ linear functions
$$Y_k(\x) + \w_k^T \x + w_{k0}$$
and then assign $\x$ to class $C_k$ if $Y_k(\x) \ge Y_j(\x)$ for all $j,k$.

![](img/2012-09-12-6.jpg)

We get the decision boundary between $C_k$ and $C_j$ by solving
$$Y_k(\x) = Y_j(\x)$$
which corresponds to the $D-1$ dimensional hyperplane
$$(\w_k - \w_j)\x + (w_{k0} - w_{j0}) = 0$$
We have the same properties as in the binary case plus the following properties

1. Simply connected.
2. Convex.

![](img/2012-09-12-7.jpg)


