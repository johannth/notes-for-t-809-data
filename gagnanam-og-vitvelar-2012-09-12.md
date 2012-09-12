% Notes
% Jóhann Þorvaldur Bergþórsson & Stefanía Bjarney Ólafsdóttir
% 12. September 2012

# Discriminant Functions

Let's start with two classes, $K=2$. The simplest linear discriminant is
$$y(\x) = \underbrace{\w^t}_{\text{weight vector}} \x + \underbrace{w_0}_{\text{threshold}}$$

An input vector $\x$ is assigned to class $C_1$ if $y(\x) > 0$, else to $C_2$.

Let's consider two points on the decision boundary $\x_A$ and $\x_B$. Then we have
$$y(\x_A) = 0 = y(\x_B)$$
by definition, and therefore
$$\w^T(\x_A - \x_B) = 0$$
so that $\w$ is orthogonal to every vector on the decision boundary.

If $\x$ is on the decision boundary surface then $y(\x) = 0$ and
$$\frac{\w^T \x}{\norm{\w}} = - \frac{w_0}{\norm{\w}}$$
is the normal distance from the origin to the decision surface.

$y(\x)$ gives a signed measure of the perpendicular distance $r$ of the point $\x$ to the decision surface. To see this we have $\x_{\perp}$
