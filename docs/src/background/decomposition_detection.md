# Decomposition detection

!!! warning
    The decomposition detection is very basic, based on minimizing the number of predicted dual variables. In some cases, it may actually be preferred to to predict more dual variables, if they are for some reason easier to learn. In that case, users should manually specify the decomposition.

L2ODLL automatically detects the decomposition to use based on the model.
The current detection logic is as follows:

- If all variables have finite upper and lower bounds **and the objective is affine in the decision variables**, use the bounded decomposition.
- If all variables have a quadratic objective term, use the convex QP decomposition.
- Otherwise, no closed form is available: build a `GenericDecomposition` by hand and pass an `optimizer` to `decompose!`.

The order of preference is to maximize the number of completed dual variables among the closed-form solutions. The analysis below shows that for a convex QP with bounds on every variable, the bounded decomposition would complete more dual variables than the convex QP decomposition.

!!! note "Current behavior"
    The bounded decomposition with a quadratic objective (predicting $y$ and $w$, completing $z_l, z_u$) is not implemented yet: the network would have to predict the quadratic slack $w$, which is not the dual of a constraint. Such models are therefore detected as `ConvexQP` (when every variable has a quadratic term), and `decompose!(model, BoundDecomposition(model))` raises an error for them.

## Bounded with quadratic objective

In the case where bounded decomposition is used with a model that has a quadratic objective, L2ODLL prefers the bounded decomposition, leading to the following models.
```math
\begin{equation}
\begin{aligned}
& \min\nolimits_{x} & x^\top Q x + c^\top x
\\
& \;\;\text{s.t.} & Ax + b \in \mathcal{C}
\\
& & l \leq x \leq u
\end{aligned}
\quad\quad\quad\quad
\begin{aligned}
& \max\nolimits_{w,y,z_l,z_u} & - b^\top y - w^\top Q w + l^\top z_l + u^\top z_u
\\
& \;\;\text{s.t.} & A^\top y + 2Qw + z_l + z_u = c
\\
& & y \in \mathcal{C}^*,\; w \in \mathbb{R}^n,\; z_l \in \mathbb{R}_+^n,\; z_u \in \mathbb{R}_-^n
\end{aligned}
\end{equation}
```

Then, the completion model is:

```math
\begin{equation}
\begin{aligned}
& \max\nolimits_{z_l,z_u} & + l^\top z_l + u^\top z_u - b^\top y - w^\top Q w
\\
& \;\;\text{s.t.} & z_l + z_u = c - A^\top y - 2Qw   
\\
& & z_l \in \mathbb{R}_+^n,\; z_u \in \mathbb{R}_-^n
\end{aligned}
\quad\quad\quad\quad
\begin{aligned}
& \min\nolimits_{x} & (c-A^\top y - 2Qw)^\top x - b^\top y - w^\top Q w
\\
& & l \leq x \leq u
\end{aligned}
\end{equation}
```

This model admits a closed form solution, $z_l = |c-A^\top y-2Qw|^+$ and $z_u = -|c-A^\top y-2Qw|^-$. Furthermore, the $x$ that defines the (sub-)gradient is given element-wise by $l$ if $c-A^\top y-2Qw > 0$, $u$ if $c-A^\top y-2Qw < 0$, and $x\in[l,u]$ otherwise. Note that the gradient of the objective with respect to $w$ is $\nabla_w = -(Q+Q^\top)w - 2Q^\top x$.

This completes $2n$ dual variables, leaving the neural network to predict $m+n$ dual variables.


Consider the convex QP decomposition in this case:
```math
\begin{equation}
\begin{aligned}
& \min\nolimits_{x} & x^\top Q x + c^\top x
\\
& \;\;\text{s.t.} & Ax + b \in \mathcal{C}
\\
& & l \leq x \leq u
\end{aligned}
\quad\quad\quad\quad
\begin{aligned}
& \max\nolimits_{w,y,z_l,z_u} & - b^\top y - w^\top Q w + l^\top z_l + u^\top z_u
\\
& \;\;\text{s.t.} & A^\top y + 2Qw + z_l + z_u = c
\\
& & y \in \mathcal{C}^*,\; w \in \mathbb{R}^n,\; z_l \in \mathbb{R}_+^n,\; z_u \in \mathbb{R}_-^n
\end{aligned}
\end{equation}
```

Then, the completion model is:

```math
\begin{equation}
\begin{aligned}
& \max\nolimits_{w} & - w^\top Q w - b^\top y + l^\top z_l + u^\top z_u 
\\
& \;\;\text{s.t.} & Qw = \frac{1}{2}(c - A^\top y - z_l - z_u)   
\\
& & w \in \mathbb{R}^n
\end{aligned}
\quad\quad\quad\quad
\begin{aligned}
& \min\nolimits_{x} & x^\top Q x + (c-A^\top y - z_l - z_u)^\top x - b^\top y + l^\top z_l + u^\top z_u
\\
& & x \in \mathbb{R}^n
\end{aligned}
\end{equation}
```

This model admits a closed form solution, $w = \frac{1}{2}Q^{-1}(c - A^\top y - z_l - z_u)$. Furthermore, the closed form dual solution in this case is $x=w$.

This completes $n$ dual variables, leaving the neural network to predict $m+2n$ dual variables.

Naturally, one may consider using both decompositions, i.e. to predict only the $y$ variables and to recover $w$, $z_l$, and $z_u$. Let us consider this case:
```math
\begin{equation}
\begin{aligned}
& \min\nolimits_{x} & x^\top Q x + c^\top x
\\
& \;\;\text{s.t.} & Ax + b \in \mathcal{C}
\\
& & l \leq x \leq u
\end{aligned}
\quad\quad\quad\quad
\begin{aligned}
& \max\nolimits_{w,y,z_l,z_u} & - b^\top y - w^\top Q w + l^\top z_l + u^\top z_u
\\
& \;\;\text{s.t.} & A^\top y + 2Qw + z_l + z_u = c
\\
& & y \in \mathcal{C}^*,\; w \in \mathbb{R}^n,\; z_l \in \mathbb{R}_+^n,\; z_u \in \mathbb{R}_-^n
\end{aligned}
\end{equation}
```

Then, the completion model is:

```math
\begin{equation}
\begin{aligned}
& \max\nolimits_{w,z_l,z_u} & - w^\top Q w + l^\top z_l + u^\top z_u - b^\top y
\\
& \;\;\text{s.t.} & 2Qw + z_l + z_u = c - A^\top y
\\
& & w \in \mathbb{R}^n,\; z_l \in \mathbb{R}_+^n,\; z_u \in \mathbb{R}_-^n
\end{aligned}
\quad\quad\quad\quad
\begin{aligned}
& \min\nolimits_{x} & x^\top Q x + (c-A^\top y)^\top x - b^\top y
\\
& & l \leq x \leq u
\end{aligned}
\end{equation}
```

This is an $n$-dimensional box-constrained convex QP, for which there is no closed form solution. Note that by using a custom generic decomposition, L2ODLL can still be used to set up this problem and solve it using a JuMP-compatible QP solver.