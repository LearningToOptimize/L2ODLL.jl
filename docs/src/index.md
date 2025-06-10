# L2ODLL.jl

!!! warning
    This documentation is a work in progress.
    Please open an issue if content is missing / erroneous

L2ODLL.jl implements the Dual Lagrangian Learning (DLL) method of [Tanneau and Hentenryck (2024)](https://arxiv.org/pdf/2402.03086) using JuMP.

## Installation

```julia
import Pkg
Pkg.add(; url="https://github.com/LearningToOptimize/L2ODLL.jl")
```

## Math Background

This package simplifies the implementation of DLL by taking as input a primal JuMP model, then automatically generating the dual projection and completion functions which can be used in the training and inference of DLL models.


### Decomposition

In DLL, the primal constraints (dual variables) are decomposed into a predicted set and a completed set.
Consider the primal-dual pair:
```math
\begin{equation}
\begin{aligned}
& \min\nolimits_{x} & c^\top x
\\
& \;\;\text{s.t.} & Ax + b \in \mathcal{C}
\\
& & x \in \mathbb{R}^n
\end{aligned}
\quad\quad\quad\quad
\begin{aligned}
& \max\nolimits_{y} & - b^\top y
\\
& \;\;\text{s.t.} & A^\top y = c
\\
& & y \in \mathcal{C}^*
\end{aligned}
\end{equation}
```
After the decomposition, we have:
```math
\begin{equation}
\begin{aligned}
& \min\nolimits_{x} & c^\top x
\\
& \;\;\text{s.t.} & Ax + b \in \mathcal{C}
\\
& \;\;\phantom{\text{s.t.}} & Hx + h \in \mathcal{K}
\\
& & x \in \mathbb{R}^n
\end{aligned}
\quad\quad\quad\quad
\begin{aligned}
& \max\nolimits_{y} & - b^\top y
\\
& \;\;\text{s.t.} & A^\top y + H^\top z = c
\\
& & y \in \mathcal{C}^*,\; z \in \mathcal{K}^*
\end{aligned}
\end{equation}
```

Then, the completion model is:

```math
\begin{equation}
\begin{aligned}
& \max\nolimits_{z} & - h^\top z - b^\top y
\\
& \;\;\text{s.t.} & H z = c - A^\top y
\\
& & z \in \mathcal{K}^*
\end{aligned}
\end{equation}
```

To train the neural network, we need the gradient of the optimal value with respect to the predicted $y$. This is $\nabla_y = -b-Ax$ where $x$ is the optimal dual solution corresponding to the affine constraints in the completion model. In the special cases below, we specify just the expression for $x$ in this formula.


#### Bounded Decomposition

When all primal variables have finite upper and lower bounds, a natural way to decompose the constraints is to have $z$ correspond to the bound constraints, and $y$ correspond to the main constraints, i.e.

```math
\begin{equation}
\begin{aligned}
& \min\nolimits_{x} & c^\top x
\\
& \;\;\text{s.t.} & Ax + b \in \mathcal{C}
\\
& & l \leq x \leq u
\end{aligned}
\quad\quad\quad\quad
\begin{aligned}
& \max\nolimits_{y,z_l,z_u} & - b^\top y - l^\top z_l - u^\top z_u
\\
& \;\;\text{s.t.} & A^\top y + I z_l + I z_u = c
\\
& & y \in \mathcal{C}^*,\; z_l \in \mathbb{R}_+^n,\; z_u \in \mathbb{R}_-^n
\end{aligned}
\end{equation}
```

Then, the completion model is:

```math
\begin{equation}
\begin{aligned}
& \max\nolimits_{z_l,z_u} & - l^\top z_l - u^\top z_u - b^\top y
\\
& \;\;\text{s.t.} & I z_l + I z_u = c - A^\top y
\\
& & z_l \in \mathbb{R}_+^n,\; z_u \in \mathbb{R}_-^n
\end{aligned}
\end{equation}
```

This model admits a closed form solution, $z_l = |c-A^\top y|^+$ and $z_u = -|c-A^\top y|^-$. Furthermore, the $x$ that defines the (sub-)gradient is given element-wise by $l$ if $z_l > 0$, $u$ if $z_u < 0$, and $x\in[l,u]$ otherwise.


#### (Strictly) Convex QP

In the convex QP case, the primal has a strictly convex quadratic objective function, i.e. $Q\succ 0$. In that case it is natural to use the main constraints as the predicted set and to complete the quadratic slack dual variables.

```math
\begin{equation}
\begin{aligned}
& \min\nolimits_{x} & x^\top Q x + c^\top x
\\
& \;\;\text{s.t.} & Ax + b \in \mathcal{C}
\\
& & x \in \mathbb{R}^n
\end{aligned}
\quad\quad\quad\quad
\begin{aligned}
& \max\nolimits_{y} & - b^\top y - z^\top Q z
\\
& \;\;\text{s.t.} & A^\top y + Q^\top z = c
\\
& & y \in \mathcal{C}^*,\; z \in \mathbb{R}^n
\end{aligned}
\end{equation}
```

Then, the completion model is:

```math
\begin{equation}
\begin{aligned}
& \max\nolimits_{z} & - z^\top Q z - b^\top y
\\
& \;\;\text{s.t.} & Q z = c - A^\top y
\\
& & z \in \mathbb{R}^n
\end{aligned}
\end{equation}
```

This model admits a closed form solution, $z = Q^{-1}(c - A^\top y)$. Furthermore, the closed form dual solution in this case is $x=z$.
