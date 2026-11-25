# Theory

This page states the optimization problems behind L2ODLL, the assumptions under
which its outputs are guaranteed to be valid, and the name of the numerical test
that checks each claim. References:

- **[DLL]** M. Tanneau and P. Van Hentenryck, *Dual Lagrangian Learning for Conic
  Optimization*, NeurIPS 2024, [arXiv:2402.03086](https://arxiv.org/abs/2402.03086).
- **[DIPOL]** M. Klamkin, M. Tanneau and P. Van Hentenryck, *Dual Interior Point
  Optimization Learning*, [arXiv:2402.02596](https://arxiv.org/abs/2402.02596).

Equation numbers such as "(DLL 9)" refer to the arXiv versions of these papers.

## Notation

| symbol | meaning |
|:--|:--|
| ``p \in \mathbb{R}^{d}`` | parameters (JuMP `MOI.Parameter` variables); the input of the proxy |
| ``x \in \mathbb{R}^n`` | primal decision variables |
| ``\mathcal{C}, \mathcal{K}`` | closed convex cones (products of MOI sets: ``\mathbb{R}_+``, ``\{0\}``, second-order, PSD, …) |
| ``\mathcal{C}^*`` | dual cone, ``\{y : y^\top s \ge 0\ \forall s \in \mathcal{C}\}`` (DLL 2) |
| ``y`` | **predicted** dual variables (one block per constraint in [`get_y`](@ref)) |
| ``z`` | **completed** dual variables (one block per constraint in [`get_z`](@ref)) |
| ``w`` | quadratic slack variables of the dual (convex quadratic objectives) |
| ``\Pi_{\mathcal{C}^*}`` | Euclidean projection onto ``\mathcal{C}^*`` (DLL 1) |
| ``\lvert v\rvert^+ = \max(v, 0)``, ``\lvert v\rvert^- = \max(-v, 0)`` | elementwise |

**Sign convention.** L2ODLL follows MathOptInterface: a constraint is written as
"function in set", ``Ax + b \in \mathcal{C}``, and its dual is ``y \in
\mathcal{C}^*`` with the signs returned by `JuMP.dual`. The papers write
``Ax \succeq_{\mathcal{K}} b``; the two forms coincide with ``b \mapsto -b``.
Maximization problems are handled by Dualization.jl, so the predicted ``y`` always
has `JuMP.dual` signs.

## Parametric primal–dual pair

For fixed parameters ``p`` consider the conic problem and its dual (DLL 4–5):

```math
\begin{equation}
\begin{aligned}
& \min\nolimits_{x} & c(p)^\top x
\\
& \;\;\text{s.t.} & Ax + b(p) \in \mathcal{C}
\end{aligned}
\quad\quad\quad\quad
\begin{aligned}
& \max\nolimits_{y} & - b(p)^\top y
\\
& \;\;\text{s.t.} & A^\top y = c(p)
\\
& & y \in \mathcal{C}^*
\end{aligned}
\tag{1}
\end{equation}
```

**Assumption (A1): parameters enter affinely in the data ``b(p)`` and ``c(p)``**
(constraint constants and linear objective coefficients, e.g. `@objective(m, Max, μ' * x)`
with parameters `μ`). The constraint matrix ``A`` and the quadratic objective ``Q``
do not depend on ``p``.

**Weak duality.** For every ``p`` and every *dual-feasible* ``y`` (``A^\top y =
c(p)``, ``y \in \mathcal{C}^*``), ``-b(p)^\top y \le`` the optimal value of the
primal. No convexity of the parameter dependence and no constraint qualification is
needed. Strong duality (equality at the optimum) holds under Slater's condition.

## Decomposition and completion

Split the constraints into a **predicted** block (cone ``\mathcal{C}``, duals ``y``)
and a **completed** block ``Hx + h(p) \in \mathcal{K}`` (duals ``z``) (DLL 6–7):

```math
\begin{equation}
\begin{aligned}
& \max\nolimits_{y,z} & - b(p)^\top y - h(p)^\top z
\\
& \;\;\text{s.t.} & A^\top y + H^\top z = c(p)
\\
& & y \in \mathcal{C}^*,\; z \in \mathcal{K}^*
\end{aligned}
\tag{2}
\end{equation}
```

Given a prediction ``\hat y``, DLL projects it, ``\bar y = \Pi_{\mathcal{C}^*}(\hat y)``,
and **completes** ``z`` by solving (DLL 9), whose dual is (DLL 8):

```math
\begin{equation}
\begin{aligned}
\Phi(\bar y, p) = & \max\nolimits_{z} & - h(p)^\top z - b(p)^\top \bar y
\\
& \;\;\text{s.t.} & H^\top z = c(p) - A^\top \bar y
\\
& & z \in \mathcal{K}^*
\end{aligned}
\quad\quad\quad\quad
\begin{aligned}
& \min\nolimits_{x} & (c(p) - A^\top \bar y)^\top x - b(p)^\top \bar y
\\
& \;\;\text{s.t.} & Hx + h(p) \in \mathcal{K}
\end{aligned}
\tag{3}
\end{equation}
```

[`dual_objective`](@ref) returns ``\Phi(\bar y, p)``; [`dual_completion`](@ref)
returns ``(\bar y, z, w)``.

**Theorem (DLL Theorem 1, dual conic completion).** *Assume that for every
``\bar y \in \mathcal{C}^*`` the right-hand problem in (3) is bounded and strictly
feasible. Then the left-hand problem in (3) has an optimal ``z``, and ``(\bar y,
z)`` is feasible for (2).*

**Theorem (DLL Theorem 2, optimal completion).** *Under the same assumptions,
``\Phi(\bar y, p)`` is a valid bound on the primal optimum, and it is the best
bound among all dual-feasible points whose ``y``-block equals ``\bar y``.*

In particular ``\Phi(y^\star, p)`` equals the optimal value when ``y^\star`` is the
``y``-block of an optimal dual solution.

## Closed-form completions

The assumption of Theorem 1 holds, and (3) has a closed form, in two structured
cases that L2ODLL detects automatically ([decomposition detection](background/decomposition_detection.md)).

**Bounded variables (DLL Example 1, eqs. 11–18; [`BoundDecomposition`](@ref)).** If
``l \le x \le u`` with finite ``l < u`` and the objective is affine in ``x``, take
the bounds as the completed block. With ``r = c(p) - A^\top \bar y``,

```math
z_l = \lvert r\rvert^+, \qquad z_u = -\lvert r\rvert^-,
\qquad
\Phi(\bar y, p) = -b(p)^\top \bar y + l^\top z_l + u^\top z_u .
\tag{4}
```

(L2ODLL keeps ``z_u \le 0``, the MOI sign of the dual of ``x \le u``; DLL writes
``z^u = \lvert r\rvert^- \ge 0`` with a minus sign in the objective.)

**Log-barrier completion (DIPOL Theorem 2, eq. 7).** Replacing the completion
objective by ``-h^\top z + \mu \sum_j (\ln z_{l,j} + \ln(-z_{u,j}))`` gives, with
``v = \mu / (u - l)`` and ``\omega = r/2``,

```math
z_l = v + \omega + \sqrt{v^2 + \omega^2}, \qquad
z_u = -v + \omega - \sqrt{v^2 + \omega^2}.
\tag{5}
```

It still satisfies ``z_l + z_u = r`` with ``z_l > 0 > z_u``, so the bound (4)
evaluated at (5) is valid, and it tends to (4) as ``\mu \to 0``; it is smooth in
``\bar y``. It is available as `L2ODLL.LogBoundedCompletion` in
`L2ODLL.bounded_builder(...; completion = :log, μ)`.

**Strictly convex quadratic objective (DLL Example 3, eqs. 21–23; [`ConvexQP`](@ref)).**
For ``\min \tfrac12 x^\top Q x + c(p)^\top x`` with ``Q \succ 0`` the dual has a free
block ``w`` with stationarity ``A^\top y - Q w = c(p)`` (the sign convention of
Dualization.jl). Predicting all constraint
duals and completing ``w``,

```math
w = Q^{-1}\big(A^\top \bar y - c(p)\big), \qquad
\Phi(\bar y, p) = -b(p)^\top \bar y - \tfrac12 w^\top Q w .
\tag{6}
```

``w`` is the minimizer of the Lagrangian, ``x(\bar y)``; at an optimal ``y^\star`` it is
the (unique) primal optimum. (The code reads the coefficient matrix of ``w`` from the
dual written by Dualization.jl, so the ``\tfrac12`` convention of the model does not
matter.)

## Projection

The projection ``\Pi_{\mathcal{C}^*}`` is computed per constraint with
MathOptSetDistances.jl (Euclidean projection, DLL eq. 1): `max(y, 0)` for
``\mathbb{R}_+``, the identity for equalities, the closed-form projection for
second-order cones, eigenvalue clipping for PSD cones. DLL also proposes a *radial*
projection (DLL 26) that avoids zero gradients on the polar cone; any map into
``\mathcal{C}^*`` keeps the guarantees, and a custom one can be passed as `proj_fn`
to [`decompose!`](@ref) (see the package guide).

## Gradient of the completed dual objective

Training needs ``\nabla_{\hat y}\,\Phi(\Pi_{\mathcal{C}^*}(\hat y), p)``.

**Theorem (DIPOL Theorem 3).** *For the bounded completion (4)–(5) with ``\mu \ge
0``, ``\Phi`` is concave in ``\bar y`` and ``-b(p) - A\hat x`` is a supergradient,
where ``\hat x`` is an optimal dual solution of the equality constraints in (3)
(the right-hand problem). For ``\mu > 0`` it is differentiable.*

**Theorem (DIPOL Theorem 4).** *For ``\mu = 0``, ``\hat x_j = l_j`` if ``r_j > 0``,
``\hat x_j = u_j`` if ``r_j < 0`` and any ``\hat x_j \in [l_j, u_j]`` if ``r_j = 0``.*

In the convention of the papers (``Ax \ge b``) this reads ``b - A\hat x``. The same
argument (partial Lagrangian of (3)) gives ``-b(p) - A\hat x`` for the generic
completion, with ``\hat x`` the dual of the equality constraints of the completion
problem (issue #5). When the network also predicts a quadratic slack ``w`` (bounded
decomposition with a quadratic objective, not implemented yet) its gradient with respect
to ``w`` is ``Q(\hat x - w)`` for the objective ``\tfrac12 x^\top Q x`` (see
[decomposition detection](background/decomposition_detection.md)).

L2ODLL differentiates the closed forms (4) and (6) with automatic differentiation
(ForwardDiff by default via DifferentiationInterface; Zygote also works), which
yields the same (super)gradient almost everywhere.

## Training problem

The proxy ``\hat y = f_\theta(p)`` is trained self-supervised by maximizing the
expected bound (DLL 27a; DIPOL 3):

```math
\max_\theta\; \mathbb{E}_{p}\big[\Phi(\Pi_{\mathcal{C}^*}(f_\theta(p)), p)\big].
\tag{7}
```

No optimal solutions are needed. For a *maximization* primal the dual is a
minimization and (7) becomes a minimization. In the L2OCore interface the loss is
always minimized, so it is ``-\text{mean}\,\Phi`` for minimization problems and
``+\text{mean}\,\Phi`` for maximization problems.

## Guarantees and their checks

Each claim is checked numerically in the test suite under the testset of the same
name (`test/guarantees.jl`, `test/l2ocore.jl`), on seeded random instances of every
supported class: bounded LPs, bounded SOCPs and strictly convex QPs (with and
without bounds), minimization and maximization, with parameters in objective
coefficients, right-hand sides and cone constants.

| name | claim | assumptions | reference |
|:--|:--|:--|:--|
| `ValidDualBound` | ``\Phi(\Pi(\hat y), p) \le \mathrm{OPT}(p)`` (``\ge`` for max) for **every** ``\hat y`` | (A1); Theorem 1's assumption (automatic for the closed forms) | DLL Thms 1–2; `L2OCore.ValidDualBound` |
| `TightAtOptimalDual` | ``\Phi(y^\star, p) = \mathrm{OPT}(p)`` | strong duality | DLL Thm 2 |
| `ProjectionIdempotent` | ``\Pi(\Pi(\hat y)) = \Pi(\hat y)`` | — | DLL eq. 1 |
| `ClosedFormMatchesSolver` | (4), (6) equal the solution of (3) computed by a solver | as above | DLL Examples 1, 3 |
| `GradientMatchesFiniteDifferences` | AD gradient = central differences | ``\hat y`` off the kinks of (4) (almost surely) | — |
| `GradientMatchesTheorem3` | ``\nabla = -b(p) - A\hat x`` with ``\hat x`` from Theorem 4 | bounded LP, ``\hat y`` in the interior of ``\mathcal{C}^*`` | DIPOL Thms 3–4 |
| `LogBarrierCompletionFeasible` | (5) satisfies ``z_l + z_u = r``, ``z_l > 0 > z_u``, → (4) as ``\mu \to 0`` | ``l < u`` | DIPOL Thm 2 |
| `CoreConsistency` | L2OCore's independent Lagrangian bound of the completed duals = ``\Phi``, dual infeasibility ``\le 10^{-8}`` | — | — |
| `BatchMatchesColumns` | matrix API = per-instance API | — | — |
| `ReverseModeAD` | Zygote gradient of the L2OCore loss = ForwardDiff | — | — |
| `StrictConvexityRequired` | `ConvexQP` is used only when ``Q \succ 0`` (``\prec 0`` for max); otherwise detection fails and the builder errors | — | DLL Example 3 |
| `ParametersInCoefficientsRejected` | a parameter multiplying a variable in a constraint raises an error | violates (A1) | — |
| `ObjectiveConstantsIncluded` | constants and parameter-only objective terms are part of ``\Phi`` (tightness holds) | — | — |
| `ConcaveInY` | ``\Phi`` is concave on ``\mathcal{C}^*`` (convex for max) | — | DIPOL Thm 3 |
| `OptimalCompletion` | for fixed ``\bar y`` the exact completion beats any other feasible one (the log-barrier one) | — | DLL Thm 2 |
| `BoundedClosedFormEq4` | ``\Phi`` equals eq. (4) computed by hand from the problem data | — | DLL Example 1 |
| `ConvexQPClosedFormEq6` | ``\Phi`` and ``w`` equal eq. (6) computed by hand | ``Q \succ 0`` | DLL Example 3 |
| `QuadSlackRecoversPrimal` | ``w(y^\star) = x^\star`` | ``Q \succ 0`` | DLL Example 3 |
| `LogBarrierGradientTheorem5` | eq. (5) maximizes the barrier completion problem and its value has derivative ``x_\mu(r)`` (closed form of DIPOL Thm 5) | ``l < u``, ``\mu > 0`` | DIPOL Thms 3, 5 |

## Deviations from the papers

- **Signs.** MOI convention throughout (see above); both forms define the same bound.
- **Projection.** Euclidean projection instead of DLL's preferred radial projection;
  a custom `proj_fn` can implement the radial one.
- **Bounded decomposition with a quadratic objective.** The background pages describe
  predicting ``(y, w)`` and completing the bounds; this is not implemented yet. Such
  models are detected as [`ConvexQP`](@ref) (if the objective is strictly convex),
  and requesting `BoundDecomposition` explicitly raises an error.
- **Log-barrier layer.** With `completion = :log`, the layer returns the plain dual
  objective (4) evaluated at the smoothed completion (5): a valid bound, smooth in
  ``\bar y``. DIPOL trains with the barrier-augmented objective (DIPOL eq. 4), whose
  supergradient is ``-b - A x_\mu`` (Theorems 3, 5); the plain objective has a
  different gradient. The barrier-augmented value can be built from
  `L2ODLL.complete_zlzu` as in the `LogBarrierGradientTheorem5` test.
- **Infeasible generic completions.** Only the two closed forms satisfy Theorem 1's
  assumption by construction. A `GenericDecomposition` completion may be infeasible
  (issue #6); the JuMP layer then throws instead of returning a bound.
