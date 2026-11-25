# Contributing to L2ODLL.jl

Thanks for your interest! Issues and pull requests are welcome.

- **Tests.** Run `julia --project -e 'using Pkg; Pkg.test()'`. Every mathematical
  claim is a named testset in `test/guarantees.jl` (see the Theory page of the
  docs); a new decomposition or completion should add instances to
  `test/instances.jl` and pass `ValidDualBound`, `TightAtOptimalDual` and the
  gradient checks. The QA tests (Aqua, JET, ExplicitImports) must pass.
- **Docs.** `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`, then
  `julia --project=docs docs/make.jl`. Doctests run; every exported name needs a
  docstring. Cite the equation of the paper a formula comes from.
- **Style.** BlueStyle-like formatting via `.JuliaFormatter.toml` for new code;
  keep diffs focused (one concern per commit).
- **Unregistered dependency.** The optional L2OCore interface lives in
  `ext/L2ODLLL2OCoreExt.jl`; L2OCore is installed from GitHub through `[sources]`
  (Julia ≥ 1.11), and CI adds it explicitly on Julia 1.10.
