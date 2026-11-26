# API reference

```@meta
CurrentModule = L2ODLL
```

## Building the DLL layers

```@docs
decompose!
L2ODLL.build_cache
L2ODLL.DLLCache
L2ODLL.get_cache
```

## Evaluating predictions

```@docs
dual_objective
dual_objective_gradient
dual_objective_and_gradient
dual_completion
```

## Decompositions

```@docs
AbstractDecomposition
BoundDecomposition
ConvexQP
GenericDecomposition
L2ODLL.can_decompose
L2ODLL.completed_constraints
get_y
get_z
L2ODLL.get_y_dual
y_shape
L2ODLL.flatten_y
L2ODLL.unflatten_y
L2ODLL.make_completion_model
```

## Layer builders and projections

```@docs
L2ODLL.bounded_builder
L2ODLL.bounded_completion_builder
L2ODLL.convex_qp_builder
L2ODLL.convex_qp_completion_builder
L2ODLL.jump_builder
L2ODLL.jump_completion_builder
L2ODLL.make_proj_fn
L2ODLL.make_jump_proj_fn
```

## L2OCore interface

Defined in the package extension `L2ODLLL2OCoreExt` (load with `import L2OCore`).

```@docs
DLLMethod
```

```@autodocs
Modules = [Base.get_extension(L2ODLL, :L2ODLLL2OCoreExt)]
```
