"""
    at(p, t)

Evaluate a possibly time-varying parameter `p` at time `t`.

Constants (numbers, static arrays, `UniformScaling`) are returned unchanged;
anything else is called as `p(t)`. This lets every model field accept either a
constant or a function of time without the call site having to care which:

```julia
at(100.0, 3.0)          # 100.0        — fixed radius
at(t -> 100 + 20t, 3.0) # 160.0        — growing radius
```

Structs holding such parameters should be parametric on the field types so the
call specializes and inlines rather than dispatching dynamically on `Function`.
"""
@inline at(x::Number, ::Real) = x
@inline at(x::StaticArray, ::Real) = x
@inline at(x::UniformScaling, ::Real) = x
@inline at(f, t::Real) = f(t)
