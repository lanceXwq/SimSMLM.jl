"""
    AbstractDeformation{D}

A map from *material* coordinates to *world* coordinates at a given time.

A label is sampled once onto a reference configuration and keeps its material coordinate `u` forever; the deformation is what moves it. This Lagrangian split is what makes a moving structure coherent — the same physical label reappears at successive times, rather than being resampled into an unrelated spot.

Subtypes implement:

    place(d::AbstractDeformation{D}, u::SVector{D}, t) -> SVector{D}
"""
abstract type AbstractDeformation{D} end

"""
    place(d, u, t) -> SVector

World position at time `t` of the material point `u`.
"""
function place end

ndim(::AbstractDeformation{D}) where {D} = D

# ---------------------------------------------------------------------------

"""
    Static{D}()

The identity deformation: material coordinates *are* world coordinates. Use for a structure that does not move.
"""
struct Static{D} <: AbstractDeformation{D} end

Static(D::Integer) = Static{Int(D)}()

@inline place(::Static{D}, u::SVector{D}, ::Real) where {D} = u

# ---------------------------------------------------------------------------

"""
    Similarity{D}(; scale=1, rot=I, shift=zeros)

Scale, then rotate, then translate — each factor either a constant or a function of time. Covers the common case of a structure that grows, spins and drifts:

```julia
# a sphere whose radius grows 20 nm/s while its centre slides along x
Similarity{3}(scale = t -> 100 + 20t, shift = t -> SA[10t, 0.0, 0.0])
```

Applied to material points on a *unit* shape, `scale` is the physical size.
"""
struct Similarity{D,S,R,C} <: AbstractDeformation{D}
    scale::S
    rot::R
    shift::C
end

function Similarity{D}(; scale=1, rot=I, shift=zero(SVector{D,Float64})) where {D}
    return Similarity{D,typeof(scale),typeof(rot),typeof(shift)}(scale, rot, shift)
end

@inline function place(m::Similarity{D}, u::SVector{D}, t::Real) where {D}
    at(m.rot, t) * (at(m.scale, t) .* u) .+ at(m.shift, t)
end

"""
    Drift(shift)
    Drift{D}(shift)

Pure translation — `shift` is a constant offset or a function `t -> SVector`. The dimension is inferred from a constant `shift`; give it explicitly when `shift` is a function.

Kept as its own type rather than a [`Similarity`](@ref) with default scale and rotation partly for speed, but mostly so that sample drift stays a distinct, separately recoverable layer from the structure's own motion:

```julia
Drift{3}(t -> SA[50t, 0.0, 0.0]) ∘ structure_motion
```
"""
struct Drift{D,C} <: AbstractDeformation{D}
    shift::C
end

Drift{D}(shift) where {D} = Drift{D,typeof(shift)}(shift)
Drift(shift::SVector{D}) where {D} = Drift{D,typeof(shift)}(shift)

@inline place(d::Drift{D}, u::SVector{D}, t::Real) where {D} = u .+ at(d.shift, t)

# ---------------------------------------------------------------------------

"""
    Composed(outer, inner)
    outer ∘ inner

Apply `inner` first, then `outer`, both at the same time `t`. Layering is how effects stack without any one of them knowing about the others — e.g. wrapping a structure's motion in a global `Drift`.
"""
struct Composed{D,A,B} <: AbstractDeformation{D}
    outer::A
    inner::B
end

function Composed(outer::AbstractDeformation{D}, inner::AbstractDeformation{D}) where {D}
    return Composed{D,typeof(outer),typeof(inner)}(outer, inner)
end

@inline place(c::Composed{D}, u::SVector{D}, t::Real) where {D} =
    place(c.outer, place(c.inner, u, t), t)

function Base.:∘(outer::AbstractDeformation{D}, inner::AbstractDeformation{D}) where {D}
    return Composed(outer, inner)
end

# ---------------------------------------------------------------------------

"""
    Bend(kappa)

Bend the unit sphere into a spherical cap of the same area, `4π`, with dimensionless curvature `kappa` (κ below, stored in the field `kappa`) — a constant or a function of time, `-1 ≤ κ ≤ 1`. Apply it to material points from `SphereSurface{3}()`.

The cap's apex sits at the origin with outward normal `+z`. `κ = 0` is a flat disk of radius 2, `κ = 1` the whole sphere (centred at `(0, 0, -1)`), and a negative `κ` curves the other way. Each material point keeps the area above it, so a uniform density stays uniform as the cap bends, and a label keeps its place relative to the rim — see `derivations/derivation.qmd`.

Size, orientation and position are left to an outer [`Similarity`](@ref): with scale `s` the cap has area `4π s²` and physical curvature `κ / s`.

```julia
# area 4π·60² nm², flat at t = 0, about a hemisphere (κ = 1/√2) at t = 10
Similarity{3}(scale = 60.0, shift = SA[0.0, 0.0, 200.0]) ∘ Bend(t -> 0.07t)
```

Holding `scale` fixed and varying `κ` is the constant-area model; varying both with `κ(t) / s(t)` fixed is the constant-curvature model.
"""
struct Bend{K} <: AbstractDeformation{3}
    kappa::K

    # An inner constructor, so that no exact-match constructor can bypass the check.
    function Bend(κ)
        κ isa Real &&
            abs(κ) > 1 &&
            throw(ArgumentError("Bend needs -1 ≤ κ ≤ 1, got κ = $κ"))
        return new{typeof(κ)}(κ)
    end
end

@noinline _bend_domain(κ) = throw(
    DomainError(κ, "Bend needs -1 ≤ κ ≤ 1; at |κ| = 1 the cap is already the whole sphere"),
)

@inline function place(b::Bend, u::SVector{3,T}, t::Real) where {T}
    κ = at(b.kappa, t)
    abs(κ) <= 1 || _bend_domain(κ)
    h = clamp(1 - u[3], zero(T), 2 * one(T))    # depth below the north pole
    ρ′ = sqrt(h * (2 - κ^2 * h))
    ρ = sqrt(u[1]^2 + u[2]^2)
    # the south pole (ρ = 0, h = 2) maps to the whole rim; send it to φ = 0
    c, s = ρ > 0 ? (u[1] / ρ, u[2] / ρ) : (one(T), zero(T))
    return SVector(ρ′ * c, ρ′ * s, -κ * h)
end

# ---------------------------------------------------------------------------

"""
    rotation(θ)              -> 2×2 SMatrix
    rotation(axis, θ)        -> 3×3 SMatrix
    rotation(n)              -> 3×3 SMatrix

Rotation matrices for use as the `rot` field of a [`Similarity`](@ref). The 3D form is Rodrigues' formula about `axis` (need not be normalized).

`rotation(n)` turns `+z` onto the direction `n` (need not be normalized) by the shortest path, about the axis `ẑ × n`. Shapes and [`Bend`](@ref) face `+z`, so this orients a structure by its normal — a cap by the outward normal at its apex:

```julia
Similarity{3}(scale = 60.0, rot = rotation(n), shift = apex) ∘ Bend(κ)
```

This is the fitters' orientation convention: DynMoFit's `rotation_matrix(n)`, and LocMoFit's `rotmat(θ, ϕ)` with `n = (sin θ cos ϕ, sin θ sin ϕ, cos θ)`. For `n` along `-z` the axis is undefined and `+y` is used, as LocMoFit does at `ϕ = 0`.
"""
rotation(θ::Real) = @SMatrix [cos(θ) -sin(θ); sin(θ) cos(θ)]

function rotation(axis::AbstractVector, θ::Real)
    k = normalize(SVector{3,Float64}(axis))
    K = @SMatrix [0.0 -k[3] k[2]; k[3] 0.0 -k[1]; -k[2] k[1] 0.0]
    return SMatrix{3,3,Float64}(I) + sin(θ) * K + (1 - cos(θ)) * K * K
end

function rotation(n::AbstractVector)
    length(n) == 3 ||
        throw(DimensionMismatch("rotation(n) needs a 3-vector, got length $(length(n))"))
    all(iszero, n) && throw(ArgumentError("rotation(n) needs a non-zero direction"))
    m = normalize(SVector{3,Float64}(n))
    a = SVector(-m[2], m[1], 0.0)    # ẑ × m, of length sin θ
    s = hypot(m[1], m[2])
    # atan(s, cos θ) stays accurate near both poles, where acos(m[3]) would not
    s > 0 && return rotation(a, atan(s, m[3]))
    return m[3] > 0 ? SMatrix{3,3,Float64}(I) : rotation(SVector(0.0, 1.0, 0.0), π)
end
