"""
    AbstractShape{D}

A reference configuration to draw label sites from. Subtypes implement

    sample_site(rng, shape) -> SVector{D,T}

One site is drawn per emission event, so `sample_site` is the primitive and `sample_sites` is a convenience wrapper over it. Shapes are conventionally defined at unit size so that a deformation's `scale` sets the physical dimensions.

Sampling a fresh site per event makes labels a continuous density over the structure rather than a fixed set of discrete points.
"""
abstract type AbstractShape{D} end

ndim(::AbstractShape{D}) where {D} = D

"""
    sample_site([rng], shape) -> SVector

Draw one material coordinate from `shape`. This is the primitive each shape implements; it allocates nothing.
"""
function sample_site end

sample_site(shape::AbstractShape) = sample_site(default_rng(), shape)

"""
    sample_sites([rng], shape, n) -> Vector{SVector}

Draw `n` material coordinates from `shape`.
"""
function sample_sites(rng::AbstractRNG, shape::AbstractShape, n::Integer)
    return [sample_site(rng, shape) for _ in 1:n]
end

sample_sites(shape::AbstractShape, n::Integer) = sample_sites(default_rng(), shape, n)

# ---------------------------------------------------------------------------

"""
    SphereSurface{D}()

Uniform points on the unit sphere (`D=3`) or unit circle (`D=2`).
"""
struct SphereSurface{D} <: AbstractShape{D} end

SphereSurface(D::Integer) = SphereSurface{Int(D)}()

@inline sample_site(rng::AbstractRNG, ::SphereSurface{D}) where {D} =
    normalize(randn(rng, SVector{D,Float64}))

"""
    Ball{D}()

Uniform points inside the unit ball.
"""
struct Ball{D} <: AbstractShape{D} end

Ball(D::Integer) = Ball{Int(D)}()

@inline sample_site(rng::AbstractRNG, ::Ball{D}) where {D} =
    normalize(randn(rng, SVector{D,Float64})) * rand(rng)^(1 / D)

"""
    Ring{D}()

Uniform points on the unit circle lying in the xy-plane. For `D=3` the third coordinate is zero, which makes it the reference configuration for a ring seen in a 3D dataset.

In the plane `Ring{2}()` and `SphereSurface{2}()` are the same distribution — the unit circle *is* the unit sphere there. They differ from `D = 3` up, where a ring is a curve and a sphere surface is a surface.
"""
struct Ring{D} <: AbstractShape{D} end

Ring(D::Integer) = Ring{Int(D)}()

@inline function sample_site(rng::AbstractRNG, ::Ring{D}) where {D}
    φ = 2π * rand(rng)
    xyz = (cos(φ), sin(φ), 0.0)
    return SVector{D,Float64}(ntuple(i -> xyz[i], Val(D)))
end

"""
    LineSegment(a, b)

Uniform points on the segment from `a` to `b`.
"""
struct LineSegment{D,T} <: AbstractShape{D}
    a::SVector{D,T}
    b::SVector{D,T}
end

function LineSegment(a::AbstractVector, b::AbstractVector)
    return LineSegment(SVector{length(a),Float64}(a), SVector{length(b),Float64}(b))
end

@inline sample_site(rng::AbstractRNG, s::LineSegment{D}) where {D} =
    s.a .+ rand(rng) .* (s.b .- s.a)

"""
    PointCloud(pts)

An explicit reference configuration: wrap a `D × N` matrix or a vector of static vectors. Sampling draws one of the points uniformly at random, so a point cloud behaves like any other shape — use it when you have model points rather than a parametric form.

With a fresh draw per event, the same point can be selected repeatedly, which is the discrete analogue of a label emitting more than once.
"""
struct PointCloud{D,T} <: AbstractShape{D}
    pts::Vector{SVector{D,T}}
end

PointCloud(pts::AbstractMatrix{<:Real}) = PointCloud(tosites(pts))

@inline sample_site(rng::AbstractRNG, s::PointCloud) =
    @inbounds s.pts[rand(rng, eachindex(s.pts))]

# ---------------------------------------------------------------------------

"""
    tosites(pts) -> Vector{SVector{D,T}}

Convert a `D × N` matrix of points into static vectors. Accepts a vector of static vectors unchanged, so callers can take either form.
"""
function tosites(pts::AbstractMatrix{T}) where {T<:Real}
    D = size(pts, 1)
    return [SVector{D,float(T)}(view(pts, :, j)) for j in axes(pts, 2)]
end

tosites(pts::AbstractVector{<:SVector}) = collect(pts)
