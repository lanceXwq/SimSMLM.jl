"""
    AbstractStructure{D}

What [`simulate`](@ref) draws signal events from: a single [`Structure`](@ref),
or a [`Composite`](@ref) of several.
"""
abstract type AbstractStructure{D} end

ndim(::AbstractStructure{D}) where {D} = D

"""
    Structure(shape, motion)

A reference configuration paired with the deformation that carries it through time. `motion` may be any [`AbstractDeformation`](@ref); use [`Static`](@ref) for something that does not move.
"""
struct Structure{D,S,M} <: AbstractStructure{D}
    shape::S
    motion::M
end

function Structure(shape::AbstractShape{D}, motion::AbstractDeformation{D}) where {D}
    Structure{D,typeof(shape),typeof(motion)}(shape, motion)
end

Structure(shape::AbstractShape{D}) where {D} = Structure(shape, Static{D}())

"""
    place(s::Structure, u, t)

World position at time `t` of the material coordinate `u`.
"""
@inline place(s::Structure{D}, u::SVector{D}, t::Real) where {D} = place(s.motion, u, t)

"""
    Composite(structure => weight, ...)

Several structures imaged together. Each signal event comes from component `k`
with probability `weight[k] / sum(weight)`, so a composite is the same as its
components simulated at those fractions of `emis_rate` and merged.

Every component keeps its own shape and deformation, so parts can move
independently — a cap that bends beside a ring that does not. Motion shared by
the whole composite, such as drift, is composed into each component with `∘`.

```julia
# a ring with a bending cap 50 nm above it
scr = Composite(
    Structure(Ring{3}(), Similarity{3}(scale = 50.0)) => 0.4,
    Structure(SphereSurface{3}(),
              Similarity{3}(scale = 40.0, shift = SA[0.0, 0.0, 50.0]) ∘ Bend(t -> 0.1t)) => 0.6,
)
```

Weights are fractions of *events*, not label densities: for an equal density
across components, make them proportional to each component's length or area.

In the [`Truth`](@ref), `component` says which component each localization came
from, and `motion[k]` is component `k`'s deformation.
"""
struct Composite{D,C<:Tuple{Vararg{Structure{D}}},K} <: AbstractStructure{D}
    components::C
    cumweights::NTuple{K,Float64}   # normalised, so the last is 1

    # An inner constructor, so the weights cannot reach the struct unchecked.
    function Composite(pairs::Pair{<:Structure,<:Real}...)
        isempty(pairs) && throw(ArgumentError("a Composite needs at least one component"))
        comps = map(first, pairs)
        w = map(p -> Float64(last(p)), pairs)
        all(x -> isfinite(x) && x >= 0, w) ||
            throw(ArgumentError("weights must be finite and non-negative, got $w"))
        total = sum(w)
        total > 0 || throw(ArgumentError("weights must not all be zero"))
        D = ndim(first(comps))
        all(c -> ndim(c) == D, comps) ||
            throw(DimensionMismatch("all components must have the same dimension"))
        return new{D,typeof(comps),length(comps)}(comps, accumulate(+, w) ./ total)
    end
end

# Noise-free position, material coordinate and component index of one signal event.
@inline function _sample_event(rng::AbstractRNG, s::Structure, t::Real)
    u = sample_site(rng, s.shape)
    return place(s.motion, u, t), u, Int32(1)
end

@inline _sample_event(rng::AbstractRNG, c::Composite, t::Real) =
    _pick(rng, c.components, c.cumweights, rand(rng), t, Int32(1))

# Walk the components to the one `v` falls in. Recursing on the tail keeps every
# branch concretely typed, so components of different types stay inferable.
@inline function _pick(rng::AbstractRNG, comps::Tuple, cw::Tuple, v::Real, t::Real, k::Int32)
    if length(comps) == 1 || v < first(cw)
        x, u, _ = _sample_event(rng, first(comps), t)
        return x, u, k
    end
    return _pick(rng, Base.tail(comps), Base.tail(cw), v, t, k + Int32(1))
end

# The deformation(s) to record in the Truth.
_motion(s::Structure) = s.motion
_motion(c::Composite) = map(x -> x.motion, c.components)

"""
    sample_position([rng], s::AbstractStructure, t) -> SVector

Draw one position from the structure as it stands at time `t`: a material
coordinate from the reference shape, pushed forward by the deformation. For a
[`Composite`](@ref), the component is drawn first.
"""
@inline sample_position(rng::AbstractRNG, s::AbstractStructure, t::Real) =
    first(_sample_event(rng, s, t))

sample_position(s::AbstractStructure, t::Real) = sample_position(default_rng(), s, t)

# ---------------------------------------------------------------------------

# Stage 2: signal or background, for every event. With no background there is
# nothing to decide.
_classify(::AbstractRNG, ::Real, ::Nothing, n::Integer) = fill(true, n)

function _classify(rng::AbstractRNG, λs::Real, bg::Background, n::Integer)
    p = _psignal(λs, bg.rate)
    u = rand(rng, n)                      # one batched draw
    issignal = Vector{Bool}(undef, n)     # not a BitVector: Truth holds Vector{Bool}
    issignal .= u .< p
    return issignal
end

# Stage 3: photon counts. Signal and background each have their own
# distribution, so the draw depends on the type decided in stage 2. Both are
# passed as models, not samplers, so the two branches stay symmetric.
_counts(rng::AbstractRNG, ::Vector{Bool}, optics::Optics, ::Nothing, n::Integer) =
    rand(rng, Distributions.sampler(photondist(optics)), n)

function _counts(rng::AbstractRNG, issignal::Vector{Bool}, optics::Optics,
    bg::Background, n::Integer)
    sig = Distributions.sampler(photondist(optics))
    bgs = Distributions.sampler(photondist(bg))
    out = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        out[i] = issignal[i] ? rand(rng, sig) : rand(rng, bgs)
    end
    return out
end

# Noise-free position, material coordinate and component of one event.
# Dispatching on the background keeps the spurious branch from existing at all
# when there is none.
@inline _place_event(rng::AbstractRNG, s::AbstractStructure, ::Nothing, ::Bool,
    t::Real, ::Type{T}) where {T} = _sample_event(rng, s, t)

@inline function _place_event(rng::AbstractRNG, s::AbstractStructure{D}, bg::Background{D},
    issig::Bool, t::Real, ::Type{T}) where {D,T}
    issig && return _sample_event(rng, s, t)
    nan = SVector{D,T}(ntuple(_ -> T(NaN), Val(D)))
    return randin(rng, bg.fov), nan, Int32(0)
end

"""
    simulate([rng], structure; emis_rate, optics, t_end | nevents, background=nothing)

Simulate an SMLM acquisition of a moving structure — a [`Structure`](@ref), or
a [`Composite`](@ref) of several.

The pipeline is four batched stages, in order:

1. all event times, from a Poisson process of rate `emis_rate`;
2. all event types — signal or background — from one Bernoulli draw;
3. all photon counts, each from the distribution its type calls for;
4. events that emitted no photons are dropped, and the survivors are localized.

Stage 4 is not a tunable threshold: an event that yields no detected photons
produces nothing to fit, so there is no localization to report. `emis_rate`
therefore counts *emission events*, and a fraction `1/(1+hτ)` of them leave no
trace. Among the rows that survive the photon count follows the geometric law
conditioned on `N ≥ 1`, with mean `1 + hτ`.

Pass exactly one of:

- `t_end` — fixed duration `[0, t_end]`, random number of events
- `nevents` — fixed number of events, random duration

`nevents` counts events, not localizations; expect about `n·hτ/(1+hτ)` rows.

Pass a [`Background`](@ref) to add spurious localizations. They come from the
same superposed Poisson process, are classified in stage 2, draw from their own
photon distribution in stage 3, and are flagged in `truth.issignal`.

Localizations carry the exact event time rather than a frame index. Frames are
recoverable afterwards if you ever want them — `floor.(locs.t ./ period) .+ 1`.

Returns the localization table and the [`Truth`](@ref) aligned with it.

# Example
```julia
using SimSMLM, StaticArrays

# a sphere growing from 100 nm at 20 nm/s, drifting along x at 10 nm/s
sphere = Structure(
    SphereSurface{3}(),
    Similarity{3}(scale = t -> 100 + 20t, shift = t -> SA[10t, 0.0, 0.0]),
)

locs, truth = simulate(sphere;
    emis_rate = 200.0,
    t_end     = 10.0,
    optics    = Optics(uncertainty = ShotNoise{3}(120.0),
                       brightness = 5000.0, ontime = 0.01),
    background = Background(rate = 20.0, meanphotons = 200.0,
                            fov = FieldOfView(SA[-500.0, -500, -500],
                                              SA[ 500.0,  500,  500])),
)

nbackground(truth) / length(truth)   # spurious fraction
```
"""
function simulate(rng::AbstractRNG, s::AbstractStructure{D}; emis_rate::Real,
    optics::Optics{D,T}, t_end::Union{Nothing,Real}=nothing,
    nevents::Union{Nothing,Integer}=nothing,
    background::Union{Nothing,Background{D}}=nothing) where {D,T}

    (t_end === nothing) == (nevents === nothing) && throw(ArgumentError(
        "pass exactly one of `t_end` or `nevents`"))

    total = background === nothing ? emis_rate : emis_rate + background.rate

    # 1. every event time
    times = t_end === nothing ? eventtimes(rng, total, nevents) :
            eventtimes(rng, total, t_end)
    nev = length(times)

    # 2. every event type, then 3. every photon count
    issignal = _classify(rng, emis_rate, background, nev)
    counts = _counts(rng, issignal, optics, background, nev)

    # 4. an event that emitted nothing leaves no localization
    keep = findall(!iszero, counts)
    n = length(keep)

    pos = Matrix{T}(undef, D, n)
    prec = Matrix{T}(undef, D, n)
    ts = Vector{T}(undef, n)
    nphot = Vector{Int32}(undef, n)
    truepos = Matrix{T}(undef, D, n)
    us = Matrix{T}(undef, D, n)
    comp = Vector{Int32}(undef, n)

    # positions are drawn only for survivors — a dropped event never needed one
    @inbounds for (i, k) in enumerate(keep)
        t = times[k]
        N = counts[k]
        x, u, c = _place_event(rng, s, background, issignal[k], t, T)
        σ = uncertainty(optics.uncertainty, N)
        v = x .+ σ .* randn(rng, SVector{D,T})

        for d in 1:D
            pos[d, i] = v[d]
            prec[d, i] = σ[d]
            truepos[d, i] = x[d]
            us[d, i] = u[d]
        end
        ts[i] = T(t)
        nphot[i] = Int32(N)
        comp[i] = c
    end

    locs = localizationtable(pos, prec, ts, nphot, Val(D))
    truth = Truth{D,T}(truepos, us, issignal[keep], comp, _motion(s))
    return locs, truth
end

simulate(s::AbstractStructure; kwargs...) = simulate(default_rng(), s; kwargs...)

"""
    simulate([rng], shape, motion; kwargs...)

Convenience form that builds the [`Structure`](@ref) for you.
"""
simulate(rng::AbstractRNG, shape::AbstractShape, motion::AbstractDeformation; kwargs...) =
    simulate(rng, Structure(shape, motion); kwargs...)

simulate(shape::AbstractShape, motion::AbstractDeformation; kwargs...) =
    simulate(default_rng(), Structure(shape, motion); kwargs...)
