# ---------------------------------------------------------------------------
# Localization precision models
# ---------------------------------------------------------------------------

"""
    AbstractUncertainty{D}

Maps a detected photon count to a per-axis localization standard deviation. Subtypes implement

    uncertainty(model, n) -> SVector{D}

Swapping the model is the supported way to make the noise more realistic without touching the rest of the pipeline.
"""
abstract type AbstractUncertainty{D} end

"""
    uncertainty(model, n) -> SVector{D}

Per-axis localization standard deviation for `n` detected photons.

Named `uncertainty` rather than `precision` because `Base.precision` already means something else entirely (the bits of significand in a float). The two words are used interchangeably for this quantity in the SMLM literature; the column on a [`Localization`](@ref) is still called `precision`.
"""
function uncertainty end

ndim(::AbstractUncertainty{D}) where {D} = D

"""
    psfwidths(D, sigma_xy; z_factor=2.5) -> SVector{D}

Per-axis Gaussian PSF widths: `sigma_xy` laterally and `z_factor * sigma_xy` axially. The axial width of a real astigmatic or biplane PSF varies strongly with z, so a constant factor is an approximation that degrades away from focus.
"""
function psfwidths(D::Integer, σ_xy::Real; z_factor=2.5)
    if D == 2
        return SVector{2,Float64}(σ_xy, σ_xy)
    elseif D == 3
        return SVector{3,Float64}(σ_xy, σ_xy, z_factor * σ_xy)
    else
        throw(
            ArgumentError(
                "psfwidths needs D of 2 or 3, got $D; pass a width vector instead"
            ),
        )
    end
end

"""
    ShotNoise(sigma)
    ShotNoise{D}(sigma_xy; z_factor=2.5)

The shot-noise-limited Cramér–Rao bound for a Gaussian PSF — no background, no pixelation, no readout noise. `sigma` holds the per-axis PSF widths s (stored in the field `sigma`; `sigma_xy` gives them through [`psfwidths`](@ref)):

    σ_loc = s / sqrt(n)

Because `n` is geometric rather than fixed, the resulting spread of `σ_loc` is heavy-tailed: rare one- and two-photon events give enormous uncertainties. The continuous approximation is Fréchet with shape 2, whose variance diverges, so summarise simulated uncertainties with medians rather than means.
"""
struct ShotNoise{D,T} <: AbstractUncertainty{D}
    sigma::SVector{D,T}
end

ShotNoise(σ::AbstractVector) = ShotNoise(SVector{length(σ),Float64}(σ))
ShotNoise{D}(σ_xy::Real; kwargs...) where {D} = ShotNoise(psfwidths(D, σ_xy; kwargs...))

@inline uncertainty(m::ShotNoise, n::Integer) = m.sigma ./ sqrt(n)

"""
    StallingaRieger(sigma; pixel, background)
    StallingaRieger{D}(sigma_xy; pixel, background, z_factor=2.5)

Stallinga & Rieger's closed-form approximation to the MLE variance with background and pixelation,

    σ_loc² = (sₐ²/n) (1 + 4v + sqrt(2v / (1 + 4v))),
    sₐ² = s² + a²/12,    v = 2π sₐ² b / (n a²)

with PSF widths s (the field `sigma`), pixel size `a` and `b` background photons per pixel. Reduces to [`ShotNoise`](@ref) as `b → 0` and `a → 0`.

Background is an image-plane quantity, so applying it per-axis in 3D is a convenient extrapolation rather than a derived result.
"""
struct StallingaRieger{D,T} <: AbstractUncertainty{D}
    sigma::SVector{D,T}
    pixel::T
    background::T
end

function StallingaRieger(σ::AbstractVector; pixel, background)
    return StallingaRieger(SVector{length(σ),Float64}(σ), float(pixel), float(background))
end
function StallingaRieger{D}(σ_xy::Real; pixel, background, kwargs...) where {D}
    return StallingaRieger(psfwidths(D, σ_xy; kwargs...); pixel, background)
end

@inline function uncertainty(m::StallingaRieger, n::Integer)
    sa² = m.sigma .^ 2 .+ m.pixel^2 / 12
    v = 2π .* sa² .* m.background ./ (n * m.pixel^2)
    return sqrt.(sa² ./ n .* (1 .+ 4 .* v .+ sqrt.(2 .* v ./ (1 .+ 4 .* v))))
end

"""
    GaussianMask(sigma; pixel, background)
    GaussianMask{D}(sigma_xy; pixel, background, z_factor=2.5)

Mortensen et al.'s variance for a Gaussian mask (unweighted least-squares) estimator, with PSF widths s in the field `sigma` and `sₐ` and `v` as in [`StallingaRieger`](@ref),

    σ_loc² = (sₐ²/n) (16/9 + 4v)

Note this does *not* reduce to the Cramér–Rao bound as `b → 0` — the estimator is not efficient, leaving the residual factor of 16/9.
"""
struct GaussianMask{D,T} <: AbstractUncertainty{D}
    sigma::SVector{D,T}
    pixel::T
    background::T
end

function GaussianMask(σ::AbstractVector; pixel, background)
    return GaussianMask(SVector{length(σ),Float64}(σ), float(pixel), float(background))
end
function GaussianMask{D}(σ_xy::Real; pixel, background, kwargs...) where {D}
    return GaussianMask(psfwidths(D, σ_xy; kwargs...); pixel, background)
end

@inline function uncertainty(m::GaussianMask, n::Integer)
    sa² = m.sigma .^ 2 .+ m.pixel^2 / 12
    v = 2π .* sa² .* m.background ./ (n * m.pixel^2)
    return sqrt.(sa² ./ n .* (16 / 9 .+ 4 .* v))
end

# ---------------------------------------------------------------------------
# Photon budget
# ---------------------------------------------------------------------------

"""
    Optics(; uncertainty, brightness, ontime)

Everything between an emission event and a localization.

- `uncertainty` — an [`AbstractUncertainty`](@ref) model
- `brightness`  — `h`, detected photons per unit time while ON
- `ontime`      — `τ`, mean ON duration (exponential)

With emission a Poisson process of rate `h` running for an `Exponential(τ)` ON period, the photon count marginalises exactly to a geometric distribution,

    N ~ Geometric(p),   p = 1 / (1 + h τ),   E[N] = h τ

so the ON duration never has to be sampled — one draw per event, not two. The race interpretation makes this transparent: emission and switch-off compete at rates `h` and `1/τ`, and `N` counts the emissions winning before the switch-off.

That marginal puts mass `p` on `N = 0`. An event that emits no detected photons leaves nothing to fit, so `simulate` drops it — not as a tunable threshold, but because there is no localization to report. `brightness` and `ontime` therefore describe *emission events*; see [`photondist`](@ref) for what survives.
"""
struct Optics{D,T,P<:AbstractUncertainty}
    uncertainty::P
    brightness::T
    ontime::T
end

function Optics(; uncertainty::AbstractUncertainty{D}, brightness, ontime) where {D}
    brightness > 0 || throw(ArgumentError("brightness must be positive, got $brightness"))
    ontime > 0 || throw(ArgumentError("ontime must be positive, got $ontime"))
    h, τ = promote(float(brightness), float(ontime))
    return Optics{D,typeof(h),typeof(uncertainty)}(uncertainty, h, τ)
end

ndim(::Optics{D}) where {D} = D

"""
    meanphotons(o) -> h * τ

Mean photons over *all* emission events, including the ones that yield nothing and are dropped. The mean of the counts actually reported is one higher — see [`photondist`](@ref).
"""
meanphotons(o::Optics) = o.brightness * o.ontime

"""
    photondist(o) -> Geometric

The photon count of an emission event: the marginal from the derivation, `Geometric(1 / (1 + h τ))`, with support `{0, 1, 2, ...}` and mean `h τ`. It depends on `brightness` and `ontime` only through their product.

`simulate` draws from this directly and then discards the events that came up zero, which is a fraction `p = 1/(1 + h τ)` of them. The counts that survive therefore follow the law conditioned on `N ≥ 1`,

    P(N = n | N ≥ 1) = (1-p)^(n-1) p,    n = 1, 2, ...

whose mean is `1 + h τ`. So `mean(locs.nphotons)` sits one above [`meanphotons`](@ref), and `length(locs)` sits a fraction `p` below the number of events.
"""
photondist(o::Optics) = Geometric(1 / (1 + meanphotons(o)))
