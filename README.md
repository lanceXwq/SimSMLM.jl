# SimSMLM

A Julia repository for simulating some SMLM data. Not for production, just to produce proof-of-concept tests for LocMoFit and DynMoFit.

What makes it different from other SMLM simulators is that the structure is allowed to **move** — a sphere whose centre translates and whose radius grows, say.

[![Build Status](https://github.com/lancexwq/SimSMLM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/lancexwq/SimSMLM.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Example

```julia
using SimSMLM, StaticArrays

# a sphere growing from 100 nm at 20 nm/s, drifting along x at 10 nm/s
sphere = Structure(
    SphereSurface{3}(),
    Similarity{3}(scale = t -> 100 + 20t, shift = t -> SA[10t, 0.0, 0.0]),
)

locs, truth = simulate(sphere;
    emis_rate = 2000.0,                                   # emission events per second
    t_end     = 10.0,                                     # acquisition length, s
    optics    = Optics(uncertainty = ShotNoise{3}(120.0), # PSF width, nm
                       brightness  = 1e5,                 # h, detected photons/s
                       ontime      = 0.01),               # τ, mean ON duration
    background = Background(rate = 200.0,                 # spurious events/s
                            meanphotons = 40.0,           # dimmer than the signal
        fov = FieldOfView(SA[-400.0,-400,-400], SA[400.0,400,400])),
)

locs[i]                # a Localization, static-vector fields, built on the stack
locs.position[i]       # that point as an SVector
positions(locs)        # the underlying contiguous 3×N matrix — same bytes
locs.nphotons          # photons behind each localization

truth.position[i]      # noise-free position, aligned element-for-element
truth.issignal         # false for a spurious localization
truth.component        # which part of a Composite it came from (0 for spurious)
nbackground(truth)     # how many spurious ones there were
errors(locs, truth) ./ precisions(locs)   # standardised residuals ~ N(0,1)

smaptable(locs)        # flat columns with SMAPpy's names, e.g. for CSV.write
smaptable(truth)       # the ground truth as its own table, row-aligned
```

Pass `nevents = 20_000` in place of `t_end` to fix the event count and let the duration float.

## Caps and composite structures

A spherical cap is the unit sphere bent by `Bend(kappa)`, and `Composite` images several structures together, each with its own motion and share of the events:

```julia
# a ring, and a cap 50 nm above it that curves from flat while the ring stays put
scr = Composite(
    Structure(Ring{3}(), Similarity{3}(scale = 50.0)) => 0.4,
    Structure(SphereSurface{3}(),
              Similarity{3}(scale = 40.0, shift = SA[0.0, 0, 50]) ∘ Bend(t -> 0.1t)) => 0.6,
)
locs, truth = simulate(scr; emis_rate = 2000.0, t_end = 5.0,
    optics = Optics(uncertainty = ShotNoise{3}(120.0), brightness = 1e5, ontime = 0.01))

truth.component        # which part each localization came from (0 for spurious)
```

## Point clouds

`PointCloud` turns a set of discrete labels into a shape. Each event draws one label, with replacement, so every label is localized repeatedly — the clustering of real blinking dyes, which a continuous shape never shows. The labels can be your own points (a fitter's model points, say) or drawn once from a continuous shape, and the deformation moves them like any other shape:

```julia
# eight labels on a 50 nm ring, NPC-like
φ = 2π .* (0:7) ./ 8
npc = Structure(PointCloud([cos.(φ) sin.(φ) zeros(8)]'), Similarity{3}(scale = 50.0))

# 200 fixed labels scattered over a lattice that curves from flat
labels = PointCloud(sample_sites(SphereSurface{3}(), 200))
cap = Structure(labels, Similarity{3}(scale = 40.0) ∘ Bend(t -> 0.1t))

locs, truth = simulate(cap; emis_rate = 2000.0, t_end = 5.0,
    optics = Optics(uncertainty = ShotNoise{3}(120.0), brightness = 1e5, ontime = 0.01))

truth.u                # the label behind each localization
```

For partial labeling, keep a random subset of the points: `PointCloud(pts[:, rand(size(pts, 2)) .< 0.6])`. Point clouds and continuous shapes mix freely in a `Composite`.

## Building blocks

- **Shapes**: `SphereSurface`, `Ball`, `Ring`, `LineSegment`, `PointCloud` (wrap your own `D×N` matrix). Orientation is the deformation's job, so shapes sit in the xy-plane and `Similarity`'s `rot` turns them.
- **Deformations**: `Static`, `Similarity` (scale/rotate/translate, each constant or `t -> ...`), `Drift`, `Bend` (unit sphere → constant-area spherical cap of dimensionless curvature `kappa`), `Composed` (via `∘`), plus `rotation` helpers — `rotation(n)` points a structure's `+z` along a normal `n`
- **Structures**: `Structure` (a shape and its deformation), `Composite` (several structures, each with its own deformation and share of the events)
- **Events**: `eventtimes` at a constant rate, over either a `t_end` or a fixed count
- **Background**: `Background` + `FieldOfView` for spurious localizations, with their own photon budget
- **Uncertainty**: `ShotNoise` (Cramér–Rao, `s/√n`), `StallingaRieger` and `GaussianMask` (both with background and pixelation)
- **Export**: `smaptable` — flat columns with SMAPpy's names (`x_nm`, `xy_err_nm`, `photons`, `time_s`, …) that `CSV.write` or `DataFrame` take directly; the truth exports as its own table

Any model field accepts either a constant or a function of time, resolved by `at(p, t)`. Structs are parametric on those field types so the call inlines; the whole pipeline is type-stable and runs ~10⁷ localizations/s.

## Further reading

- `NOTE.md` — the model and its assumptions, spherical caps, composite structures, storage, and implementation notes.
- `derivations/derivation.qmd` — the maths: photon counts, precision models, the constant-area cap map, Poisson splitting.
