# AGENTS.md

Guidance for AI coding agents working in this repository. Workspace-wide rules (data layout, units, objectives, compatibility, working with the owner) are in `../AGENTS.md`; read that first.

## What this is

**SimSMLM.jl** simulates single-molecule localization microscopy (SMLM) datasets from **moving** structures. For example, a sphere whose centre drifts and whose radius grows, or a spherical cap with a drifting center and a changing curvature. It exists to produce test data for the fitting packages LocMoFit and DynMoFit, so it must be **accurate enough to test them**: its output must follow the stated model closely enough that a fit failing on SimSMLM data points to the fitter, not the simulator. It is not production software, and the owner values a small codebase over features: do not add capabilities that were not asked for.

## Commands

```sh
julia --project=. -e 'using Pkg; Pkg.test()'        # full suite, ~5 s, must pass
julia --project=. -e 'using SimSMLM'                 # quick load check
```

CI and `[compat]` target Julia 1.13 only (see Compatibility in `../AGENTS.md`); no need to keep compatibility with older versions.

## Layout

| Path | Contents |
|---|---|
| `src/SimSMLM.jl` | Module: `using` lines, all exports, `include` order |
| `src/timevarying.jl` | `at(p, t)` — any parameter may be a constant or a function of time |
| `src/types.jl` | `Localization`, `LocalizationTable`, `Truth`, `columnview`, accessors |
| `src/deformation.jl` | `place(d, u, t)`: `Static`, `Similarity`, `Drift`, `Composed` (`∘`), `rotation` |
| `src/shapes.jl` | `sample_site(rng, shape)`: `SphereSurface`, `Ball`, `Ring`, `LineSegment`, `PointCloud` |
| `src/events.jl` | `eventtimes(rng, rate, t_end)` and `eventtimes(rng, rate, n)` |
| `src/background.jl` | `FieldOfView`, `Background` (spurious localizations with their own photon budget) |
| `src/optics.jl` | Uncertainty models (`ShotNoise`, `StallingaRieger`, `GaussianMask`), `Optics`, `photondist` |
| `src/simulate.jl` | `Structure` and the `simulate` pipeline |
| `test/runtests.jl` | Single file, one `@testset` per concern |
| `derivations/derivation.qmd` | The owner's derivations of the photon and precision models (gitignored).  All math should be checked there first. If new math is added, notes must be created there.  |
| `papers/` | Reference PDFs (gitignored) |
| `simulation.jl` (root) | The owner's **old** code, kept for reference only. Not part of the package; do not include, test, or "fix" it. |

## The model (what `simulate` does)

Four batched stages, in this order — keep it that way:

1. **Times** — all event times from a Poisson process at `emis_rate` (+ `background.rate` if present). The clock starts at the first event, so `times[1] == 0.0`.
2. **Types** — one batched Bernoulli draw labels each event signal or background, with `P(signal) = λs / (λs + λb)` (Poisson superposition).
3. **Photons** — each event's count from its type's distribution: `photondist(optics)` or `photondist(background)`, both `Geometric(1/(1+mean))`.
4. **Filter & localize** — events with **zero** photons are dropped (not a tunable threshold — there is no `min_photons`). Survivors get a fresh position sampled from the structure *at that instant* (`sample_site` → `place`), an uncertainty `σ = uncertainty(model, N)`, and Gaussian noise.

Consequences worth knowing before touching the maths:

- `emis_rate` counts *emission events*; a fraction `1/(1+hτ)` leave no localization. `nevents` counts events, not rows.
- Surviving photon counts are geometric conditioned on `N ≥ 1`, so `mean(locs.nphotons) ≈ 1 + meanphotons(o)`, not `meanphotons(o)`.
- `σ = s/√N` is heavy-tailed (Fréchet, shape 2, infinite variance). Assert on **medians**, not means.
- Every event is an independent draw: there is no emitter identity and no repeated blinking. Rates are constant (no time-varying `emis_rate`).
- `Background` shares the PSF/uncertainty model, so a dimmer background is automatically worse-localized.

The physical derivations live in `derivations/derivation.qmd`; if you change the photon or precision model, check it against that file.

## Data layout (important)

- Each `Localization` is a struct with fields including a localization's position, precision, and time tag.
- The position and precision shall each be a `SVector{D,T}` with `D ∈ {2,3}` as a **type parameter**.
- When creating a vector of localizations, the vector should be created as a `StructArray`, such that all positions are in one contiguous memory block. The same for precisions, and time tag.

## Conventions

- **Type stability is a requirement.** Structs are parametric on field types (including function-valued fields) so `at(f, t)` inlines. When a type parameter like `D` is known, construct with it explicitly (`Truth{D,T}(...)`, `Val(D)`) — deriving it from `size(m, 1)` erases it from the inferred type. Check new code paths with `@inferred simulate(...)`.
- **Performance comes before simplicity** (workspace objectives: accuracy, then performance, then simplicity). The hot loop is allocation-free; ~10⁷ events in well under a second — keep it that way, and benchmark before and after touching it. Preallocate when sizes are known; build `Distributions.sampler(...)` once, outside loops.
- Every sampling function takes `rng::AbstractRNG` first, with a `default_rng()` convenience method.
- Validate constructor arguments with `ArgumentError`/`DimensionMismatch`. When a struct has `SVector` fields and needs validation, use an **inner** constructor — Julia's auto-generated exact-match constructor otherwise bypasses an outer validating one.
- `Base.precision` exists, so the uncertainty function is named `uncertainty`; the `Localization` field is still `precision`. Avoid exporting names that clash with `Base`.
- Docstrings on every exported name, explaining *why*, not just *what*. Match the existing prose style and comment density.
- Dependencies: Distributions, LinearAlgebra, Random, StaticArrays, StructArrays. Ask before adding more.

## Testing

- Use `StableRNG(seed)` (via the `rng()` helper), never a seeded default RNG — Julia's default stream is not stable across versions.
- Tests are **statistical**: assert against the closed-form expectation with a tolerance a few standard errors wide (`rtol`/`atol`). Derive the expected value; don't tune thresholds to whatever the current output happens to be.
- A failing statistical test may be a wrong expectation rather than a bug — work out the theory before changing either.
- For every public behaviour change, update the tests **and** the README, and grep for stale references (renamed keywords, removed types) across `src/`, `test/` and `README.md`.

## Working with the owner

The general rules are in `../AGENTS.md`. Specific to this package:

- Recent history is a steady *removal* of features (frame grids, labelling, blinking kinetics, time-varying rates, `min_photons`) — don't reintroduce them.
- Docstrings can lag behind the code after the owner's edits; when they disagree, the code is usually the intent — ask if unsure, and fix the docstring.
