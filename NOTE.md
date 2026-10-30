# Notes

The details behind the README: the model SimSMLM simulates, what it deliberately leaves out, and how it is implemented. The maths is in `derivations/derivation.qmd`, which holds maths only; anything about the code belongs here or in comments next to it.

## Model

Every emission event is independent, and yields at most one localization:

1. **When** — event times are a Poisson process of constant `emis_rate`, so the waiting time between events is exponential. `eventtimes`. The clock starts at the first event, so `locs.t` begins at exactly zero.
2. **Where** — a material coordinate is drawn from the reference shape and pushed forward by the deformation to give the position *at that instant*. `sample_site` + `place`. For a `Composite`, the component is drawn first.
3. **How bright** — the photon count is geometric, from whichever distribution the event's type calls for (see below). `photondist`.
4. **How well localized** — an `AbstractUncertainty` model turns the photon count into a per-axis standard deviation, and the reported position is the truth plus that noise, and plus the blur for a signal event (see below).

The four stages are batched in that order: all event times, then all event types, then all photon counts, then the events that emitted nothing are dropped and the survivors localized.

`nevents` fixes the event count instead of the duration — useful when comparing fits at a matched number of events. It counts *events*, so expect a fraction `1/(1+hτ)` fewer rows.

### Time

Localizations carry the **exact event time**, not a frame index. Since each event yields at most one localization, `t` subsumes anything a frame number would tell you, and frames are recoverable afterwards with `floor.(locs.t ./ period) .+ 1`. If a fitter should only ever see binned time, quantize `t` to the frame centre — that is the one place frames would change the data rather than just label it.

### Background

Passing a `Background` adds spurious localizations — the false positives a real pipeline produces from noise, autofluorescence and out-of-focus emitters — drawn uniformly in a `FieldOfView` and flagged in `truth.issignal`.

Because independent Poisson processes superpose, one stream is drawn at `λ_signal + rate` and each event is labelled signal with probability `λ_signal / (λ_signal + rate)`. That is exact and costs one exponential draw per event instead of two processes and a merge.

Background carries its own `meanphotons`, separate from the signal's. Since both share the PSF and uncertainty model, a dimmer background is also a *less precisely localized* one — which is what real false positives look like. Pass `meanphotons(optics)` to make them indistinguishable in brightness.

### Blur

`simulate(...; blur)` displaces each signal event's emitter from its structure point by an isotropic Gaussian of standard deviation `blur` (nm, default 0): the size of the label, and any spread of the same kind that the precision does not account for. It is LocMoFit's fitted `sigma`, and the law is `@eq-blur` in the derivations: each signal localization scatters about its structure point with per-axis variance `precision² + blur²`.

- **What is reported.** The precision column stays the localization's own `σ`, since a real pipeline reports the uncertainty of the position estimate and knows nothing of the label. `truth.position` stays the structure point, so `place(truth.motion, truth.u[i], locs.t[i]) == truth.position[i]` still holds, and `errors(locs, truth)` divided by `sqrt.(precisions(locs) .^ 2 .+ blur^2)` is standard normal on the signal rows. The displaced emitter positions are not stored.
- **Background** rows are not blurred: they have no structure point to be displaced from.
- **One draw.** The displacement and the localization error are independent Gaussians, so the code draws once with standard deviation `sqrt(σ² + blur²)` per axis instead of twice, which has exactly the same law. A blur therefore uses no extra random numbers: the same seed gives the same events, photons, precisions and truth with or without it, and with `blur = 0` exactly the data it always gave.
- **Cost.** One `sqrt` per signal localization. At 10⁷ events the time without a blur did not change measurably (0.37–0.41 s before and after, back to back), and with one it was 0.37–0.50 s, with no new allocations.

### Photon count

With emission a Poisson process at rate `h` running for an `Exponential(τ)` ON period, the photon count marginalises **exactly** to a geometric distribution:

```text
N | T ~ Poisson(hT),  T ~ Exponential(τ)   ⟹   N ~ Geometric(1/(1+hτ)),  E[N] = hτ
```

So the ON duration never has to be sampled — one draw per event, not two. The race interpretation makes it obvious: emission and switch-off compete at rates `h` and `1/τ`, and `N` counts the emissions that win before the switch-off. See `derivations/derivation.qmd`.

That marginal puts mass `p = 1/(1+hτ)` on `N = 0`. An event that emits no detected photons leaves nothing to fit, so it is dropped — not as a tunable threshold, but because there is no localization to report. `rate` therefore counts **emission events**, and a fraction `p` of them leave no trace.

The rows that survive follow the law conditioned on `N ≥ 1`:

```text
P(N = n | N ≥ 1) = (1-p)^(n-1) p,   n = 1, 2, ...     mean 1 + hτ
```

So `mean(locs.nphotons)` sits one above `meanphotons(o)`, and `length(locs)` sits a fraction `p` below the number of events.

One consequence worth remembering: the induced spread of `σ = s/√N` is heavy-tailed (Fréchet with shape 2, whose variance diverges), because a rare one- or two-photon event gives an enormous uncertainty. Summarise simulated uncertainties with **medians**, not means.

### Relation to a population of blinking emitters

A constant event rate is the limit of a large pool of fluorophores that each wait in the dark and then emit once. `M` emitters each waiting `Exponential(1/k_on)` before firing once and bleaching give

```text
λ(t) = M · k_on · exp(-k_on · t)
```

which is constant only while `k_on · t ≪ 1` — while the pool is barely touched. Over a longer acquisition the pool depletes and the rate decays; the simulator assumes the sparse regime where it does not.

Because each emitter fires exactly once in that picture, **drawing a fresh position per event is not an approximation** — a fluorophore that emits once is statistically indistinguishable from a position sampled anew. That equivalence is what lets the simulator drop the emitter list for continuous shapes. Where persistent labels do matter, a `PointCloud` of fixed label sites brings them back.

## Spherical caps: `Bend`

A cap is the unit sphere bent by `Bend(kappa)` (κ below; the field `kappa`), the constant-area map (`@eq-bend` in the derivations), implemented as an `AbstractDeformation{3}` acting on material points from `SphereSurface{3}()` — the unit sphere centred at the origin, so `h = 1 - u_z`. `κ` is dimensionless, `-1 ≤ κ ≤ 1`, and may be a constant or `t -> κ`: `0` is a flat disk of radius 2, `±1` the whole sphere. The apex sits at the origin with outward normal `+z`, and an outer `Similarity` sets the size, orientation and position: with scale `s` and rotation `R`, the cap has area `4π s²`, physical curvature `κ / s`, and outward normal `R * ẑ` at the apex. `rotation(n)` gives the `R` that points that normal along `n`.

```julia
# area 4π·60² nm², flat at t = 0 and about a hemisphere (κ = 1/√2) at t = 10,
# apex at z = 200 nm, tilted 30° about x
cap = Structure(
    SphereSurface{3}(),
    Similarity{3}(scale = 60.0, rot = rotation(SA[1.0, 0, 0], π / 6), shift = SA[0.0, 0, 200])
      ∘ Bend(t -> 0.07t),
)
```

Every material point keeps the area above it, so labels stay uniform on the cap at every curvature. Varying `s` and `κ` in time covers both growth models:

- **constant area** (a flat lattice that curves): fixed `scale`, varying `κ`;
- **constant curvature** (a cap that grows): `scale = t -> s(t)` with `Bend(t -> c * s(t))` for a physical curvature `c`.

Implementation notes:

- **Constant event rate.** A cap growing at constant curvature keeps the same total event rate, so it does not gain labels: its label density falls as `1/A(t)`. Time-varying rates were removed from SimSMLM deliberately, so this is accepted rather than fixed.
- **Relation to the fitters.** Simulated ground truth maps one-to-one onto the fitted cap parameters:
  - DynMoFit: this is the constant-area map of `DynMoFit.jl/derivations/derivation.qmd`. Its unit lattice is centred at `(0, 0, -1)`, so its depth is `-z` where SimSMLM's is `1 - u_z`. Its cap `(κ, scale, normal n, apex a)` is `Similarity{3}(scale = scale, rot = rotation(n), shift = a) ∘ Bend(κ)`.
  - LocMoFit: its `SphericalCap` with curvature `c`, half-angle `α`, orientation `(θ, ϕ)` and apex `(x, y, z)` is `Similarity{3}(scale = sin(α/2) / c, rot = rotation(n), shift = SA[x, y, z]) ∘ Bend(sin(α/2))` with `n = (sin θ cos ϕ, sin θ sin ϕ, cos θ)`. For the same material points the two agree point for point, to rounding (checked to 2e-13).
- **`rotation(n)`** is the shortest rotation taking `+z` onto `n`, about `ẑ × n` — the convention of LocMoFit's `rotmat(θ, ϕ)` and DynMoFit's `rotation_matrix(n)`. It goes through Rodrigues with the angle `atan(sin θ, cos θ)`, which stays accurate at both poles; DynMoFit's closed form divides by `1 + n_z`. At `n = -ẑ` the axis is undefined and `+y` is used, matching LocMoFit's `rotmat(π, 0)`.
- **Numerics.**
  - `normalize(randn(...))` can give `|u_z|` a rounding error above 1, which would make `h(2 - κ²h)` slightly negative; `h` is clamped to `[0, 2]`.
  - The azimuth is computed as `(u_x, u_y) / ρ` with `ρ = √(u_x² + u_y²)`, not through the ratio `ρ′/ρ = √((2 - κ²h)/(2 - h))`, which divides by zero at the south pole.
  - At the north pole `ρ′ = 0`, so the undefined direction does not matter. At the south pole the image is the whole rim, and the code takes `φ = 0`.
- **Validation.**
  - A constant `κ` is checked once, in the inner constructor (`ArgumentError`).
  - A time-varying `κ` can only be checked when evaluated, so `place` checks it at every call and throws a `DomainError` from a `@noinline` helper, keeping the hot path small.
  - `Bend` assumes `|u| = 1` and does not check it: applying it to anything other than the unit sphere is meaningless.

## Composite structures: `Composite`

`Composite(structure => weight, ...)` images several structures together. Each keeps its own shape and deformation, so parts can move independently, and receives a share of the signal events. Motion shared by every component, such as drift, is composed into each with `∘`.

```julia
# a ring, and a cap 50 nm above it that bends while the ring stays flat
scr = Composite(
    Structure(Ring{3}(), Similarity{3}(scale = 50.0)) => 0.4,
    Structure(SphereSurface{3}(),
              Similarity{3}(scale = 40.0, shift = SA[0.0, 0, 50]) ∘ Bend(t -> 0.1t)) => 0.6,
)
locs, truth = simulate(scr; emis_rate = 2000.0, t_end = 5.0, optics = optics)

truth.component[i]                 # 1 or 2, or 0 for a spurious localization
truth.motion[truth.component[i]]   # the deformation that placed row i
```

It implements the splitting of `@eq-thinning`: for each signal event, one uniform draw against the normalised cumulative weights picks the component, which then samples and places the event exactly as a `Structure` would. The photon and precision stages never look at the component, so a composite is identical in law to its components simulated at `π_k · emis_rate` and merged.

- **Weights** are fractions of *events*, not label densities: for an equal density, make them proportional to each component's length or area. They are constant, so if a component's size changes over time, its share of events does not.
- **Type stability.** Components are kept in a `Tuple`, and `_pick` recurses on `Base.tail`, so every branch has a concrete type even when components mix shape and deformation types. Measured at 10⁷ events: 0.77 s against 0.68 s for a plain `Structure`, with the same 52 allocations; the hot loop stays allocation-free.
- **No nesting.** Components must be `Structure`s, not other `Composite`s, so `truth.component[i]` names the deformation that placed row `i` unambiguously. A nested composite can be flattened by multiplying the weights.
- **`Truth`.** `component` is `1` throughout for a plain `Structure` and `0` for background rows. `motion` is the structure's deformation for a `Structure` and a tuple of the components' deformations for a `Composite`.
- **Random stream.** A composite uses one extra uniform per signal event. A plain `Structure` consumes exactly the draws it did before `Composite` existed, so existing seeds reproduce existing data.

## Storage

A `LocalizationTable` is a `StructArray` of `Localization`s, but each column is backed by one contiguous `D × N` matrix, viewed through `columnview` as `N` static vectors. The table therefore has both faces at no cost, because they are literally the same bytes:

```julia
locs[i].position       # SVector — stack-allocated, nothing on the heap
locs.position[i]       # the same SVector
positions(locs)        # the D×N matrix itself; parent(locs.position)
```

Iterating rows, iterating the point column, and scanning the raw matrix all allocate zero bytes. Writing through `positions(locs)` is visible in `locs[i]`, since there is only one copy of the data.

`Truth` is stored the same way, so `place(truth.motion, truth.u[i], locs.t[i])` takes static vectors directly. For a `Composite`, `truth.motion` holds one deformation per component: use `truth.motion[truth.component[i]]`.

## Export

`smaptable(locs)` and `smaptable(truth)` flatten the localizations and their ground truth into column vectors named as in SMAPpy, so downstream tools read simulated and experiment data the same way. Where SMAP and SMAPpy differ, SMAPpy's names win.

- **Names.** Position `x_nm`, `y_nm`, `z_nm`; per-axis precision `x_err_nm`, `y_err_nm`, `z_err_nm` (SMAPpy's rule: every precision is `<quantity>_err_<unit>`); lateral precision `xy_err_nm`, the RMS of the two, which SMAPpy tables carry alongside them (SMAP's `locprecnm`); photons `photons`.
- **Time.** SMAP and SMAPpy tables count `frame`s. The exact event time goes in `time_s`, SMAPpy's name for a time in seconds. Never `t`: SMAPpy's CSV reader maps a column named `t` to `frame`.
- **Truth is its own table.** As in SMAPpy, where the true spots are a second table compared against the localizations, the truth uses the same position names `x_nm`, `y_nm`, `z_nm`, plus `is_signal` and `component`. Row `i` is the truth behind localization `i`, so the two tables join by row; times live in the localization table only.
- **Naming, not compatibility.** The names follow SMAPpy's conventions; exports are not tested against SMAPpy's CSV reader, which is still incomplete.
- **No dependency.** A `NamedTuple` of vectors is a Tables.jl table by definition, so `CSV.write` and `DataFrame` accept it while SimSMLM depends on neither. Julia code downstream does not need it at all: `positions(locs)` and `precisions(locs)` are already the `D×N` matrices the fitters take.
- **Copies.** The columns are fresh vectors (row slices of the `D×N` storage), so editing an exported table never changes the simulation.

## Assumptions

Deliberately not modeled:

- **Blinking kinetics.** Events pick their positions independently of one another and of time. On a continuous shape every event is a fresh draw, so there is no clustering around individual labels. A `PointCloud` of fixed labels does give that clustering — each event draws one label, with replacement, so every label is localized repeatedly — but each label's events still form a memoryless Poisson stream: no on/off correlations, no bleaching.
- **A time-varying emission rate.** `emis_rate` is constant, so pool depletion over a long acquisition is not represented.
- **Labeling efficiency as a separable parameter.** On a continuous shape it folds into the event rate along with the emitter count. With discrete labels, a `PointCloud` of a random subset of the label sites expresses it directly.
- **Frames and time binning.** Time is continuous; there is no camera, no exposure, and no readout dead time. A real acquisition would report which frame an event landed in, not when exactly it happened.
- **Motion blur.** The position is evaluated at a single instant, so the ON time is assumed short compared with any timescale on which the structure moves.
- **Merged emitters.** Two events close in space and time would be fitted as one localization in real data; this is what caps the usable duty cycle in practice.
- **Field-of-view clipping.** The `FieldOfView` places background only; signal localizations outside it are still reported.
- **Detection bias.** Every event that emits at least one photon is detected and localized, however dim, and the reported uncertainty is the true one. Real pipelines miss faint events and misreport their precision.
- **Axial PSF width** is a constant multiple of the lateral one, independent of z.
