"""
    Localization{D,T}

One localization in `D` dimensions, as a fitting routine would report it.

- `position`  — noisy position estimate
- `precision` — per-axis standard deviation of that estimate
- `t`         — time of the emission event
- `nphotons`  — photons detected

A dataset is a [`LocalizationTable`](@ref), whose rows are these. There is no
emitter index — each event is an independent draw from the structure, so there
is no persistent emitter to point back to. The noise-free position of each event
is kept alongside, in [`Truth`](@ref).
"""
struct Localization{D,T}
    position::SVector{D,T}
    precision::SVector{D,T}
    t::T
    nphotons::Int32
end

"""
    columnview(m::Matrix{T}, Val(D)) -> AbstractVector{SVector{D,T}}

View a `D × N` matrix as `N` static vectors, sharing memory with it.

This is the hinge of the storage layout: points are written into one contiguous
`D × N` block, and the same bytes are read back as static vectors. Neither view
copies, and writing through either is visible in the other.
"""
function columnview(m::Matrix{T}, ::Val{D}) where {D,T}
    size(m, 1) == D ||
        throw(DimensionMismatch("expected $D rows, got $(size(m, 1))"))
    return reinterpret(reshape, SVector{D,T}, m)
end

# The matrix underneath a column of points, however that column is stored.
_matrix(c::Base.ReinterpretArray) = parent(c)
_matrix(c::AbstractVector{SVector{D,T}}) where {D,T} = reinterpret(reshape, T, c)

"""
    LocalizationTable{D,T}

A simulated dataset: a `StructArray` of [`Localization`](@ref)s.

Every column is stored as one contiguous `D × N` matrix and viewed through
[`columnview`](@ref), so the table has both faces at no cost:

- `locs[i]` is a `Localization` with static-vector fields, built on the stack
- `locs.position[i]` is that point as an `SVector`
- `positions(locs)` is the underlying `D × N` matrix itself

Nothing is converted or copied between them — they are the same bytes.
"""
const LocalizationTable{D,T} = StructArray{Localization{D,T},1}

# Not a constructor for the alias: `LocalizationTable{D,T}` resolves to a
# `StructArray` type, and adding constructors to it would collide with the ones
# StructArrays defines.
function localizationtable(position::Matrix{T}, precision::Matrix{T}, t::Vector{T},
    nphotons::Vector{Int32}, ::Val{D}) where {D,T}
    n = length(t)
    size(position, 2) == n && size(precision, 2) == n && length(nphotons) == n ||
        throw(DimensionMismatch("all columns must have the same length"))
    return StructArray{Localization{D,T}}((
        position=columnview(position, Val(D)),
        precision=columnview(precision, Val(D)),
        t=t, nphotons=nphotons,
    ))
end

"""
    positions(locs) -> D×N matrix

The contiguous matrix underlying the position column. [`precisions`](@ref) does
the same for precision. Neither copies.
"""
positions(l::LocalizationTable) = _matrix(l.position)

"""
    precisions(locs) -> D×N matrix

The contiguous matrix underlying the precision column. See [`positions`](@ref).
"""
precisions(l::LocalizationTable) = _matrix(l.precision)

ndim(::LocalizationTable{D}) where {D} = D

"""
    Truth{D,T,M,C}

Ground truth for a simulated dataset, aligned element-for-element with the
localization table it came from, and stored the same way — contiguous matrices
viewed as columns of static vectors.

- `position` — the noise-free position each localization was drawn from
- `u`        — the material coordinate on the reference shape
- `issignal` — `false` for a spurious localization (see [`Background`](@ref))
- `component` — which component of a [`Composite`](@ref) produced it: `1` for a
  plain [`Structure`](@ref), `0` for a spurious localization
- `motion`   — the deformation in force; for a `Composite`, a tuple with one per
  component, so row `i` was placed by `motion[component[i]]`

Because events that emitted no photons are dropped, this must be built alongside
the localizations rather than reconstructed afterwards.

`u` is `NaN` on background columns: a spurious localization has no material
coordinate, and a silent zero there would quietly corrupt any fit that used it.
"""
struct Truth{D,T,M,C<:AbstractVector{SVector{D,T}}}
    position::C
    u::C
    issignal::Vector{Bool}
    component::Vector{Int32}
    motion::M
end

function Truth{D,T}(position::Matrix{T}, u::Matrix{T}, issignal::Vector{Bool},
    component::Vector{Int32}, motion) where {D,T}
    length(issignal) == length(component) == size(position, 2) == size(u, 2) ||
        throw(DimensionMismatch("all columns must have the same length"))
    pc = columnview(position, Val(D))
    uc = columnview(u, Val(D))
    return Truth{D,T,typeof(motion),typeof(pc)}(pc, uc, issignal, component, motion)
end

Base.length(tr::Truth) = length(tr.issignal)
Base.firstindex(::Truth) = 1
Base.lastindex(tr::Truth) = length(tr)
Base.eachindex(tr::Truth) = Base.OneTo(length(tr))
ndim(::Truth{D}) where {D} = D

"""
    nsignal(truth), nbackground(truth)

How many localizations came from the structure, and how many are spurious.
"""
nsignal(tr::Truth) = count(tr.issignal)
nbackground(tr::Truth) = count(!, tr.issignal)

"""
    truepositions(truth) -> D×N matrix

The contiguous matrix underlying the noise-free position column.
"""
truepositions(tr::Truth) = _matrix(tr.position)

"""
    errors(locs, truth) -> D×N matrix

Per-axis localization error, `position - truth`. Dividing by `precisions(locs)`
should give standard normal residuals.
"""
function errors(locs::LocalizationTable{D,T}, tr::Truth{D,T}) where {D,T}
    length(locs) == length(tr) ||
        throw(DimensionMismatch("$(length(locs)) localizations vs $(length(tr)) truth entries"))
    return positions(locs) .- truepositions(tr)
end

"""
    smaptable(locs)  -> NamedTuple of column vectors
    smaptable(truth) -> NamedTuple of column vectors

A localization table, or its ground truth, as flat columns named as in SMAPpy
(the Python successor of SMAP), for handing to downstream tools.

For the localizations:

| column | content |
|---|---|
| `x_nm`, `y_nm`, `z_nm` | position |
| `x_err_nm`, `y_err_nm`, `z_err_nm` | per-axis precision |
| `xy_err_nm` | lateral precision, the RMS of `x_err_nm` and `y_err_nm` |
| `photons` | detected photons |
| `time_s` | event time, s |

For the truth, a separate table with the same position names:

| column | content |
|---|---|
| `x_nm`, `y_nm`, `z_nm` | noise-free position |
| `is_signal` | `false` for a spurious localization |
| `component` | which part of a [`Composite`](@ref) it came from, `0` for spurious |

Row `i` of the truth table is the truth behind row `i` of the localization
table. In 2D the `z` columns are left out.

A `NamedTuple` of vectors is a table to Tables.jl, so `CSV.write(path,
smaptable(locs))` and `DataFrame(smaptable(truth))` work without SimSMLM
depending on either. The columns are copies.
"""
function smaptable(locs::LocalizationTable{D,T}) where {D,T}
    D in (2, 3) || throw(ArgumentError("smaptable needs D = 2 or 3, got D = $D"))
    p, σ = positions(locs), precisions(locs)
    z = D == 3 ? (; z_nm = p[3, :]) : (;)
    zerr = D == 3 ? (; z_err_nm = σ[3, :]) : (;)
    return (; x_nm = p[1, :], y_nm = p[2, :], z...,
        x_err_nm = σ[1, :], y_err_nm = σ[2, :],
        xy_err_nm = sqrt.((view(σ, 1, :) .^ 2 .+ view(σ, 2, :) .^ 2) ./ 2), zerr...,
        photons = copy(locs.nphotons), time_s = copy(locs.t))
end

function smaptable(tr::Truth{D}) where {D}
    D in (2, 3) || throw(ArgumentError("smaptable needs D = 2 or 3, got D = $D"))
    g = truepositions(tr)
    z = D == 3 ? (; z_nm = g[3, :]) : (;)
    return (; x_nm = g[1, :], y_nm = g[2, :], z...,
        is_signal = copy(tr.issignal), component = copy(tr.component))
end
