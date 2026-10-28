"""
    SimSMLM

Simulation of single-molecule localization microscopy datasets from **moving** structures.

Every emission event is independent. An event happens at a time drawn from a Poisson process; its position is sampled from the structure *as it stands at that instant*; its photon count follows the geometric distribution that an exponential ON period and a Poisson emission rate imply. One event yields at most one localization.

The deformation is the piece that makes this a simulation of a moving structure rather than a static one: sampling a position at time `t` means sampling from the reference shape and pushing that material coordinate forward with [`place`](@ref).

See [`simulate`](@ref) for the entry point.
"""
module SimSMLM

using Random: AbstractRNG, default_rng, randexp
using LinearAlgebra: I, UniformScaling, normalize
using StaticArrays: SVector, SMatrix, StaticArray, @SMatrix
using StructArrays: StructArray
using Distributions: Distributions, Geometric

# time-varying parameters
export at

# core types
export Localization, LocalizationTable, Truth, columnview
export positions, precisions, truepositions, errors, nsignal, nbackground, smaptable

# geometry
export AbstractShape, SphereSurface, Ball, Ring, LineSegment, PointCloud
export sample_site, sample_sites, tosites

# motion
export AbstractDeformation, Static, Similarity, Composed, Drift, Bend
export place, rotation

# events
export eventtimes

# background
export FieldOfView, Background, randin, volume

# optics
export AbstractUncertainty, ShotNoise, StallingaRieger, GaussianMask
export Optics, uncertainty, psfwidths, meanphotons, photondist

# pipeline
export AbstractStructure, Structure, Composite, simulate, sample_position, ndim

include("timevarying.jl")
include("types.jl")
include("deformation.jl")
include("shapes.jl")
include("events.jl")
include("background.jl")
include("optics.jl")
include("simulate.jl")

end # module
