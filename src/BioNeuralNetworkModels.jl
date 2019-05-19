module BioNeuralNetworkModels

using Simulation73
using Parameters
using DifferentialEquations: DEDataArray

export AbstractConnectivity, AbstractSpace, AbstractNonlinearity

# Exporting Connectivities
export ExpSumSqDecayingConnectivity, ExpSumAbsDecayingConnectivity

# Exporting Meshes
export Lattice, PeriodicLattice,
    Segment, Circle, Torus

export coordinates, origin_idx, distances

export Pops, one_pop, one_pop_size, one_pop_zero

# Exporting Nonlinearities
export SigmoidNonlinearity, GaussianNonlinearity, Sech2Nonlinearity

include("helpers.jl")
include("meshes.jl")
include("connectivity.jl")

end # module
