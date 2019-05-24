module NeuralModels

using Simulation73
using Parameters
using DifferentialEquations: DEDataArray
using RecursiveArrayTools
using TensorOperations
using MacroTools: splitdef, combinedef, splitarg
using StaticArrays

export AbstractConnectivity, AbstractSpace, AbstractNonlinearity, AbstractStimulus

# Exporting Connectivities
export ExpSumSqDecayingConnectivity, ExpSumAbsDecayingConnectivity

# Exporting Meshes
export Lattice, PeriodicLattice,
    Segment, Circle, Torus

export coordinates, origin_idx, distances

# Exporting Nonlinearities
export SigmoidNonlinearity, GaussianNonlinearity, Sech2Nonlinearity

export AbstractHeterogeneousNeuralData, AbstractHomogeneousNeuralData

export make_mutator

export NoStimulus

include("neural_data.jl")
include("meshes.jl")
include("connectivity.jl")
include("nonlinearity.jl")
include("stimulus.jl")
end # module
