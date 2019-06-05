module NeuralModels

using Simulation73
using Parameters
using DifferentialEquations: DEDataArray
using RecursiveArrayTools
using TensorOperations
using MacroTools: splitdef, combinedef, splitarg
using StaticArrays

export AbstractConnectivity, AbstractNonlinearity, AbstractStimulus

# Exporting Connectivities
export ExpSumSqDecayingConnectivity, ExpSumAbsDecayingConnectivity

# Exporting Meshes
export Lattice, PeriodicLattice,
    Segment, Circle, Torus

export origin_idx, distances

# Exporting Nonlinearities
export SigmoidNonlinearity, GaussianNonlinearity, Sech2Nonlinearity

export AbstractHeterogeneousNeuralData, AbstractHomogeneousNeuralData, population

export make_mutator

export NoStimulus

include("helpers.jl")
include("neural_data.jl")
include("meshes.jl")
include("connectivity.jl")
include("nonlinearity.jl")
include("stimulus.jl")
end # module
