module NeuralModels

using Simulation73
using Parameters
using DifferentialEquations: DEDataArray
using RecursiveArrayTools
using TensorOperations
using MacroTools: splitdef, combinedef, splitarg
using StaticArrays
using FFTW

export AbstractConnectivity, AbstractNonlinearity, AbstractStimulus

# Exporting Connectivities
export ExpSumSqDecayingConnectivity, ExpSumAbsDecayingConnectivity, directed_weights
export AbstractExpDecayingConnectivity, FFT

# Exporting Nonlinearities
export SigmoidNonlinearity, GaussianNonlinearity, Sech2Nonlinearity

export AbstractHeterogeneousNeuralData, AbstractHomogeneousNeuralData, population

export make_mutator

export NoStimulus, MultipleDifferentStimuli, MultipleSameStimuli

include("helpers.jl")
include("neural_data.jl")
include("connectivity.jl")
include("nonlinearity.jl")
include("stimulus.jl")
end # module
