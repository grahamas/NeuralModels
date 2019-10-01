module NeuralModels

using Simulation73
using Parameters
using DifferentialEquations: DEDataArray
using RecursiveArrayTools
#using TensorOperations
using MacroTools: splitdef, combinedef, splitarg
using StaticArrays
using FFTW
using LinearAlgebra



export AbstractConnectivityParameter, AbstractConnectivityAction

# Exporting Connectivities
export GaussianConnectivityParameter, ExpSumAbsDecayingConnectivityParameter, directed_weights,
    AbstractExpDecayingConnectivityParameter, FFTParameter, FFTAction

# Exporting Nonlinearities
export AbstractNonlinearity,
    SigmoidNonlinearity, GaussianNonlinearity, Sech2Nonlinearity

export AbstractHeterogeneousNeuralData, AbstractHomogeneousNeuralData

export make_mutator

export AbstractStimulusParameter, AbstractStimulusAction,
    NoStimulusParameter, NoStimulusAction,
    GaussianNoiseStimulusParameter, GaussianNoiseStimulusAction,
    AbstractTransientBumpStimulusParameter, TransientBumpStimulusAction,
    SharpBumpStimulusParameter

include("helpers.jl")
include("neural_data.jl")
include("connectivity.jl")
include("nonlinearity.jl")
include("stimulus.jl")
end # module
