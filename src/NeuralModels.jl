module NeuralModels

using Simulation73
using DifferentialEquations: DEDataArray
using StaticArrays
using FFTW
using LinearAlgebra
using Parameters

export AbstractConnectivityParameter, AbstractConnectivityAction

# Exporting Connectivities
export GaussianConnectivityParameter, ExpAbsSumDecayingConnectivityParameter, directed_weights,
    AbstractExpDecayingConnectivityParameter, FFTParameter, FFTAction

# Exporting Nonlinearities
export AbstractNonlinearity,
    SimpleSigmoidNonlinearity,
    RectifiedZeroedSigmoidNonlinearity,
    RectifiedSigmoidNonlinearity,
    GaussianNonlinearity, Sech2Nonlinearity,
    DifferenceOfSigmoids

export AbstractHeterogeneousNeuralData, AbstractHomogeneousNeuralData

export make_mutator

export AbstractTransientBumpStimulusParameter, TransientBumpStimulusAction,
    CircleStimulusParameter, RectangleStimulusParameter

include("helpers.jl")
include("neural_data.jl")
include("connectivity.jl")
include("nonlinearity.jl")
include("stimulus.jl")
end # module
