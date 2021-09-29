module NeuralModels

using Simulation73
using Parameters
using RecursiveArrayTools
#using TensorOperations
using MacroTools: splitdef, combinedef, splitarg
using StaticArrays
using FFTW
using LinearAlgebra
using Optim
using AxisIndices

export make_mutator


include("helpers.jl")
include("connectivity.jl")

export AbstractConnectivityParameter, AbstractConnectivityAction,
    GaussianConnectivityParameter, ExpAbsSumDecayingConnectivityParameter, 
    AbstractExpDecayingConnectivityParameter, FFTParameter, FFTAction,
    directed_weights

include("nonlinearity_functions.jl")

export binary_switch_off_on, binary_switch_off_on_off,
    simple_sigmoid, inverse_simple_sigmoid, zeroed_sigmoid,
    rectified_zeroed_sigmoid, rectified_unzeroed_sigmoid,
    sech2,
    gaussian,
    difference_of_simple_sigmoids,
    product_of_simple_sigmoids

include("nonlinearity_wrappers.jl")

export get_firing_fn, get_blocking_fn

export AbstractNonlinearityParameter,
    AbstractNonlinearityAction,
    SimpleSigmoidNonlinearity,
    RectifiedZeroedSigmoidNonlinearity,
    RectifiedSigmoidNonlinearity,
    GaussianNonlinearity, Sech2Nonlinearity,
    DifferenceOfSigmoidsParameter,
    DifferenceOfSigmoids,
    NormedDifferenceOfSigmoidsParameter,
    NormedDifferenceOfSigmoids,
    AbstractSigmoidNonlinearityAction,
    AbstractSigmoidNonlinearityParameter,
    AbstractDifferenceOfSigmoidsAction,
    AbstractDifferenceOfSigmoidsParameter,
    AbstractDifferenceOfErfsAction,
    AbstractDifferenceOfErfsParameter,
    AbstractErfNonlinearityAction,
    AbstractErfNonlinearityParameter,
    ErfNonlinearity,
    DifferenceOfErfs,
    DifferenceOfErfsParameter

include("stimulus.jl")

export AbstractTransientBumpStimulusParameter, TransientBumpStimulusAction,
    CircleStimulusParameter, RectangleStimulusParameter
export stimulus_duration

end # module
