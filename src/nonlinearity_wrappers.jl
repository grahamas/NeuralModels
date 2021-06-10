############## Generic Nonlinearity ##############

abstract type AbstractNonlinearityParameter{T} <: AbstractParameter{T} end
abstract type AbstractNonlinearityAction{T} <: AbstractAction{T} end

#FIXME the parameter/action dichotomy is meaningful elsewhere, but still messy and should be eliminated


struct NonlinearityParameterWrapperAction{T,PARAM<:AbstractNonlinearityParameter{T}} <: AbstractNonlinearityAction{T}
    p::PARAM
end
(param::AbstractNonlinearityParameter)(space::AbstractSpace) = NonlinearityParameterWrapperAction(param)
(wrapper::NonlinearityParameterWrapperAction)(arr1, arr2, t) = wrapper.p(arr1, arr2, t)

############## Sigmoid Nonlinearity ##############
abstract type AbstractSigmoidNonlinearityParameter{T} <: AbstractNonlinearityParameter{T} end
abstract type AbstractSigmoidNonlinearityAction{T} <: AbstractNonlinearityAction{T} end

struct RectifiedZeroedSigmoidNonlinearity{T} <: AbstractSigmoidNonlinearityParameter{T} # rectified, zeroed
    a::T
    θ::T
    RectifiedZeroedSigmoidNonlinearity(a::T,θ::T) where T = new{T}(a,θ)
end
RectifiedZeroedSigmoidNonlinearity(; a, θ) = RectifiedZeroedSigmoidNonlinearity(a,θ)
(s::RectifiedZeroedSigmoidNonlinearity)(inplace::AbstractArray, ignored_source=nothing, ignored_t=nothing) = inplace .= rectified_zeroed_sigmoid.(inplace, s.a, s.θ)


near_zero_start(a,θ) = 0.0 <= rectified_unzeroed_sigmoid(0.0,a,θ) < 0.05
struct RectifiedSigmoidNonlinearity{T} <: AbstractSigmoidNonlinearityParameter{T} #rectified, unzeroed
    a::T
    θ::T
    function RectifiedSigmoidNonlinearity(a::T,θ::T) where T
        if !near_zero_start(a, θ)
            @warn "RectifiedSigmoidNonlinearity non-zero near zero input ($(rectified_unzeroed_sigmoid(0.0,a,θ)))"
            return missing
        end
        new{T}(a,θ)
    end
end
RectifiedSigmoidNonlinearity(; a, θ) = RectifiedSigmoidNonlinearity(a,θ)
(s::RectifiedSigmoidNonlinearity)(inplace::AbstractArray, ignored_source=nothing, ignored_t=nothing) = inplace .= rectified_unzeroed_sigmoid.(inplace, s.a, s.θ)

struct SimpleSigmoidNonlinearity{T} <: AbstractSigmoidNonlinearityParameter{T} #unrectified, unzeroed
    a::T
    θ::T
end
SimpleSigmoidNonlinearity(; a, θ) = SimpleSigmoidNonlinearity(a,θ)
(s::SimpleSigmoidNonlinearity)(inplace::AbstractArray, ignored_source=nothing, ignored_t=nothing) = inplace .= simple_sigmoid.(inplace, s.a, s.θ)

############

### Sech2 ###


struct Sech2Nonlinearity{T} <: AbstractNonlinearityParameter{T}
    a::T
    θ::T
    Sech2Nonlinearity(a::T,θ::T) where T = new{T}(a,θ)
end
Sech2Nonlinearity(; a, θ) = Sech2Nonlinearity(a,θ)
(sn::Sech2Nonlinearity)(output::AbstractArray, ignored_source=nothing, ignored_t=nothing) = output .= sech2.(output,sn.a,sn.θ)

############

### Gaussian ###

struct GaussianNonlinearity{T} <: AbstractNonlinearityParameter{T}
    sd::T
    θ::T
    GaussianNonlinearity(sd::T,θ::T) where T = new{T}(sd,θ)
end
GaussianNonlinearity(; sd, θ) = GaussianNonlinearity(sd,θ)
(nonl::GaussianNonlinearity)(output::AbstractArray, ignored_source=nothing, ignored_t=nothing) = output .= gaussian.(output, nonl.sd, nonl.θ)

### Difference of Sigmoids
abstract type AbstractDifferenceOfSigmoidsParameter{T} <: AbstractNonlinearityParameter{T} end
abstract type AbstractDifferenceOfSigmoidsAction{T} <: AbstractNonlinearityAction{T} end

#FIXME this is stupid; just use the functions, don't nest
function dos_fn!(output::AbstractArray, scratch, 
        firing_sigmoid, blocking_sigmoid, 
        ignored_source=nothing, ignored_t=nothing
    )
    scratch .= output
    blocking_sigmoid(scratch, ignored_source, ignored_t)
    firing_sigmoid(output, ignored_source, ignored_t)
    output .-= scratch
end

function nonnegative_everywhere(fsig::AbstractSigmoidNonlinearityParameter{T},
        bsig::AbstractSigmoidNonlinearityParameter{T}
    ) where T
    max_θ = max(fsig.θ, bsig.θ)
    min_a = min(fsig.a, fsig.a)
    max_test_val = max_θ + (10.0 / min_a)
    test_step = min_a / 10.0
    test_vals = 0.0:test_step:max_test_val |> collect
    dos_fn!(test_vals, copy(test_vals), fsig, bsig)
    all(test_vals .>= 0)
end

struct DifferenceOfSigmoidsParameter{T,S<:AbstractSigmoidNonlinearityParameter{T}} <: AbstractDifferenceOfSigmoidsParameter{T}
    firing_sigmoid::S
    blocking_sigmoid::S
    function DifferenceOfSigmoidsParameter(fsig::S,
                                   bsig::S) where {T, S<:Union{RectifiedSigmoidNonlinearity{T},RectifiedZeroedSigmoidNonlinearity{T}}}
        # enforce that rectified sigmoids are still rectified
        if !nonnegative_everywhere(fsig, bsig)
            return missing
        end
        new{T,S}(fsig,bsig)
    end
    DifferenceOfSigmoidsParameter(fsig::S,bsig::S) where {T, S<:AbstractSigmoidNonlinearityParameter{T}} = new{T,S}(fsig, bsig)
end
DifferenceOfSigmoidsParameter(::Any, ::Missing) = missing
DifferenceOfSigmoidsParameter(::Missing, ::Any) = missing
DifferenceOfSigmoidsParameter(::Missing, ::Missing) = missing

DifferenceOfSigmoidsParameter(sigmoid_type=RectifiedSigmoidNonlinearity; firing_a, firing_θ, blocking_a, blocking_θ) = DifferenceOfSigmoidsParameter(sigmoid_type(; θ=firing_θ,a=firing_a), sigmoid_type(; θ=blocking_θ,a=blocking_a))

get_firing_sigmoid(dosp::DifferenceOfSigmoidsParameter) = dosp.firing_sigmoid
get_blocking_sigmoid(dosp::DifferenceOfSigmoidsParameter) = dosp.blocking_sigmoid


struct DifferenceOfSigmoids{T,DOSP<:DifferenceOfSigmoidsParameter{T},ARR} <: AbstractDifferenceOfSigmoidsAction{T}
    dosp::DOSP
    scratch::ARR
end

(dosp::DifferenceOfSigmoidsParameter)(space::AbstractSpace) = DifferenceOfSigmoids(dosp, zero(space))

function (dos::DifferenceOfSigmoids)(output::AbstractArray, ignored_source=nothing, ignored_t=nothing)
    dos_fn!(output, dos.scratch, dos.dosp.firing_sigmoid, dos.dosp.blocking_sigmoid, ignored_source, ignored_t)
end



struct NormedDifferenceOfSigmoidsParameter{T,S<:AbstractSigmoidNonlinearityParameter{T},DOSP<:DifferenceOfSigmoidsParameter{T,S}} <: AbstractDifferenceOfSigmoidsParameter{T}
    dosp::DOSP
    function NormedDifferenceOfSigmoidsParameter(dos::DOSP) where {T, S<:Union{RectifiedSigmoidNonlinearity{T},RectifiedZeroedSigmoidNonlinearity{T}}, DOSP<:DifferenceOfSigmoidsParameter{T,S}}
        # enforce that rectified sigmoids are still rectified
        new{T,S,DOSP}(dos)
    end
    NormedDifferenceOfSigmoidsParameter(firing, blocking) = NormedDifferenceOfSigmoidsParameter(DifferenceOfSigmoidsParameter(firing, blocking))
    NormedDifferenceOfSigmoidsParameter(::Missing) = missing
end

get_firing_sigmoid(ndosp::NormedDifferenceOfSigmoidsParameter) = get_firing_sigmoid(ndosp.dosp)
get_blocking_sigmoid(ndosp::NormedDifferenceOfSigmoidsParameter) = get_blocking_sigmoid(ndosp.dosp)

function NormedDifferenceOfSigmoidsParameter(sigmoid_type=RectifiedSigmoidNonlinearity; firing_a, firing_θ, blocking_a, blocking_θ) 
    NormedDifferenceOfSigmoidsParameter(
         DifferenceOfSigmoidsParameter(sigmoid_type; 
                firing_θ = firing_θ,
                firing_a = firing_a,
                blocking_θ = blocking_θ,
                blocking_a = blocking_a
            )
        )
end

struct NormedDifferenceOfSigmoids{T,DOS<:DifferenceOfSigmoids{T}} <: AbstractDifferenceOfSigmoidsAction{T}
    dos::DOS
    norm_factor::T
end

function calc_norm_factor(dosp::DifferenceOfSigmoidsParameter{T}) where T
    # FIXME should restrict to provided space
    dos = DifferenceOfSigmoids(dosp, T[0.0])
    function call_dos(x)
        dos(x)
        return -only(x)  # want max
    end
    θf = get_firing_sigmoid(dosp).θ
    θb = get_blocking_sigmoid(dosp).θ
    best_guess = (θb - θf) / 2 + θf
    # FIXME if af == ab, then best_guess is correct and we can skip optim
    opt_result = optimize(call_dos, T[best_guess], BFGS())
    opt_maximum = -Optim.minimum(opt_result)
    opt_minimizer = Optim.minimizer(opt_result)
    @assert only(opt_minimizer) > 0.0
    @assert 0.0 < opt_maximum <= 1.0
    return 1. / opt_maximum
end


function (ndosp::NormedDifferenceOfSigmoidsParameter{T})(space::AbstractSpace) where T
    dos = DifferenceOfSigmoids(ndosp.dosp, zero(space))
    norm_factor = calc_norm_factor(ndosp.dosp)
    return NormedDifferenceOfSigmoids(dos, norm_factor)
end


function (ndos::NormedDifferenceOfSigmoids)(output::AbstractArray, ignored_source=nothing, ignored_t=nothing)
    dos_fn!(output, ndos.dos.scratch, ndos.dos.dosp.firing_sigmoid, ndos.dos.dosp.blocking_sigmoid, ignored_source, ignored_t)
    output .* ndos.norm_factor
end
