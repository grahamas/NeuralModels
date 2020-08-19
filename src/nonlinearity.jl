abstract type AbstractNonlinearity{T} <: AbstractAction{T} end
abstract type AbstractSigmoidNonlinearity{T} <: AbstractNonlinearity{T} end

### Sigmoid ###
scalar_exp!(A) = A .= exp.(A)

"""
The sigmoid function is defined

```math
\\begin{align}
\\mathcal{S}(x) = \\frac{1}{1 + \\exp(-a(x - θ))}
\\end{align}
```
where ``a`` describes the slope's steepness and ``θ`` describes translation of the slope's center away from zero.

This is "simple" because in practice we use the rectified sigmoid.
"""
function simple_sigmoid_fn(x, a, theta)
    1.0 / (1 + exp(-a * (x - theta)))
end

function zeroed_sigmoid_fn(x, a, theta)
    simple_sigmoid_fn(x, a, theta) - simple_sigmoid_fn(0., a, theta)
end

"""
A rectified version of `simple_sigmoid_fn`.

In practice, we use rectified sigmoid functions because firing rates cannot be negative.

TODO: Rename to rectified_sigmoid_fn.
"""
function rectified_zeroed_sigmoid_fn(x, a, theta)
    max(0, zeroed_sigmoid_fn(x, a, theta))
end
struct RectifiedZeroedSigmoidNonlinearity{T} <: AbstractSigmoidNonlinearity{T} # rectified, zeroed
    a::T
    θ::T
    RectifiedZeroedSigmoidNonlinearity(a::T,θ::T) where T = new{T}(a,θ)
end
RectifiedZeroedSigmoidNonlinearity(; a, θ) = RectifiedZeroedSigmoidNonlinearity(a,θ)
(s::RectifiedZeroedSigmoidNonlinearity)(inplace::AbstractArray, ignored_source, ignored_t) = inplace .= rectified_zeroed_sigmoid_fn.(inplace, s.a, s.θ)

function rectified_unzeroed_sigmoid_fn(x, a, theta)
    max(0, simple_sigmoid_fn(x, a, theta))
end
near_zero_start(a,θ) = 0.0 <= rectified_unzeroed_sigmoid_fn(0.0,a,θ) < 0.05
struct RectifiedSigmoidNonlinearity{T} <: AbstractSigmoidNonlinearity{T} #rectified, unzeroed
    a::T
    θ::T
    function RectifiedSigmoidNonlinearity(a::T,θ::T) where T
        if !near_zero_start(a, θ)
            return missing
        end
        new{T}(a,θ)
    end
end
RectifiedSigmoidNonlinearity(; a, θ) = RectifiedSigmoidNonlinearity(a,θ)
(s::RectifiedSigmoidNonlinearity)(inplace::AbstractArray, ignored_source, ignored_t) = inplace .= rectified_unzeroed_sigmoid_fn.(inplace, s.a, s.θ)

struct SimpleSigmoidNonlinearity{T} <: AbstractSigmoidNonlinearity{T} #unrectified, unzeroed
    a::T
    θ::T
end
SimpleSigmoidNonlinearity(; a, θ) = SimpleSigmoidNonlinearity(a,θ)
(s::SimpleSigmoidNonlinearity)(inplace::AbstractArray, ignored_source, ignored_t) = inplace .= simple_sigmoid_fn.(inplace, s.a, s.θ)

############

### Sech2 ###

function sech2_fn(x, a, θ)
    @. 1 - tanh(a * (x - θ))^2
end
struct Sech2Nonlinearity{T} <: AbstractNonlinearity{T}
    a::T
    θ::T
    Sech2Nonlinearity(a::T,θ::T) where T = new{T}(a,θ)
end
Sech2Nonlinearity(; a, θ) = Sech2Nonlinearity(a,θ)
(sn::Sech2Nonlinearity)(output::AbstractArray, ignored_source, ignored_t) = output .= sech2_fn.(output,sn.a,sn.θ)

############

### Gaussian ###

function gaussian_fn(x, sd, θ)
    @. exp(-((x - θ) / sd)^2 ) - exp(-(-θ / sd)^2)
end
struct GaussianNonlinearity{T} <: AbstractNonlinearity{T}
    sd::T
    θ::T
    GaussianNonlinearity(sd::T,θ::T) where T = new{T}(sd,θ)
end
GaussianNonlinearity(; sd, θ) = GaussianNonlinearity(sd,θ)
(gaussian::GaussianNonlinearity)(output::AbstractArray, ignored_source, ignored_t) = output .= gaussian_fn.(output,gaussian.sd,gaussian.θ)

##############

function dos_fn!(output::AbstractArray, firing_sigmoid, blocking_sigmoid, ignored_source=nothing, ignored_t=nothing)
    blocked = copy(output) # FIXME should preallocate in struct
    blocking_sigmoid(blocked, ignored_source, ignored_t)
    firing_sigmoid(output, ignored_source, ignored_t)
    output .-= blocked
end

function nonnegative_everywhere(fsig::AbstractSigmoidNonlinearity{T},bsig::AbstractSigmoidNonlinearity{T}) where T
    max_θ = max(fsig.θ, bsig.θ)
    min_a = min(fsig.a, fsig.a)
    max_test_val = max_θ + (10.0 / min_a)
    test_step = min_a / 10.0
    test_vals = 0.0:test_step:max_test_val |> collect
    dos_fn!(test_vals, fsig, bsig)
    all(test_vals .>= 0)
end

### Difference of Sigmoids
struct DifferenceOfSigmoids{T,S<:AbstractSigmoidNonlinearity{T}} <: AbstractNonlinearity{T}
    firing_sigmoid::S
    blocking_sigmoid::S
    function DifferenceOfSigmoids(fsig::S,
                                   bsig::S) where {T, S<:Union{RectifiedSigmoidNonlinearity{T},RectifiedZeroedSigmoidNonlinearity{T}}}
        # enforce that rectified sigmoids are still rectified
        if !nonnegative_everywhere(fsig, bsig)
            return missing
        end
        new{T,S}(fsig,bsig)
    end
    DifferenceOfSigmoids(fsig::S,bsig::S) where {T, S<:AbstractSigmoidNonlinearity{T}} = new{T,S}(fsig, bsig)
end
DifferenceOfSigmoids(::Any, ::Missing) = missing
DifferenceOfSigmoids(::Missing, ::Any) = missing
DifferenceOfSigmoids(::Missing, ::Missing) = missing

DifferenceOfSigmoids(sigmoid_type=RectifiedSigmoidNonlinearity; firing_a, firing_θ, blocking_a, blocking_θ) = DifferenceOfSigmoids(sigmoid_type(; θ=firing_θ,a=firing_a), sigmoid_type(; θ=blocking_θ,a=blocking_a))

function (dos::DifferenceOfSigmoids)(output::AbstractArray, ignored_source, ignored_t)
    dos_fn!(output, dos.firing_sigmoid, dos.blocking_sigmoid, ignored_source, ignored_t)
end


