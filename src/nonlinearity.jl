abstract type AbstractNonlinearity{T} <: AbstractAction{T} end

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

"""
A rectified version of `simple_sigmoid_fn`.

In practice, we use rectified sigmoid functions because firing rates cannot be negative.

TODO: Rename to rectified_sigmoid_fn.
"""
function rectified_sigmoid_fn(x, a, theta)
    max(0, simple_sigmoid_fn(x, a, theta) - simple_sigmoid_fn(0, a, theta))
end
struct SigmoidNonlinearity{T} <: AbstractNonlinearity{T}
    a::T
    θ::T
    SigmoidNonlinearity(a::T,θ::T) where T = new{T}(a,θ)
end
SigmoidNonlinearity(; a, θ) = SigmoidNonlinearity(a,θ)
(s::SigmoidNonlinearity)(inplace::AbstractArray, ignored_source, ignored_t) = inplace .= rectified_sigmoid_fn.(inplace, s.a, s.θ)


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
GaussianNonlinearity(; sd, θ) where T = GaussianNonlinearity(sd,θ)
(gaussian::GaussianNonlinearity)(output::AbstractArray, ignored_source, ignored_t) = output .= gaussian_fn.(output,gaussian.sd,gaussian.θ)

##############

### Difference of Sigmoids
struct DifferenceOfSigmoids{T} <: AbstractNonlinearity{T}
    firing_sigmoid::SigmoidNonlinearity{T}
    blocking_sigmoid::SigmoidNonlinearity{T}
    DifferenceOfSigmoids(fsig::SigmoidNonlinearity{T},bsig::SigmoidNonlinearity{T}) where T = new{T}(fsig,bsig)
end
DifferenceOfSigmoids(; firing_a, firing_θ, blocking_a, blocking_θ) where T = DifferenceOfSigmoids(SigmoidNonlinearity(; θ=firing_θ,a=firing_a), SigmoidNonlinearity(; θ=blocking_θ,a=blocking_a))
(dos::DifferenceOfSigmoids)(output::AbstractArray, ignored_source, ignored_t) = output .= NeuralModels.rectified_sigmoid_fn.(output, dos.firing_sigmoid.a, dos.firing_sigmoid.θ) - NeuralModels.rectified_sigmoid_fn.(output, dos.blocking_sigmoid.a, dos.blocking_sigmoid.θ)
    
