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
    1.0 / 1 + exp(-a * (x - theta))
end

"""
A rectified version of `simple_sigmoid_fn`.

In practice, we use rectified sigmoid functions because firing rates cannot be negative.

TODO: Rename to rectified_sigmoid_fn.
"""
function rectified_sigmoid_fn(x, a, theta)
    max(0, simple_sigmoid_fn(x, a, theta) .- simple_sigmoid_fn(0, a, theta))
end
struct SigmoidNonlinearity{T} <: AbstractNonlinearity{T}
    a::T
    θ::T
    SigmoidNonlinearity(a::T,θ::T) where T = SigmoidNonlinearity{T}(a,θ)
end
SigmoidNonlinearity(; a, θ) = SigmoidNonlinearity(a,θ)
(s::SigmoidNonlinearity)(inplace::AbstractArray) = inplace .= rectified_sigmoid_fn(inplace, a, theta)


############

### Sech2 ###

function sech2_fn(x, a, θ)
    @. 1 - tanh(a * (x - θ))^2
end
struct Sech2Nonlinearity{T} <: AbstractNonlinearity{T}
    a::T
    θ::T
    Sech2Nonlinearity(a::T,θ::T) where T = Sech2Nonlinearity{T}(a,θ)
end
Sech2Nonlinearity{T}(; a, θ) where T = Sech2Nonlinearity(a,θ)
(sn::Sech2Nonlinearity)(output::AbstractArray) = output .= sech2_fn.(output,sn.a,sn.θ)

############

### Gaussian ###

function gaussian_fn(x, sd, θ)
    @. exp(-((x - θ) / sd)^2 ) - exp(-(-θ / sd)^2)
end
struct GaussianNonlinearity{T} <: AbstractNonlinearity{T}
    sd::T
    θ::T
    Sech2Nonlinearity(sd::T,θ::T) where T = Sech2Nonlinearity{T}(sd,θ)
end
GaussianNonlinearity{T}(; a, θ) where T = GaussianNonlinearity(a,θ)
(gaussian::GaussianNonlinearity)(output::AbstractArray) = output .= gaussian_fn.(output,gaussian.sd,gaussian.θ)
