
############## Binary switch ##############

binary_switch_off_on(x, θ_on) = θ_on <= x ? 1. : 0.
binary_switch_off_on_off(x, θ_on, θ_off) = θ_on <= x <= θ_off ? 1. : 0.

############## Sigmoids ##############

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
function simple_sigmoid(x, a, theta)
    1.0 / (1 + exp(-a * (x - theta)))
end

function inverse_simple_sigmoid(fx, a, theta)
    - (1 / a) * log((1 / fx) - 1) + theta
end

function zeroed_sigmoid(x, a, theta)
    simple_sigmoid(x, a, theta) - simple_sigmoid(0., a, theta)
end

"""
A rectified version of `simple_sigmoid`.

In practice, we use rectified sigmoid functions because firing rates cannot be negative.

TODO: Rename to rectified_sigmoid.
"""
function rectified_zeroed_sigmoid(x, a, theta)
    max(0, zeroed_sigmoid(x, a, theta))
end

function rectified_unzeroed_sigmoid(x, a, theta)
    max(0, simple_sigmoid(x, a, theta))
end

############## Sech^2 ##############

function sech2(x, a, θ)
    1 - tanh(a * (x - θ))^2
end

############## Gaussian ##############

function gaussian(x, sd, θ)
    exp(-((x - θ) / sd)^2 ) - exp(-(-θ / sd)^2)
end

############## Difference of Sigmoids ##############

function difference_of_simple_sigmoids(x, a_up, θ_up, a_down, θ_down)
    simple_sigmoid(x, a_up, θ_up) - simple_sigmoid(x, a_down, θ_down)
end

############## Product of Sigmoids ##############

function product_of_simple_sigmoids(x, a_up, θ_up, a_down, θ_down)
    simple_sigmoid(x, a_up, θ_up) * (1-simple_sigmoid(x, a_down, θ_down))
end

########################################################