abstract type AbstractConnectivity{T,D} <: AbstractParameter{T} end
abstract type AbstractTensorConnectivity{T,D} <: AbstractConnectivity{T,D} end

macro make_make_tensor_connectivity_mutator(num_dims)
    D = eval(num_dims)
    to_syms = [Symbol(:to,:_,i) for i in 1:D]
    from_syms = [Symbol(:from,:_,i) for i in 1:D]
    D_P = D + 1
    D_CONN_P = D + D + 2
    D_CONN = D + D
    tensor_prod_expr = @eval @macroexpand @tensor dA[$(to_syms...),i] = dA[$(to_syms...),i] + connectivity_tensor[$(to_syms...),$(from_syms...),i,j] * A[$(from_syms...),j]
    quote
        @memoize Dict function make_mutator(conn::AbstractArray{<:AbstractTensorConnectivity{T,$D}}, space::AbstractSpace{T,$D}) where {T}
            connectivity_tensor::Array{T,$D_CONN_P} = tensor(conn, space)
            function connectivity!(dA::Array{T,$D_P}, A::Array{T,$D_P}, t::T) where T
                $tensor_prod_expr
            end
        end
    end |> esc
end

@make_make_tensor_connectivity_mutator 1
@make_make_tensor_connectivity_mutator 2

function directed_weights(arr::AbstractArray{<:AbstractTensorConnectivity{T,N}}, space::AbstractSpace{T,N}) where {T,N}
    ret_tensor = Array{T,(N+N+2)}(undef, size(space)..., size(space)..., P, P)
    for dx in CartesianIndices(arr)
        view_slice_last(ret_tensor, dx) .= directed_weights(arr[dx], space)
    end
    return ret_tensor
end

abstract type AbstractDecayingConnectivity{T,N} <: AbstractTensorConnectivity{T,N} end
@with_kw struct ExpSumAbsDecayingConnectivity{T,N} <: AbstractDecayingConnectivity{T,N}
    amplitude::T
    spread::NTuple{N,T}
end
@with_kw struct ExpSumSqDecayingConnectivity{T,N} <: AbstractDecayingConnectivity{T,N}
    amplitude::T
    spread::NTuple{N,T}
end
ExpSumAbsDecayingConnectivity{T,1}(;amplitude=nothing::T, spread=nothing::T) where T = ExpSumAbsDecayingConnectivity{T,1}(;amplitude=amplitude, spread=(spread,))
ExpSumSqDecayingConnectivity{T,1}(;amplitude=nothing::T, spread=nothing::T) where T,, = ExpSumSqDecayingConnectivity{T,1}(;amplitude=amplitude, spread=(spread,))


function directed_weights(::Type{ExpSumAbsDecayingConnectivity{T,N}}, coord_distances::Tup, amplitude::T, spread::Tup, step_size::Tup) where {T,N, Tup<:NTuple{N,T}}
    amplitude * step_size * exp(
        -sum(abs.(distance ./ spread))
    ) / (2 * prod(spread))
end


function directed_weights(::Type{ExpSumSqDecayingConnectivity{T,N}}, coord_distances::Tup, amplitude::T, spread::Tup, step_size::Tup) where {T,N, Tup<:NTuple{N,T}}
    amplitude * prod(step_size) * exp(
        -sum( (coord_distances ./ spread) .^ 2)
    ) / (2 * prod(spread))
end

@memoize function directed_weights(connectivity::CONN, locations::AbstractSpace{T,N}) where {T,N,CONN<:AbstractDecayingConnectivity{T,N}}
    distances = get_distances(locations)
    step_size = step(locations)
    return directed_weights.(Ref(CONN), distances, connectivity.amplitude, Ref(connectivity.spread), Ref(step_size))
end
