abstract type AbstractConnectivity{T,D} <: AbstractParameter{T} end
abstract type AbstractTensorConnectivity{T,D} <: AbstractConnectivity{T,D} end
macro make_make_tensor_connectivity_mutator(num_dims)
    D = eval(num_dims)
    to_syms = [Symbol(:to,:_,i) for i in 1:D]
    from_syms = [Symbol(:from,:_,i) for i in 1:D]
    D_P = D + 1
    D_CONN_P = D + D + 2
    D_CONN = D + D
    # FIXME using neural_data led to leading pop dimension in dA,A but not in connectivity_tensor
    tensor_prod_expr = @eval @macroexpand @tensor dA[i,$(to_syms...)] = dA[i,$(to_syms...)] + connectivity_tensor[$(to_syms...),$(from_syms...),i,j] * A[j,$(from_syms...)]
    quote
        function make_mutator(conn::AbstractArray{<:AbstractTensorConnectivity{T,$D}}, space::AbstractSpace{T,$D}) where {T}
            connectivity_tensor::Array{T,$D_CONN_P} = directed_weights(conn, space)
            @debug "done."
            function connectivity!(dA::PopsData, A::PopsData, t::T) where {T, PopsData<:AbstractHeterogeneousNeuralData{T,$D}}
                @debug "Performing tensor product..."
                $tensor_prod_expr
                @debug "done."
            end
        end
    end |> esc
end

@make_make_tensor_connectivity_mutator 1
@make_make_tensor_connectivity_mutator 2

function view_slice_last(arr::AbstractArray{T,N}, dx::Int) where {T,N}
    view(arr, ntuple(_ -> Colon(), N - 1)..., dx)
end

function view_slice_last(arr::AbstractArray{T,N}, dx::CartesianIndex{DX}) where {T,N,DX}
    view(arr, ntuple(_ -> Colon(), N - DX)..., dx)
end


function directed_weights(arr::SMatrix{P,P,<:AbstractTensorConnectivity{T,N}}, space::AbstractSpace{T,N}) where {T,N,P}
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

function directed_weights(::Type{ExpSumAbsDecayingConnectivity{T,N}}, coord_distances::Tup, amplitude::T, spread::Tup, step_size::Tup) where {T,N, Tup<:NTuple{N,T}}
    amplitude * prod(step_size) * exp(
        -sum(abs.(coord_distances ./ spread))
    ) / (2 * prod(spread))
end


function directed_weights(::Type{ExpSumSqDecayingConnectivity{T,N}}, coord_distances::Tup, amplitude::T, spread::Tup, step_size::Tup) where {T,N, Tup<:NTuple{N,T}}
    amplitude * prod(step_size) * exp(
        -sum( (coord_distances ./ spread) .^ 2)
    ) / (2 * prod(spread))
end

 function directed_weights(connectivity::CONN, locations::AbstractSpace{T,N}) where {T,N,CONN<:AbstractDecayingConnectivity{T,N}}
    dists = distances(locations)
    step_size = step(locations)
    return directed_weights.(Ref(CONN), dists, connectivity.amplitude, Ref(connectivity.spread), Ref(step_size))
end
