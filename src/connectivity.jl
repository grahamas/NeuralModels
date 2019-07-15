abstract type AbstractConnectivity{T,N_CDT} <: AbstractParameter{T} end
abstract type AbstractTensorConnectivity{T,N_CDT} <: AbstractConnectivity{T,N_CDT} end
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
        function make_mutator(conn::AbstractArray{<:AbstractTensorConnectivity{T,N_CDT}}, lattice::AbstractLattice{T,$D,N_CDT}) where {T,N_CDT}
            connectivity_tensor::Array{T,$D_CONN_P} = directed_weights(conn, lattice)
            @debug "done."
            function connectivity!(dA::PopsData, A::PopsData, t::T) where {T, PopsData<:AbstractHeterogeneousNeuralData{T,$D_P}}
                @debug "Performing tensor product..."
                $tensor_prod_expr
                @debug "done."
            end
        end
    end |> esc
end

@make_make_tensor_connectivity_mutator 1
@make_make_tensor_connectivity_mutator 2

function directed_weights(arr::SMatrix{P,P,<:AbstractTensorConnectivity{T,N_CDT}}, lattice::AbstractLattice{T,N_ARR,N_CDT}) where {T,N_ARR,N_CDT,P}
    ret_tensor = Array{T,(N_ARR+N_ARR+2)}(undef, size(lattice)..., size(lattice)..., P, P)
    for dx in CartesianIndices(arr)
        view_slice_last(ret_tensor, dx) .= directed_weights(arr[dx], lattice)
    end
    return ret_tensor
end

abstract type AbstractDecayingConnectivity{T,N_CDT} <: AbstractTensorConnectivity{T,N_CDT} end
@with_kw struct ExpSumAbsDecayingConnectivity{T,N_CDT} <: AbstractDecayingConnectivity{T,N_CDT}
    amplitude::T
    spread::NTuple{N_CDT,T}
end
@with_kw struct ExpSumSqDecayingConnectivity{T,N_CDT} <: AbstractDecayingConnectivity{T,N_CDT}
    amplitude::T
    spread::NTuple{N_CDT,T}
end

function directed_weights(::Type{ExpSumAbsDecayingConnectivity{T,N_CDT}}, coord_differences::Tup, amplitude::T, spread::Tup, step_size::Tup) where {T,N_CDT, Tup<:NTuple{N_CDT,T}}
    amplitude * prod(step_size) * exp(
        -sum(abs.(coord_differences ./ spread))
    ) / (2 * prod(spread))
end


function directed_weights(::Type{ExpSumSqDecayingConnectivity{T,N_CDT}}, coord_differences::Tup, amplitude::T, spread::Tup, step_size::Tup) where {T,N_CDT, Tup<:NTuple{N_CDT,T}}
    amplitude * prod(step_size) * exp(
        -sum( (coord_differences ./ spread) .^ 2)
    ) / (2 * prod(spread))
end

 function directed_weights(connectivity::CONN, locations::AbstractLattice{T,N_ARR,N_CDT})::AbstractArray{T} where {T,N_ARR,N_CDT,CONN<:AbstractDecayingConnectivity{T,N_CDT}}
    diffs = differences(locations)
    step_size = step(locations)
    return directed_weights.(Ref(CONN), diffs, connectivity.amplitude, Ref(connectivity.spread), Ref(step_size))
end
