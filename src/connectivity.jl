abstract type AbstractConnectivity{T,N_CDT} <: AbstractParameter{T} end
abstract type AbstractTensorConnectivity{T,N_CDT} <: AbstractConnectivity{T,N_CDT} end

# struct FFT{T,N_CDT,C<:AbstractConnectivity{T,N_CDT}} <: AbstractConnectivity{T,N_CDT}
#     connectivity::C
#     kernel_size::NTuple{N_CDT,Int}
#     FFT{T,N_CDT,C}(conn::C,ksize::NT) where {T,N_CDT,C<:AbstractConnectivity{T,N_CDT},NT<:NTuple{N_CDT,Int}} = new(conn, ksize)
# end
struct FFT{T,N_CDT,C<:AbstractConnectivity{T,N_CDT}} <: AbstractConnectivity{T,N_CDT}
    connectivity::C
    FFT{T,N_CDT,C}(conn::C) where {T,N_CDT,C<:AbstractConnectivity{T,N_CDT}} = new(conn)
end
FFT(c::C) where {T,N_CDT,C<:AbstractConnectivity{T,N_CDT}} = FFT{T,N_CDT,C}(c)

function make_mutator(conns::AbstractArray{<:FFT{T}}, space::AbstractPeriodicLattice{T,2}) where T
    # @assert all(map(conns) do conn
    #     all(conn.kernel_size .< space.extent/2) && all(conn.kernel_size .% 2 .== 1)
    # end) # Assert the kernel is smaller in all dimensions than half the space AND odd
    # @assert all(conn.kern)
    kernels = [kernel(conn.connectivity, space) for conn in conns]
    kernels_fftd = rfft.(kernels)

    P = size(conns,1)
    populations = population_repeat(zeros(T,size(space)...), P)
    preallocs = [population(populations, pop) for pop in 1:P]
    first_dim_space = size(space, 1)
    fft_ops = plan_rfft.(preallocs)
    @show [op.ialign for op in fft_ops]
    fftd_preallocs = [op * prealloc for (op, prealloc) in zip(fft_ops, preallocs)]
    ifft_ops = plan_irfft.(fftd_preallocs, first_dim_space) # TODO: try ifft_op = inv(fft_op)
    function connectivity!(dA::PopsData, A::PopsData, t::T) where {T, PopsData<:AbstractHeterogeneousNeuralData{T}}
        #A_fftd = [rfft(population(A,i)) for i in 1:size(A,1)]
        #A_fftd = [fft_op * collect(population(A,i)) for i in 1:size(A,1)]
        for ix in CartesianIndices(kernels_fftd)
            # # mul!(fftd_prealloc, fft_op, population(A, ix[2]))
            # # fftd_prealloc .*= kernels_fftd[ix]
            # # mul!(prealloc, ifft_op, fftd_prealloc)
            # # population(dA, ix[1]) .+= prealloc
            # population(dA, ix[1]) .+= ifft_op * (fft_op * collect(population(A, ix[2])) .* kernels_fftd[ix])
            population(dA, ix[1]) .+= ifft_ops[ix[2]] * ((fft_ops[ix[2]] * population(A, ix[2])) .* kernels_fftd[ix])
        end
    end
end

abstract type AbstractExpDecayingConnectivity{T,N_CDT} <: AbstractTensorConnectivity{T,N_CDT} end
@with_kw struct ExpSumAbsDecayingConnectivity{T,N_CDT} <: AbstractExpDecayingConnectivity{T,N_CDT}
    amplitude::T
    spread::NTuple{N_CDT,T}
end
@with_kw struct ExpSumSqDecayingConnectivity{T,N_CDT} <: AbstractExpDecayingConnectivity{T,N_CDT}
    amplitude::T
    spread::NTuple{N_CDT,T}
end

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
            function connectivity!(dA::PopsData, A::PopsData, t::T) where {T, PopsData<:AbstractHeterogeneousNeuralData{T,$D_P}}
                $tensor_prod_expr
            end
        end
    end |> esc
end

@make_make_tensor_connectivity_mutator 1
@make_make_tensor_connectivity_mutator 2

# Version for directed weights from source
function directed_weights(connectivity::CONN, locations::AbstractLattice{T,N_ARR,N_CDT},
                          source_location::NTuple{N_CDT,T}) where {T,N_ARR,N_CDT,CONN<:AbstractExpDecayingConnectivity{T,N_CDT}}
    diffs = differences(locations, source_location)
    step_size = step(locations)
    return directed_weights.(Ref(CONN), diffs, connectivity.amplitude, Ref(connectivity.spread), Ref(step_size))
end

function directed_weights(connectivity::CONN,
                          locations::AbstractLattice{T,N_ARR,N_CDT}) where {T,N_ARR,N_CDT,CONN<:AbstractExpDecayingConnectivity{T,N_CDT}}
    diffs = differences(locations)
    step_size = step(locations)
    return directed_weights.(Ref(CONN), diffs, connectivity.amplitude, Ref(connectivity.spread), Ref(step_size))
end

function directed_weights(arr::SMatrix{P,P,<:AbstractTensorConnectivity{T,N_CDT}}, lattice::AbstractLattice{T,N_ARR,N_CDT}) where {T,N_ARR,N_CDT,P}
    ret_tensor = Array{T,(N_ARR+N_ARR+2)}(undef, size(lattice)..., size(lattice)..., P, P)
    for dx in CartesianIndices(arr)
        view_slice_last(ret_tensor, dx) .= directed_weights(arr[dx], lattice)
    end
    return ret_tensor
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

function kernel(conn::AbstractConnectivity{T,N_CDT}, lattice::AbstractSpace{T,N_CDT}) where {T,N_CDT}
    directed_weights(conn, lattice, coordinates(lattice)[origin_idx(lattice)])
end
