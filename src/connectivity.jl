abstract type AbstractConnectivityParameter{T,N_CDT} <: AbstractParameter{T} end
abstract type AbstractConnectivityAction{T,N_CDT} <: AbstractSpaceInteraction{T,N_CDT} end
struct NaiveConnectivityAction{T,N_CDT,CONN,SPACE} <: AbstractConnectivityAction{T,N_CDT}
    conn::CONN
    space::SPACE
    NaiveConnectivityAction(conn::CONN, space::SPACE) where {T,N,CONN<:AbstractConnectivityParameter{T,N},SPACE<:AbstractSpace{T,N}} = new{T,N,CONN,SPACE}(conn,space)
end
(a::AbstractConnectivityParameter)(space::AbstractSpace) = NaiveConnectivityAction(a,space)
function (a::NaiveConnectivityAction)(output, input, ignored_t)
    coords = coordinates(a.space)
    for (i_coord, coord) in enumerate(coords)
        weights = directed_weights(a.conn, a.space, coord)
        output[i_coord] = sum(weights .* input)
    end
end
        
struct FFTParameter{T,N_CDT,C<:AbstractConnectivityParameter{T,N_CDT}} <: AbstractConnectivityParameter{T,N_CDT}
    connectivity::C
    FFTParameter(c::C) where {T,N_CDT,C<:AbstractConnectivityParameter{T,N_CDT}} = new{T,N_CDT,C}(c)
end
struct FFTAction{T,N_CDT,KERN<:AbstractArray{Complex{T},N_CDT},OP,IOP} <: AbstractConnectivityAction{T,N_CDT}
    kernel::KERN
    fft_op::OP
    ifft_op::IOP
    FFTAction(kernel::KERN, fft_op::OP, ifft_op::IOP) where {T,N,KERN<:AbstractArray{Complex{T},N},OP,IOP} = new{T,N,KERN,OP,IOP}(kernel, fft_op, ifft_op)
end
# TODO: Assumes populations
function (fftp::FFTParameter)(space::AbstractSpace)
    kern = kernel(fftp.connectivity, space)
    kernel_fftd = rfft(kern)
    multi_pop_space = population_repeat(zeros(space),2)
    single_pop_zeros = population(multi_pop_space, 1)
    fft_op = plan_rfft(single_pop_zeros; flags=(FFTW.PATIENT | FFTW.UNALIGNED))
    ifft_op = plan_irfft(fft_op * single_pop_zeros, size(space,1); flags=(FFTW.PATIENT | FFTW.UNALIGNED))
    FFTAction(kernel_fftd, fft_op, ifft_op)
end
function (a::FFTAction)(output::AbstractArray, input::AbstractArray, ignored_t)
    output .+= fftshift(real(a.ifft_op * ((a.fft_op * input) .* a.kernel)))
end

abstract type AbstractExpDecayingConnectivityParameter{T,N_CDT} <: AbstractConnectivityParameter{T,N_CDT} end
(t::Type{<:AbstractExpDecayingConnectivityParameter})(; amplitude, spread) = t(amplitude, spread)
struct ExpSumAbsDecayingConnectivityParameter{T,N_CDT} <: AbstractExpDecayingConnectivityParameter{T,N_CDT}
    amplitude::T
    spread::NTuple{N_CDT,T}
end
struct GaussianConnectivityParameter{T,N_CDT} <: AbstractExpDecayingConnectivityParameter{T,N_CDT}
    amplitude::T
    spread::NTuple{N_CDT,T}
end

function unit_scale(arr::AbstractArray)
    @show sum(arr)
    arr ./ sum(arr)
end

# Version for directed weights from source
function directed_weights(connectivity::CONN, locations::AbstractLattice{T,N_ARR,N_CDT},
                          source_location::NTuple{N_CDT,T}) where {T,N_ARR,N_CDT,CONN<:AbstractExpDecayingConnectivityParameter{T,N_CDT}}
    diffs = differences(locations, source_location)
    step_size = step(locations)
    unscaled = directed_weight_unscaled.(Ref(CONN), diffs, Ref(connectivity.spread), Ref(step_size))
    return connectivity.amplitude .* (unscaled ./ sum(unscaled))
end

function directed_weights(connectivity::CONN,
                          locations::AbstractLattice{T,N_ARR,N_CDT}) where {T,N_ARR,N_CDT,CONN<:AbstractExpDecayingConnectivityParameter{T,N_CDT}}
    diffs = differences(locations)
    step_size = step(locations)
    center_norm_val = sum(directed_weight_unscaled.(Ref(CONN), differences(locations, coordinates(locations)[origin_idx(locations)]), Ref(connectivity.spread), Ref(step_size)))
    return connectivity.amplitude .* (directed_weight_unscaled.(Ref(CONN), diffs, Ref(connectivity.spread), Ref(step_size)) ./ center_norm_val)
end

function directed_weights(connectivity::CONN, locations::AbstractLattice{T,N_ARR,N_CDT},
                          source_location::NTuple{N_CDT,T}) where {T,N_ARR,N_CDT,CONN<:GaussianConnectivityParameter{T,N_CDT}}
    diffs = differences(locations, source_location)
    step_size = step(locations)
    unscaled = directed_weight_unscaled.(Ref(CONN), diffs, Ref(connectivity.spread), Ref(step_size))
    return connectivity.amplitude .* (unscaled ./ prod(connectivity.spread) ./ π^(N_ARR/2) .* prod(step(locations)))
end

function directed_weights(connectivity::CONN,
                          locations::AbstractLattice{T,N_ARR,N_CDT}) where {T,N_ARR,N_CDT,CONN<:GaussianConnectivityParameter{T,N_CDT}}
    diffs = differences(locations)
    step_size = step(locations)
    return connectivity.amplitude .* (
               directed_weight_unscaled.(Ref(CONN), diffs, Ref(connectivity.spread), Ref(step_size)) ./
               (prod(spread) * π^(N_CDT/2) * prod(step(locations))))
end
function directed_weight_unscaled(::Type{ExpSumAbsDecayingConnectivityParameter{T,N_CDT}}, coord_differences::Tup, spread::Tup, step_size::Tup) where {T,N_CDT, Tup<:NTuple{N_CDT,T}}
    exp(
        -sum(abs.(coord_differences ./ spread))
    )
end

function directed_weight_unscaled(::Type{GaussianConnectivityParameter{T,N_CDT}}, coord_differences::Tup, spread::Tup, step_size::Tup) where {T,N_CDT, Tup<:NTuple{N_CDT,T}}
    exp(
        -sum( (coord_differences ./ spread) .^ 2)
    )
end

function kernel(conn::AbstractConnectivityParameter{T,N_CDT}, lattice::AbstractSpace{T,N_CDT}) where {T,N_CDT}
    directed_weights(conn, lattice, coordinates(lattice)[origin_idx(lattice)])
end
