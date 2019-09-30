
abstract type AbstractStimulusParameter{T,D} <: AbstractParameter{T} end
abstract type AbstractStimulusAction{T,D} <: AbstractSpaceAction{T,D} end

# Naturally map stimulus arrays
function (stim_params::AbstractArray{<:AbstractStimulusParameter{T,N}})(space::AbstractSpace{T,N}) where {T,N}
    map(stim_params) do param
        param(space)
    end
end
# Stimulus actions are applied IN ORDER
function (stims::AbstractArray{<:AbstractStimulusAction})(args...)
    for stim in stims
        stim(args...)
    end
end

struct NoStimulusParameter{T,N} <: AbstractStimulusParameter{T,N} end
struct NoStimulusAction{T,N} <: AbstractStimulusAction{T,N} end
function (nostim::NoStimulusParameter{T,N})(space::AbstractSpace{T,N}) where {T,N}
    NoStimulusAction{T,N}()
end
(ns::NoStimulusAction)(args...) = nothing



### Gaussian Noise ###
struct GaussianNoiseStimulusParameter{T,N} <: AbstractStimulusParameter{T,N}
    mean::T
    sd::T
end
struct GaussianNoiseStimulusAction{T,N} <: AbstractStimulusAction{T,N}
    mean::T
    sd::T
end
function GaussianNoiseStimulusParameter{T,N}(; sd::Union{T,Nothing}=nothing, SNR::Union{T,Nothing}=nothing, mean::T=0.0) where {T,N}
    @assert xor(sd == nothing, SNR == nothing)
    if sd == nothing
        sd = sqrt(1/10 ^ (SNR / 10))
    end
    GaussianNoiseStimulusParameter{T,N}(mean, sd)
end
function gaussian_noise!(val::AT, mean::T, sd::T) where {T, AT<:AbstractArray{T}} # assumes signal power is 0db
    val .+= randn(size(val))
    val .*= sd
    val .+= mean
end
function (wns::GaussianNoiseStimulusParameter{T,N})(space::AbstractSpace{T,N}) where {T,N}
    GaussianNoiseStimulusAction{T,N}(wns.mean, wns.sd) # Not actually time dependent
end
function (wns::GaussianNoiseStimulusAction{T,N})(val::AbstractArray{T,N}, ignored_val, ignored_t) where {T,N}
    gaussian_noise!(val, wns.mean, wns.sd) # Not actually time dependent
end
##########################

### Transient bumps ###
# Subtypes of TransientBumpStimulusParameter generate TransientBumpStimulusActions (NOT subtypes)
abstract type AbstractTransientBumpStimulusParameter{T,N} <: AbstractStimulusParameter{T,N} end
struct TransientBumpStimulusAction{T,N,FRAME,WINDOWS} <: AbstractStimulusAction{T,N}
    bump_frame::FRAME
    time_windows::WINDOWS
    function TransientBumpStimulusAction(bump_frame::FRAME,time_windows::WINDOWS) where {T,N,FRAME<:AbstractArray{T,N},WINDOWS}
        new{T,N,FRAME,WINDOWS}(bump_frame,time_windows)
    end
end
function (bump_param::AbstractTransientBumpStimulusParameter{T,N})(space::AbstractSpace{T,N}) where {T,N}
    bump_frame = on_frame(bump_param, space)
    TransientBumpStimulusAction(bump_frame, bump_param.time_windows)
end
function (bump::TransientBumpStimulusAction{T,N,FRAME})(val::FRAME, ignored, t::T) where {T,N,FRAME<:AbstractArray{T,N}}
    for window in bump.time_windows
        if window[1] <= t < window[2]
            val .+= bump.bump_frame
        end
    end
end

struct SharpBumpStimulusParameter{T,N_CDT} <: AbstractTransientBumpStimulusParameter{T,N_CDT}
    width::T
    strength::T
    time_windows::Array{Tuple{T,T},1}
    center::NTuple{N_CDT,T}
end

function SharpBumpStimulusParameter(; strength, width,
        duration=nothing, time_windows=nothing, center)
    if time_windows == nothing
        return SharpBumpStimulusParameter(width, strength, [(zero(typeof(strength)), duration)], center)
    else
        @assert duration == nothing
        return SharpBumpStimulusParameter(width, strength, time_windows, center)
    end
end

distance(x1::NTuple{N},x2::NTuple{N}) where N = sqrt(sum((x1 .- x2) .^ 2))
function on_frame(sbs::SharpBumpStimulusParameter{T,N_CDT}, space::AbstractSpace{T,N_ARR,N_CDT}) where {T,N_ARR,N_CDT}
    coords = coordinates(space)
    frame = zero(space)
    half_width = sbs.width / 2.0
    distances = distance.(coords, Ref(sbs.center))
    on_center = (distances .< half_width) .| (distances .≈ half_width)
    frame[on_center] .= sbs.strength
    return frame
end

################################
