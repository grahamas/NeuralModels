



### Gaussian Noise ###
struct GaussianNoiseStimulusParameter{T} <: AbstractStimulusParameter{T}
    mean::T
    sd::T
end
struct GaussianNoiseStimulusAction{T,N} <: AbstractStimulusAction{T,N}
    mean::T
    sd::T
end
function GaussianNoiseStimulusParameter(; sd::Union{T,Nothing}=nothing, SNR::Union{T,Nothing}=nothing, mean::T=0.0) where {T}
    @assert xor(sd == nothing, SNR == nothing)
    if sd == nothing
        sd = sqrt(1/10 ^ (SNR / 10))
    end
    GaussianNoiseStimulusParameter(mean, sd)
end
function gaussian_noise!(val::AT, mean::T, sd::T) where {T, AT<:AbstractArray{T}} # assumes signal power is 0db
    val .+= randn(size(val))
    val .*= sd
    val .+= mean
end
function (wns::GaussianNoiseStimulusParameter{T})(space::AbstractSpace{T,N}) where {T,N}
    GaussianNoiseStimulusAction{T,N}(wns.mean, wns.sd) # Not actually time dependent
end
function (wns::GaussianNoiseStimulusAction{T,N})(val::AbstractArray{T,N}, ignored_val, ignored_t) where {T,N}
    gaussian_noise!(val, wns.mean, wns.sd) # Not actually time dependent
end
##########################

### Transient bumps ###
# Subtypes of TransientBumpStimulusParameter generate TransientBumpStimulusActions (NOT subtypes)
abstract type AbstractTransientBumpStimulusParameter{T} <: AbstractStimulusParameter{T} end
struct TransientBumpStimulusAction{T,N,FRAME,WINDOWS} <: AbstractStimulusAction{T,N}
    baseline::T
    bump_frame::FRAME
    time_windows::WINDOWS
    function TransientBumpStimulusAction(baseline::T, bump_frame::FRAME,time_windows::WINDOWS) where {T,N,FRAME<:AbstractArray{T,N},WINDOWS}
        new{T,N,FRAME,WINDOWS}(baseline,bump_frame,time_windows)
    end
end
function (bump_param::AbstractTransientBumpStimulusParameter{T})(space::AbstractSpace{T,N}) where {T,N}
    bump_frame = on_frame(bump_param, space)
    TransientBumpStimulusAction(bump_param.baseline, bump_frame, bump_param.time_windows)
end
function (bump::TransientBumpStimulusAction{T,N})(val::AbstractArray{T,N}, ignored, t::T) where {T,N}
    for window in bump.time_windows
        if window[1] <= t < window[2]
            val .+= bump.bump_frame
        else
            val .+= bump.baseline
        end
    end
end

struct SharpBumpStimulusParameter{T} <: AbstractTransientBumpStimulusParameter{T}
    width::T
    strength::T
    time_windows::Array{Tuple{T,T},1}
    center::Union{NTuple,Nothing}
    baseline::T
end

function SharpBumpStimulusParameter(; strength, width,
        duration=nothing, time_windows=nothing, center=nothing, baseline=0.0)
    if time_windows == nothing
        return SharpBumpStimulusParameter(width, strength, [(zero(typeof(strength)), duration)], center, baseline)
    else
        @assert duration == nothing
        return SharpBumpStimulusParameter(width, strength, time_windows, center, baseline)
    end
end

distance(x1::NTuple{N},x2::NTuple{N}) where N = sqrt(sum((x1 .- x2) .^ 2))
function on_frame(sbs::SharpBumpStimulusParameter{T}, space::AbstractSpace{T,N_ARR,N_CDT}) where {T,N_ARR,N_CDT}
    coords = coordinates(space)
    frame = zero(space) .+ sbs.baseline
    half_width = sbs.width / 2.0
    center_val = sbs.center == nothing ? coords[origin_idx(space)] : sbs.center
    distances = distance.(coords, Ref(center_val))
    on_center = (distances .< half_width) .| (distances .≈ half_width)
    frame[on_center] .= sbs.strength
    return frame
end

################################
