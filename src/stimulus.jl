
abstract type AbstractStimulus{T,D} <: AbstractParameter{T} end

 Dict function make_mutator(stimulus_arr::AbstractArray{<:AbstractStimulus{T}}, space::AbstractSpace) where T
    stimulus_mutators = [make_stimulus(stim, space) for stim in stimulus_arr]
    function stimulus_mutator!(dA::DATA, A::DATA, t::T) where {T,D,DATA<:AbstractHeterogeneousNeuralData{T,D}}
        for (stimulus!, population_dA) in zip(stimulus_mutators, dA)
            stimulus!(population_dA, t)
        end
    end
end

struct NoStimulus{T,N} <: AbstractStimulus{T,N} end
function make_stimulus(nostim::NoStimulus{T,N}, space::AbstractSpace{T,N}) where {T,N,AT<: AbstractArray{T,N}}
    (val,t) -> return
end
