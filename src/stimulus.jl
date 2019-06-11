
abstract type AbstractStimulus{T,D} <: AbstractParameter{T} end

function make_mutator(stimulus_arr::AbstractArray{<:AbstractStimulus{T}}, space::AbstractSpace) where T
    stimulus_mutators = [make_stimulus(stim, space) for stim in stimulus_arr]
    function stimulus_mutator!(dA::DATA, A::DATA, t::T) where {T,D,DATA<:AbstractHeterogeneousNeuralData{T,D}}
        @views for (i_stim, stimulus!) in enumerate(stimulus_mutators)
            stimulus!(population(dA,i_stim), t)
        end
    end
end

struct NoStimulus{T,N} <: AbstractStimulus{T,N} end
function make_stimulus(nostim::NoStimulus{T,N}, space::AbstractSpace{T,N}) where {T,N,AT<: AbstractArray{T,N}}
    (val,t) -> return
end

struct MultipleDifferentStimuli{T,N} <: AbstractStimulus{T,N}
    stimuli::AbstractArray{AbstractStimulus{T,N}}
end
# Allow for pop arrays
function MultipleDifferentStimuli{T,N}(arr::AbstractArray{<:AbstractArray}) where {T,N}
    [MultipleDifferentStimuli{T,N}(collect(stims)) for stims in zip(arr...)]
end
function make_stimulus(stims::MultipleDifferentStimuli, space::AbstractSpace)
    stimulus_mutators! = make_stimulus.(stims, Ref(space))
    (val, t) -> (stimulus_mutators! .|> (fn!) -> fn!(val,t))
end

struct MultipleSameStimuli{T,N,S} <: AbstractStimulus{T,N}
    stimuli::Array{S,1}
end
function MultipleSameStimuli{S,NS}(; kwargs...) where {T,N,S <: AbstractStimulus{T,N}}
    arg_names, arg_value_arrays = keys(kwargs), values(kwargs)
    list_of_arg_values = zip(arg_value_arrays...)
    MultipleStimuli{T,N,S}([S(; Dict(name => value for (name, value) in zip(arg_names, arg_values))...)
        for arg_values in list_of_arg_values])
end
