
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

abstract type MultipleStimuli{T,N} <: AbstractStimulus{T,N} end
struct MultipleDifferentStimuli{T,N} <: MultipleStimuli{T,N}
    stimuli::AbstractArray{AbstractStimulus{T,N}}
end
# Allow for pop arrays
function MultipleDifferentStimuli{T,N}(arr::AbstractArray{<:AbstractArray}) where {T,N}
    [MultipleDifferentStimuli{T,N}(collect(stims)) for stims in zip(arr...)]
end
function make_stimulus(stims::MultipleStimuli, space::AbstractSpace)
    stimulus_mutators! = make_stimulus.(stims.stimuli, Ref(space)) |> collect
    (val, t) -> (stimulus_mutators! .|> (fn!) -> fn!(val,t))
end

struct MultipleSameStimuli{T,N,S,NS} <: MultipleStimuli{T,N}
    stimuli::NTuple{NS,S}
end
function expand_to_tuple(val, ::Type{Val{NS}}) where NS
    NTuple{NS}(val for _ in 1:NS)
end
function expand_to_tuple(tup::NTuple{NS}, ::Type{Val{NS}}) where NS 
    tup
end
function MultipleSameStimuli{T,N,S,NS}(; kwargs...) where {T,N,NS,S <: AbstractStimulus{T,N}}
    arg_names, arg_values = keys(kwargs), values(kwargs)
    arg_expanded_values = expand_to_tuple.(collect(arg_values), Val{NS})
    list_of_arg_values = zip(arg_expanded_values...)
    MultipleSameStimuli{T,N,S,NS}(
                  NTuple{NS,S}(
                      S(; Dict{Symbol,Any}(name => value for (name, value) in zip(arg_names, arg_values))...)
                  for arg_values in list_of_arg_values
                  )
             )
end
