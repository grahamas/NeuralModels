
abstract type AbstractHomogeneousNeuralData{T,N} <: DEDataArray{T,N} end
const AbstractHeterogeneousNeuralData{T,N,P} = ArrayPartition{T,<:NTuple{P,<:AbstractHomogeneousNeuralData{T,N}}}

@inline unit(data::AbstractHeterogeneousNeuralData, ix) = @view data[ix]

