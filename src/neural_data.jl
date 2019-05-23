
abstract type AbstractHomogeneousNeuralData{T,N} <: DEDataArray{T,N} end
abstract type AbstractHeterogeneousNeuralData{T,N} <: AbstractVectorOfArray{T,N} end

@inline unit(data::AbstractHeterogeneousNeuralData, ix) = @view data[ix]
