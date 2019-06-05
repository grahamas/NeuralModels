
abstract type AbstractHomogeneousNeuralData{T,N} <: AbstractArray{T,N} end
const AbstractHeterogeneousNeuralData{T,N} = AbstractArray{T,N}

# FIXME not real dispatch, since it's just an alias
@inline population(A::AbstractHeterogeneousNeuralData{T,N}, i) where {T,N} = view_slice_first(A, i)
