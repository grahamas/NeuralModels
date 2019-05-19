
abstract type AbstractNeuralData{T,N} <: DEDataArray{T,N} end

abstract type AbstractNeuralData{T,N} <: AbstractNeuralData{T,N} end
abstract type AbstractNeuralFieldData{T,N} <: AbstractNeuralData{T,N} end
