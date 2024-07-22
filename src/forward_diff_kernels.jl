#
# Copyright (c) 2023 Tim Oliver Schneider, Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using NNlib

import ForwardDiff: Dual, Partials

_view(X, colons, k) = view(X, colons..., k...)
_view(X, colons, k::Union{Integer, CartesianIndex}) = view(X, colons..., k)

maximum_dims(dims::AbstractArray{<:Integer}) = (maximum(dims),)
function maximum_dims(dims::AbstractArray{NTuple{N, T}}) where {N, T}
    ntuple(i -> maximum(x -> x[i], dims), N)
end
function maximum_dims(dims::AbstractArray{CartesianIndex{N}}) where {N}
    ntuple(i -> maximum(x -> x[i], dims), N)
end

scatter_empty(op, T) = Base.reduce_empty(op, T)
scatter_empty(op::typeof(-), T) = zero(T)
scatter_empty(op::typeof(/), T) = one(T)
scatter_empty(op::typeof(min), T) = typemax(T)
scatter_empty(op::typeof(max), T) = typemin(T)
# scatter_empty(op::typeof(mean), T) = zero(T)

function NNlib.scatter(op::OP, src::AnyCuArray{Tsrc, Nsrc}, idx::AnyCuArray{Tidx, Nidx};
        init = nothing, dstsize = nothing) where {
        Tsrc <: Dual{T, V, N}, Tidx, Nsrc, Nidx, OP} where {T, V, N}
    dims = Nsrc - Nidx
    dstsz = isnothing(dstsize) ? (size(src)[1:dims]..., maximum_dims(idx)...) : dstsize
    dst = similar(src, Tsrc, dstsz)
    xinit = isnothing(init) ? scatter_empty(op, Tsrc) : init
    fill!(dst, xinit)

    return my_scatter(op, dst, src, idx)
end

function my_scatter(op::OP,
        dst::AnyCuArray{Tdst, Ndst},
        src::AnyCuArray{Tsrc, Nsrc},
        idx::AnyCuArray{Tidx, Nidx}) where {
        Tsrc <: Dual{T, V, N}, Nsrc, Tidx, Nidx, Tdst, Ndst, OP} where {T, V, N}

    # Allocate Cuda Memory for the Kernels to read from
    dst_i = CuArray{V}(undef, (size(dst)..., N + 1))
    src_i = CuArray{V}(undef, (size(src)..., N + 1))

    # Create a tuple similar to Tidx but with one more element
    if Tidx <: Tuple
        t = typeof(ntuple(x -> one(Tidx.parameters[1]), length(t.parameters) + 1))
    else
        t = typeof(ntuple(x -> one(Tidx), 2))
    end

    # Allocate Space for the larger idx array that is need for scatter!
    idx_i = CuArray{t}(undef, (size(idx)..., N + 1))

    # Copy dst, src and idx to there cuArray Copy
    copy_dual_to_matrix(dst_i, dst)
    copy_dual_to_matrix(src_i, src)
    modify_index(idx_i, idx)

    # Now the scatter! function of the cuDNN library can be run
    NNlib.scatter!(op, dst_i, src_i, idx_i)

    # Create an array that contains tuples for the partials of a Dual Number
    #function f(x...)
    #    return x
    #end
    #tuples = f.([view(dst_i, repeat([:], ndims(dst_i)-1)..., i) for i in 2:(N+1)]...)

    slices = [view(dst_i, repeat([:], ndims(dst_i) - 1)..., i) for i in 2:(N + 1)]
    tuples = cu(reshape(
        [Tuple(slices[i][j] for i in 1:length(slices)) for j in 1:prod(size(dst))],
        size(dst)))

    # Copy the Result of the Computation into the destination array
    modify_dual(dst, view(dst_i, repeat([:], ndims(dst_i) - 1)..., 1), tuples)

    return dst
end

function copy_dual_to_matrix(dst_i, dst)
    kernel = CUDA.@cuda launch=false copy_dual_to_matrix_kernel(dst_i, dst)
    config = CUDA.launch_configuration(kernel.fun; max_threads = 256)
    threads = min(config.threads, length(dst_i))
    blocks = cld(length(dst_i), threads)

    kernel(dst_i, dst; threads = threads, blocks = blocks)
end

function copy_dual_to_matrix_kernel(
        src::CuDeviceArray{V}, dst::CuDeviceArray{<:Dual{T, V, N}}) where {T, V, N}
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if index <= length(src)
        # Convert the Linear Index index into Koordinates in src
        ci = CartesianIndices(src)
        loc = Tuple(ci[index])

        # Copy all values of a dual number into there respectiv location
        if loc[end] == 1
            src[loc...] = dst[loc[1:(end - 1)]...].value
        else
            src[loc...] = dst[loc[1:(end - 1)]...].partials[loc[end] - 1]
        end
    end

    return nothing
end

function modify_index(src, dst)
    kernel = CUDA.@cuda launch=false modify_index_kernel(src, dst)
    config = CUDA.launch_configuration(kernel.fun; max_threads = 256)
    threads = min(length(src), config.threads)
    blocks = cld(length(src), threads)

    kernel(src, dst; threads = threads, blocks = blocks)
end

function modify_index_kernel(src::CuDeviceArray, dst::CuDeviceArray)
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if index <= length(src)
        ci = CartesianIndices(src)
        loc = Tuple(ci[index])
        src[loc...] = (dst[loc[1:(end - 1)]...]..., loc[end])
    end

    return nothing
end

function modify_dual(dst, v, src)
    kernel = CUDA.@cuda launch=false dual_kernel(dst, v, src)
    config = CUDA.launch_configuration(kernel.fun; max_threads = 256)
    threads = min(length(dst), config.threads)
    blocks = cld(length(dst), threads)

    kernel(dst, v, src; threads = threads, blocks = blocks)
end

function dual_kernel(dst::CuDeviceArray{<:Dual{T, V, N}}, v::CuDeviceArray{V},
        src::CuDeviceArray{NTuple{N, V}}) where {T, V, N}
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if index <= length(src)
        ci = CartesianIndices(dst)
        loc = Tuple(ci[index])
        dst[loc...] = Dual{T, V, N}(v[loc...], Partials{N, V}(src[loc...]))
    end

    return nothing
end
