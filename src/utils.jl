#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using KernelAbstractions

"""
    triangles_to_edges(faces::AbstractArray{T, 2} where T <: Integer)

Converts the given faces of a mesh to edges.

# Arguments
- `faces`: Two-dimensional array with the node indices in the first dimension.

# Returns
- A tuple containing the edge pairs. (See [parse_edges](@ref))
"""
function triangles_to_edges(faces::AbstractArray{T, 2} where T <: Integer)
    edges = hcat(faces[1:2, :], faces[2:3, :], permutedims(hcat(faces[3, :], faces[1, :])))
    return parse_edges(edges)
end

"""
    parse_edges(edges)

Converts the given edges to unique pairs of senders and receivers (in both directions).

# Arguments
- `edges`: A two-dimensional Array containing the edges. The first dimension represents a sender-receiver pair.

# Returns
- A tuple containing the bi-directional sender-receiver pairs. The first index is one direction, the second index the other one.
"""
function parse_edges(edges)
    receivers = minimum(edges, dims=1)
    senders = maximum(edges, dims=1)
    packed_edges = vcat(senders, receivers)
    unique_edges = unique(packed_edges, dims=2)
    senders = unique_edges[1, :]
    receivers = unique_edges[2, :]

    return vcat(senders, receivers), vcat(receivers, senders)
end

"""
    one_hot(indices, depth, offset = 0)

Constructs a onehot matrix of Bool with the given indices.

# Arguments
- `indices`: Indices for the onehot matrix.
- `depth`: Depth of the matrix. The second dimension will be clipped or padded with zeros to the depth.
- `offset = 0`: Offset of the matrix in the second dimension.

# Returns
- `result`: The onehot matrix from the given arguments.
"""
function one_hot(indices, depth, offset = 0)
    result = zeros(Bool, depth, length(indices))
    for (i, x) in enumerate(indices)
        if x + offset <= depth && x + offset > 0
            result[x + offset, i] = 1
        end
    end
    return result
end

"""
    minmaxnorm(input::AbstractArray, input_min, input_max, new_min = 0.0f0, new_max = 1.0f0)

Normalizes the given input to the new given range.

# Arguments
- `input`: Data that should be normalized.
- `input_min`: Minimum of the given data.
- `input_max`: Maximum of the given data.
- `new_min = 0.0f0`: New minimum of the normalized data.
- `new_max = 1.0f0`: New maximum of the normalized data.

# Returns
- The normalized data.
"""
function minmaxnorm(input::AbstractArray, input_min, input_max, new_min = 0.0f0, new_max = 1.0f0)
    @assert input_min <= input_max "minimum of input has to be lower than or equal to maximum of input : $input_min > $input_max"
    @assert new_min <= new_max "minimum of output has to be lower than or equal to maximum of output : $new_min > $new_max"
    if input_min == input_max
        return typeof(input) <: CuArray ? gpu_device()(zeros(Float32, size(input))) : zeros(Float32, size(input))
    else
        return ((input .- input_min) / (input_max - input_min)) * (new_max - new_min) .+ new_min
    end
end

"""
    mse_reduce(target, output)

Calculates the mean squared error of the given arguments with [Tullio](@ref) for GPU compatibility.

# Arguments
- `target`: Ground truth from the data.
- `output`: Output of the network.

# Returns
- The calculated mean squared error.
"""
mse_reduce(target, output) = begin
    @assert ndims(target) == 2 && ndims(output) == 2 "Only supported dimension is 2: dims = (target => $(dims(target)), output => $(dims(output))"
    @tullio R[x] := (target[y, x] - output[y, x]) ^ 2
end

"""
    tullio_reducesum(a, dims)

Implementation of the function [reducesum](@ref) with [Tullio](@ref) for GPU compatibility.

# Arguments
- `a`: Array as input for reducesum.
- `dims`: Along which dimension should be reduced. Only dimension 1 and 2 are supported.

# Returns
- The reduced array.
"""
tullio_reducesum(a, dims) = begin
    @assert dims == 1 || dims == 2 "Only supported dims are 1 and 2: dims = $dims"
    if dims == 1
        @tullio R[1, x] := a[y, x]
    elseif dims == 2
        @tullio R[x] := a[x, y]
    end
end
