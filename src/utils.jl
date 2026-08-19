# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
# Copyright (c) 2023 Julian Trommer
#
# This file is a Julia adaptation of code from DeepMind's MeshGraphNets and has
# been modified from the original.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications made for GraphNetCore.jl are licensed under the MIT License.
# See the LICENSE file in the project root for details.

"""
    triangles_to_edges(faces)

Converts the given faces of a mesh to edges.

## Arguments
- `faces`: Two-dimensional array with the node indices in the first dimension.

## Returns
- Tuple containing the edge pairs. (See [`parse_edges`](@ref))
"""
function triangles_to_edges(faces::AbstractArray{T, 2} where {T <: Integer})
    edges = hcat(faces[1:2, :], faces[2:3, :], permutedims(hcat(faces[3, :], faces[1, :])))

    return parse_edges(edges)
end

"""
    parse_edges(edges)

Converts the given edges to unique pairs of senders and receivers (in both directions).

## Arguments
- `edges`: Two-dimensional Array containing the edges. The first dimension represents a sender-receiver pair.

## Returns
- Tuple containing the bi-directional sender-receiver pairs. The first index is one direction, the second index the other one.
"""
function parse_edges(edges)
    receivers = minimum(edges; dims = 1)
    senders = maximum(edges; dims = 1)
    packed_edges = vcat(senders, receivers)
    unique_edges = unique(packed_edges; dims = 2)
    senders = unique_edges[1, :]
    receivers = unique_edges[2, :]

    return vcat(senders, receivers), vcat(receivers, senders)
end

"""
    one_hot(indices, depth, offset = 0)

Constructs a onehot matrix of Float32 with the given indices.

## Arguments
- `indices`: Indices for the onehot matrix.
- `depth`: Depth of the matrix. The second dimension will be clipped or padded with zeros to the depth.
- `offset = 0`: Offset of the matrix in the second dimension.

## Returns
- Onehot matrix from the given arguments.
"""
function one_hot(indices, depth, offset = 0)
    result = zeros(Float32, depth, length(indices))
    for (i, x) in enumerate(indices)
        if x + offset <= depth && x + offset > 0
            result[x + offset, i] = 1
        end
    end

    return result
end

"""
    minmaxnorm(input, input_min, input_max, new_min = 0.0f0, new_max = 1.0f0)

Normalizes the given input to the new given range.

## Arguments
- `input`: Data that should be normalized.
- `input_min`: Minimum of the given data.
- `input_max`: Maximum of the given data.
- `new_min = 0.0f0`: New minimum of the normalized data.
- `new_max = 1.0f0`: New maximum of the normalized data.

## Returns
- Normalized data.
"""
function minmaxnorm(
        input::AbstractArray, input_min, input_max, new_min = 0.0f0, new_max = 1.0f0)
    @assert minimum(input_max - input_min)>0.0f0 "minimum of input has to be lower than maximum of input : $input_min >= $input_max"
    @assert minimum(new_max - new_min)>0.0f0 "minimum of output has to be lower than maximum of output : $new_min >= $new_max"
    return ((input .- input_min) ./ (input_max - input_min)) .* (new_max - new_min) .+
           new_min
end

function minmaxnorm(
        input::Reactant.TracedRArray{T, 2}, input_min, input_max, new_min = 0.0f0, new_max = 1.0f0) where {T}
    return ((input .- input_min) ./ (input_max - input_min)) .* (new_max - new_min) .+
           new_min
end
