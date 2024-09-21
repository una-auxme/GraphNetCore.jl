#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module GraphNetCore

using CUDA, cuDNN
using Lux
using Tullio
using Random

const NAME_TYPE = Union{Nothing, String, Symbol}

include("utils.jl")
include("normaliser.jl")
include("graph_network.jl")

# feature_graph.jl
export FeatureGraph
# graph_network.jl
export GraphNetwork
# normaliser.jl
export NormaliserOffline, NormaliserOfflineMinMax, NormaliserOfflineMeanStd,
       NormaliserOnline

# graph_network.jl
export build_model, step!, save!, load
# normaliser.jl
export inverse_data
# utils.jl
export triangles_to_edges, parse_edges, one_hot, minmaxnorm, mse_reduce, inverse_data

end
