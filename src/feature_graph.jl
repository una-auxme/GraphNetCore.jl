#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

"""
    FeatureGraph(nf, ef, senders, receivers)

Data structure that is used as an input for the [GraphNetCore.GraphNetwork](@ref).

# Arguments
- `nf`: Node features of the graph.
- `ef`: edge features of the graph.
- `senders`: List of nodes in the mesh where graph edges start.
- `receivers`: List of nodes in the mesh where graph edges end.
"""
mutable struct FeatureGraph
    nf
    ef
    senders
    receivers
end

function update_features!(g::FeatureGraph; nf, ef)
    g.nf = nf
    g.ef = ef
    return g
end

@inline function aggregate_edge_features(graph::FeatureGraph)
    return vcat(graph.nf[:, graph.senders], graph.nf[:, graph.receivers], graph.ef)
end

@inline function aggregate_node_features(graph::FeatureGraph, updated_edge_features)
    return vcat(graph.nf, NNlib.scatter(+, updated_edge_features, graph.receivers, dstsize = size(graph.nf)))
end
