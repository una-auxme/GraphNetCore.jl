#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

"""
    FeatureGraph(nf, ef, senders, receivers)

Data structure that is used as an input for the [`GraphNetwork`](@ref).

## Arguments
- `nf`: Node features of the graph.
- `ef`: edge features of the graph.
- `senders`: List of nodes in the mesh where graph edges start.
- `receivers`: List of nodes in the mesh where graph edges end.
"""
mutable struct FeatureGraph{F <: AbstractArray, T <: AbstractArray}
    nf::F
    ef::F
    senders::T
    receivers::T
end

"""
    update_features!(g; nf, ef)

Updates the node and edge features of the given [`FeatureGraph`](@ref).

## Arguments
- `g`: [`FeatureGraph`](@ref) that should be updated.

## Keyword Arguments
- `nf`: Updated node features.
- `ef`: Updated edge features.

## Returns
- Updated graph as a [`FeatureGraph`](@ref) struct.
"""
function update_features!(g::FeatureGraph; nf, ef)
    g.nf = nf
    g.ef = ef

    return g
end

"""
    aggregate_edge_features(graph)

Aggregates the edge features based on the senders and receivers of the given [`FeatureGraph`](@ref).

## Arguments
- `graph`: [`FeatureGraph`](@ref) which node and edge features are used.

## Returns
- Two-dimensional array with the
    - 1. dimension containing the concatenated features as new edge features and the
    - 2. dimension representing the individual edges.
"""
@inline function aggregate_edge_features(graph::FeatureGraph)
    return vcat(graph.nf[:, graph.senders], graph.nf[:, graph.receivers], graph.ef)
end

"""
    aggregate_node_features(graph, updated_edge_features)

Aggregates the node features based on the given [`FeatureGraph`](@ref) and updated edge features.

## Arguments
- `graph`: [`FeatureGraph`](@ref) which node features are used.
- `updated_edge_features`: New edge features that were calculated in a previous step.

## Returns
- Two dimensional array with the
    - 1. dimension containing the concatenated features as new node features and the
    - 2. dimension representing the individual nodes.
"""
@inline function aggregate_node_features(graph::FeatureGraph, updated_edge_features)
    return vcat(graph.nf,
        NNlib.scatter(+, updated_edge_features, graph.receivers; dstsize = size(graph.nf)))
end
