#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

struct Encoder{N, E} <: Lux.AbstractLuxContainerLayer{(:node_layer, :edge_layer)}
    node_layer::N
    edge_layer::E
end

function (e::Encoder)(graph::FeatureGraph, ps, st)
    nf, stn = e.node_layer(graph.nf, ps.node_layer, st.node_layer)
    ef, ste = e.edge_layer(graph.ef, ps.edge_layer, st.edge_layer)
    return FeatureGraph(graph; nf = nf, ef = ef), (; node_layer = stn, edge_layer = ste)
end

struct Processor{N, E} <: Lux.AbstractLuxContainerLayer{(:node_layer, :edge_layer)}
    node_layer::N
    edge_layer::E
end

function (p::Processor)(graph::FeatureGraph, ps, st)
    uef, ste = update_edge_features(p.edge_layer, ps.edge_layer, st.edge_layer, graph)
    unf, stn = update_node_features(p.node_layer, ps.node_layer, st.node_layer, graph, uef)
    return FeatureGraph(graph; nf = graph.nf + unf, ef = graph.ef + uef),
    (; node_layer = ste, edge_layer = stn)
end

@inline function update_edge_features(el, ps, st, graph::FeatureGraph)
    features = aggregate_edge_features(graph)

    return el(features, ps, st)
end

@inline function update_node_features(
        nl, ps, st, graph::FeatureGraph, updated_edge_features)
    features = aggregate_node_features(graph, updated_edge_features)

    return nl(features, ps, st)
end

struct Decoder{D} <: Lux.AbstractLuxWrapperLayer{:decode_layer}
    decode_layer::D
end

function (d::Decoder)(graph::FeatureGraph, ps, st)
    df, std = d.decode_layer(graph.nf, ps, st)
    return df, std
end
