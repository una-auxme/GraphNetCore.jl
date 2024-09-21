#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

struct Encoder{T <: NamedTuple, N <: NAME_TYPE} <: Lux.AbstractLuxWrapperLayer{:layers}
    layers::T
    name::N
end

function Encoder(node_model, edge_model; name::NAME_TYPE = nothing)
    fields = (Symbol("node_model_fn"), Symbol("edge_model_fn"))

    return Encoder(NamedTuple{fields}((node_model, edge_model)), name)
end

function (e::Encoder)(graph::FeatureGraph, ps, st::NamedTuple{fields}) where {fields}
    encode!(e.layers, graph, ps, st)
end

function encode!(
        layers::NamedTuple{fields}, graph, ps, st::NamedTuple{fields}) where {fields}
    nf, stn = layers[:node_model_fn](graph.nf, ps[:node_model_fn], st[:node_model_fn])
    ef, ste = layers[:edge_model_fn](graph.ef, ps[:edge_model_fn], st[:edge_model_fn])
    new_st = NamedTuple{fields}((stn, ste))

    return update_features!(graph; nf = nf, ef = ef), new_st
end

struct Processor{T <: NamedTuple, N <: NAME_TYPE} <:
       Lux.AbstractLuxWrapperLayer{:layers}
    layers::T
    name::N
end

function Processor(node_model, edge_model; name::NAME_TYPE = nothing)
    fields = (Symbol("node_model_fn"), Symbol("edge_model_fn"))

    return Processor(NamedTuple{fields}((node_model, edge_model)), name)
end

function (p::Processor)(graph::FeatureGraph, ps, st::NamedTuple{fields}) where {fields}
    process!(p.layers, graph, ps, st)
end

function process!(layers::NamedTuple{fields}, graph::FeatureGraph,
        ps, st::NamedTuple{fields}) where {fields}
    uef, ste = update_edge_features(
        layers[:edge_model_fn], ps[:edge_model_fn], st[:edge_model_fn], graph)
    unf, stn = update_node_features(
        layers[:node_model_fn], ps[:node_model_fn], st[:node_model_fn], graph, uef)
    new_st = NamedTuple{fields}((stn, ste))

    return update_features!(graph; nf = graph.nf + unf, ef = graph.ef + uef), new_st
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

struct Decoder{T <: NamedTuple, N <: NAME_TYPE} <: Lux.AbstractLuxWrapperLayer{:layers}
    layers::T
    name::N
end

function Decoder(model; name::NAME_TYPE = nothing)
    fields = (Symbol("model"),)

    return Decoder(NamedTuple{fields}((model,)), name)
end

function (d::Decoder)(graph::FeatureGraph, ps, st::NamedTuple{fields}) where {fields}
    decode!(d.layers, graph, ps, st)
end

function decode!(layers::NamedTuple{fields}, graph::FeatureGraph,
        ps, st::NamedTuple{fields}) where {fields}
    y, stm = layers[:model](graph.nf, ps[:model], st[:model])
    new_st = NamedTuple{fields}((stm,))

    return y, new_st
end
