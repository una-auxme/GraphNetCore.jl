#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

#######
# Lux #
#######

struct EncoderLux{N, E} <: Lux.AbstractLuxContainerLayer{(:node_layer, :edge_layer)}
    node_layer::N
    edge_layer::E
end

function (e::EncoderLux)(graph::FeatureGraph, ps, st)
    nf, stn = e.node_layer(graph.nf, ps.node_layer, st.node_layer)
    ef, ste = e.edge_layer(graph.ef, ps.edge_layer, st.edge_layer)
    return FeatureGraph(graph; nf = nf, ef = ef), (; node_layer = stn, edge_layer = ste)
end

struct ProcessorLux{N, E} <: Lux.AbstractLuxContainerLayer{(:node_layer, :edge_layer)}
    node_layer::N
    edge_layer::E
end

function (p::ProcessorLux)(graph::FeatureGraph, ps, st)
    uef, ste = p.edge_layer(aggregate_edge_features(graph), ps.edge_layer, st.edge_layer)
    unf, stn = p.node_layer(
        aggregate_node_features(graph, uef), ps.node_layer, st.node_layer)
    return FeatureGraph(graph; nf = graph.nf + unf, ef = graph.ef + uef),
    (; node_layer = ste, edge_layer = stn)
end

struct DecoderLux{D} <: Lux.AbstractLuxContainerLayer{(:decode_layer,)}
    decode_layer::D
end

function (d::DecoderLux)(graph::FeatureGraph, ps, st)
    df, std = d.decode_layer(graph.nf, ps, st)
    return df, std
end

########
# Flux #
########

struct EncoderFlux{N <: Flux.Chain, E <: Flux.Chain}
    node_layer::N
    edge_layer::E
end

Flux.@layer EncoderFlux

function (e::EncoderFlux)(graph::FeatureGraph)
    nf = e.node_layer(graph.nf)
    ef = e.edge_layer(graph.ef)
    return FeatureGraph(graph; nf = nf, ef = ef)
end

struct ProcessorFlux{N <: Flux.Chain, E <: Flux.Chain}
    node_layer::N
    edge_layer::E
end

Flux.@layer ProcessorFlux

function (p::ProcessorFlux)(graph::FeatureGraph)
    uef = p.edge_layer(aggregate_edge_features(graph))
    unf = p.node_layer(aggregate_node_features(graph, uef))
    return FeatureGraph(graph; nf = graph.nf + unf, ef = graph.ef + uef)
end

struct DecoderFlux{D <: Flux.Chain}
    decode_layer::D
end

Flux.@layer DecoderFlux

function (d::DecoderFlux)(graph::FeatureGraph)
    return d.decode_layer(graph.nf)
end

function Lux.convert_flux_model(l::EncoderFlux; kwargs...)
    return EncoderLux(Lux.convert_flux_model(l.node_layer; kwargs...),
        Lux.convert_flux_model(l.edge_layer; kwargs...))
end

function Lux.convert_flux_model(l::ProcessorFlux; kwargs...)
    return ProcessorLux(Lux.convert_flux_model(l.node_layer; kwargs...),
        Lux.convert_flux_model(l.edge_layer; kwargs...))
end

function Lux.convert_flux_model(l::DecoderFlux; kwargs...)
    return DecoderLux(Lux.convert_flux_model(l.decode_layer; kwargs...))
end

function Lux.convert_flux_model(l::Flux.LayerNorm; kwargs...)
    return Lux.LayerNorm(l.size; dims = 1)
end

function luxparams_to_fluxstate(ps)
    nt = []
    for key in keys(ps)
        if hasproperty(ps[key], :weight)
            push!(nt, (weight = ps[key].weight, bias = ps[key].bias, σ = ()))
        elseif hasproperty(ps[key], :scale)
            push!(nt,
                (λ = (),
                    diag = (scale = ps[key].scale[:, 1], bias = ps[key].bias[:, 1], σ = ()),
                    ϵ = 1.0f-5,
                    size = (size(ps[key].scale, 1),),
                    affine = true))
        elseif hasproperty(ps[key], :node_layer)
            push!(nt,
                (node_layer = luxparams_to_fluxstate(ps[key].node_layer),
                    edge_layer = luxparams_to_fluxstate(ps[key].edge_layer)))
        elseif hasproperty(ps[key], :decode_layer)
            push!(nt, (decode_layer = luxparams_to_fluxstate(ps[key].decode_layer),))
        else
            push!(nt, luxparams_to_fluxstate(ps[key]))
        end
    end
    return (layers = Tuple(nt),)
end
