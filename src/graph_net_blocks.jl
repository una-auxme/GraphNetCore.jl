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
    uef, ste = p.edge_layer(aggregate_edge_features(graph), ps.edge_layer, st.edge_layer)
    unf,
    stn = p.node_layer(
        aggregate_node_features(graph, uef), ps.node_layer, st.node_layer)
    return FeatureGraph(graph; nf = graph.nf + unf, ef = graph.ef + uef),
    (; node_layer = ste, edge_layer = stn)
end

struct Decoder{D} <: Lux.AbstractLuxWrapperLayer{:decode_layer}
    decode_layer::D
end

function (d::Decoder)(graph::FeatureGraph, ps, st)
    df, std = d.decode_layer(graph.nf, ps, st)
    return df, std
end
