#
# Copyright (c) 2023 Julian Trommer
# Copyright (c) 2026 Josef Jouaux
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

struct TestSliceLayer{R, M} <: Lux.AbstractLuxLayer
    rows::R
    marker::M
end
function (layer::TestSliceLayer)(x, ps, st)
    return x[layer.rows, :], (; marker = layer.marker)
end

@testset "graph_net_blocks.jl" begin
    graph = FeatureGraph(Float32[1 2 3; 4 5 6], Float32[7 8 9; 10 11 12],
        [1, 2, 1], [2, 3, 3])
    block_ps = (; node_layer = (;), edge_layer = (;))
    block_st = (; node_layer = (;), edge_layer = (;))

    encoder = GraphNetCore.Encoder(
        TestSliceLayer([2, 1], :encoded_nodes),
        TestSliceLayer([2, 1], :encoded_edges))
    encoded_graph, encoder_state = encoder(graph, block_ps, block_st)
    @test encoded_graph.nf == graph.nf[[2, 1], :]
    @test encoded_graph.ef == graph.ef[[2, 1], :]
    @test encoded_graph.senders === graph.senders
    @test encoded_graph.receivers === graph.receivers
    @test encoder_state.node_layer.marker == :encoded_nodes
    @test encoder_state.edge_layer.marker == :encoded_edges

    processor = GraphNetCore.Processor(
        TestSliceLayer(1:2, :processed_nodes),
        TestSliceLayer(5:6, :processed_edges))
    processed_graph, processor_state = processor(graph, block_ps, block_st)
    @test processed_graph.nf == 2 .* graph.nf
    @test processed_graph.ef == 2 .* graph.ef
    @test processed_graph.senders === graph.senders
    @test processed_graph.receivers === graph.receivers
    @test processor_state.node_layer.marker == :processed_nodes
    @test processor_state.edge_layer.marker == :processed_edges

    decoder = GraphNetCore.Decoder(TestSliceLayer(1:1, :decoded))
    decoded, decoder_state = decoder(graph, (;), (;))
    @test decoded == graph.nf[1:1, :]
    @test decoder_state.marker == :decoded
end
