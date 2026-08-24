#
# Copyright (c) 2023 Julian Trommer
# Copyright (c) 2026 Josef Jouaux
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

@testset "feature_graph.jl" begin
    graph = FeatureGraph(Float32[1 2 3; 4 5 6], Float32[10 20 30],
        [1, 2, 1], [2, 3, 3])

    @test GraphNetCore.aggregate_edge_features(graph) ==
          Float32[1 2 1; 4 5 4; 2 3 3; 5 6 6; 10 20 30]

    updated_edges = Float32[1 2 3; 4 5 6]
    @test GraphNetCore.aggregate_node_features(graph, updated_edges) ==
          Float32[1 2 3; 4 5 6; 0 1 5; 0 4 11]

    updated_graph = FeatureGraph(graph; nf = 2 .* graph.nf)
    @test updated_graph.nf == 2 .* graph.nf
    @test updated_graph.ef === graph.ef
    @test updated_graph.senders === graph.senders
    @test updated_graph.receivers === graph.receivers
end
