#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using GraphNetCore
using Test
using Aqua

using CUDA, cuDNN, Lux

import Random: MersenneTwister

@testset "GraphNetCore.jl" begin
    hascuda = CUDA.has_cuda()

    if hascuda
        gpu = gpu_device()
    else
        @warn "No CUDA installation detected! Skipping GPU tests..."
    end

    cpu = cpu_device()

    @testset "Aqua.jl" begin
        # TODO remove ambiguity in load
        # Ambiguities in external packages
        @testset "Method ambiguity" begin
            Aqua.test_ambiguities([GraphNetCore]; broken = true)
        end
        Aqua.test_all(GraphNetCore; ambiguities = false)
    end

    @testset "utils.jl" begin
        #   3 - 4
        #  / \ /
        # 1 - 2
        faces = [1 2
                 2 4
                 3 3]
        @test triangles_to_edges(faces) == ([2, 4, 3, 4, 3, 1, 2, 2, 3, 1],
            [1, 2, 2, 3, 1, 2, 4, 3, 4, 3])

        edges = [1 1 2 2 4
                 2 3 4 3 3]
        @test parse_edges(edges) ==
              ([2, 3, 4, 3, 4, 1, 1, 2, 2, 3], [1, 1, 2, 2, 3, 2, 3, 4, 3, 4])

        indices = [1, 3, 2, 4]
        @test one_hot(indices, 5, 0) == Bool[1 0 0 0
                                             0 0 1 0
                                             0 1 0 0
                                             0 0 0 1
                                             0 0 0 0]
        @test one_hot(indices, 5, -1) == Bool[0 0 1 0
                                              0 1 0 0
                                              0 0 0 1
                                              0 0 0 0
                                              0 0 0 0]
        @test one_hot(indices, 3, 0) == Bool[1 0 0 0
                                             0 0 1 0
                                             0 1 0 0]

        @test minmaxnorm([2.0f0], 1.0f0, 1.0f0) == [0.0f0]
        hascuda &&
            @test minmaxnorm(gpu([1.0f0, 2.0f0]), 1.0f0, 1.0f0) == gpu([0.0f0, 0.0f0])
        @test minmaxnorm([1.4f0, 2.3f0, 3.9f0, 4.0f0], -4.0f0, 4.0f0, 0.0f0, 1.0f0) ==
              [0.675f0, 0.7875f0, 0.9875f0, 1.0f0]
        @test_throws AssertionError minmaxnorm([2.0f0], 1.5f0, 0.5f0)
        @test_throws AssertionError minmaxnorm([2.0f0], 1.0f0, 1.0f0, 1.5f0, 0.5f0)

        mse_reduce_target = [2.0f0 1.3f0
                             0.6f0 1.7f0]
        mse_reduce_output = [2.0f0 1.5f0
                             0.2f0 1.8f0]
        @test mse_reduce(mse_reduce_target, mse_reduce_output) ≈ [0.16f0, 0.05f0]
        hascuda && @test mse_reduce(gpu(mse_reduce_target), gpu(mse_reduce_output)) ≈
              gpu([0.16f0, 0.05f0])

        reduce_arr = [1.0f0 2.0f0
                      3.0f0 4.0f0]
        @test GraphNetCore.tullio_reducesum(reduce_arr, 1) == [4.0f0 6.0f0]
        @test GraphNetCore.tullio_reducesum(reduce_arr, 2) == [3.0f0, 7.0f0]
        hascuda &&
            @test GraphNetCore.tullio_reducesum(gpu(reduce_arr), 1) == gpu([4.0f0 6.0f0])
        hascuda &&
            @test GraphNetCore.tullio_reducesum(gpu(reduce_arr), 2) == gpu([3.0f0, 7.0f0])
    end

    @testset "normaliser.jl" begin
        norm_off = NormaliserOfflineMinMax(-10.0f0, 10.0f0, -1.0f0, 1.0f0)
        norm_on_cpu = NormaliserOnline(
            Dict{String, Any}("max_accumulations" => 10000.0f0, "std_epsilon" => 1.0f-8,
                "acc_count" => 2000.0f0, "num_accumulations" => 200.0f0,
                "acc_sum" => [142.32f0, 63.24f0],
                "acc_sum_squared" => [20254.9824f0, 3999.2976f0]), cpu)
        norm_on_gpu = NormaliserOnline(
            Dict{String, Any}("max_accumulations" => 10000.0f0, "std_epsilon" => 1.0f-8,
                "acc_count" => 2000.0f0, "num_accumulations" => 200.0f0,
                "acc_sum" => gpu([142.32f0, 63.24f0]),
                "acc_sum_squared" => gpu([20254.9824f0, 3999.2976f0])), gpu)
        norm_dict_cpu = Dict{String, Union{NormaliserOffline, NormaliserOnline}}(
            "norm_off" => norm_off, "norm_on" => norm_on_cpu)
        norm_dict_gpu = Dict{String, Union{NormaliserOffline, NormaliserOnline}}(
            "norm_off" => norm_off, "norm_on" => norm_on_gpu)

        @test inverse_data(norm_off, [0.0f0]) == [0.0f0]
        @test inverse_data(norm_off, [-0.5f0, -0.25f0, 0.1f0, 0.75f0]) ==
              [-5.0f0, -2.5f0, 1.0f0, 7.5f0]
        hascuda && @test inverse_data(norm_off, gpu([-0.5f0, -0.25f0, 0.1f0, 0.75f0])) ==
              gpu([-5.0f0, -2.5f0, 1.0f0, 7.5f0])

        norm_dict_cpu_test = GraphNetCore.deserialize(
            GraphNetCore.serialize(norm_dict_cpu), cpu)
        @test norm_dict_cpu["norm_off"].data_min ==
              norm_dict_cpu_test["norm_off"].data_min &&
              norm_dict_cpu["norm_off"].data_max ==
              norm_dict_cpu_test["norm_off"].data_max &&
              norm_dict_cpu["norm_off"].target_min ==
              norm_dict_cpu_test["norm_off"].target_min &&
              norm_dict_cpu["norm_off"].target_max ==
              norm_dict_cpu_test["norm_off"].target_max
        @test norm_dict_cpu["norm_on"].max_accumulations ==
              norm_dict_cpu_test["norm_on"].max_accumulations &&
              norm_dict_cpu["norm_on"].std_epsilon ==
              norm_dict_cpu_test["norm_on"].std_epsilon &&
              norm_dict_cpu["norm_on"].acc_count ==
              norm_dict_cpu_test["norm_on"].acc_count &&
              norm_dict_cpu["norm_on"].num_accumulations ==
              norm_dict_cpu_test["norm_on"].num_accumulations &&
              norm_dict_cpu["norm_on"].acc_sum == norm_dict_cpu_test["norm_on"].acc_sum &&
              norm_dict_cpu["norm_on"].acc_sum_squared ==
              norm_dict_cpu_test["norm_on"].acc_sum_squared

        if hascuda
            norm_dict_gpu_test = GraphNetCore.deserialize(
                GraphNetCore.serialize(norm_dict_gpu), gpu)
            @test norm_dict_gpu["norm_off"].data_min ==
                  norm_dict_gpu_test["norm_off"].data_min &&
                  norm_dict_gpu["norm_off"].data_max ==
                  norm_dict_gpu_test["norm_off"].data_max &&
                  norm_dict_gpu["norm_off"].target_min ==
                  norm_dict_gpu_test["norm_off"].target_min &&
                  norm_dict_gpu["norm_off"].target_max ==
                  norm_dict_gpu_test["norm_off"].target_max
            @test norm_dict_gpu["norm_on"].max_accumulations ==
                  norm_dict_gpu_test["norm_on"].max_accumulations &&
                  norm_dict_gpu["norm_on"].std_epsilon ==
                  norm_dict_gpu_test["norm_on"].std_epsilon &&
                  norm_dict_gpu["norm_on"].acc_count ==
                  norm_dict_gpu_test["norm_on"].acc_count &&
                  norm_dict_gpu["norm_on"].num_accumulations ==
                  norm_dict_gpu_test["norm_on"].num_accumulations &&
                  norm_dict_gpu["norm_on"].acc_sum ==
                  norm_dict_gpu_test["norm_on"].acc_sum &&
                  norm_dict_gpu["norm_on"].acc_sum_squared ==
                  norm_dict_gpu_test["norm_on"].acc_sum_squared
        end
    end

    @testset "GraphNetwork" begin
        nf = [0.5f0 -0.25f0 -0.75f0
              -2.4f0 3.6f0 1.2f0]
        ef = [3.0f0 0.0f0 -3.0f0 0.0f0 -4.0f0 -4.0f0
              0.0f0 4.0f0 4.0f0 -3.0f0 0.0f0 3.0f0
              3.0f0 4.0f0 5.0f0 3.0f0 4.0f0 5.0f0]
        senders = [2, 3, 3, 1, 1, 2]
        receivers = [1, 1, 2, 2, 3, 3]
        output = [0.7324772f0 -0.027799817f0 0.1475548f0;
                  0.42122957f0 -0.6571782f0 -0.15739384f0]

        graph = FeatureGraph(nf, ef, senders, receivers)

        model = build_model(2, 2, 2, 1, 16, 1)
        ps, st = Lux.setup(MersenneTwister(1234), model)

        out, _ = model(graph, ps, st)

        @test out ≈ output

        if hascuda
            graph = FeatureGraph(gpu(nf), gpu(ef), gpu(senders), gpu(receivers))

            ps = gpu(ps)
            st = gpu(st)

            out, _ = model(graph, ps, st)

            @test out ≈ gpu(output)
        end
    end
end
