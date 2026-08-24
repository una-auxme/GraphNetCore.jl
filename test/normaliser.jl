#
# Copyright (c) 2023 Julian Trommer
# Copyright (c) 2026 Josef Jouaux
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

@testset "normaliser.jl" begin
    norm_off = NormaliserOfflineMinMax(-10.0f0, 10.0f0, -1.0f0, 1.0f0)
    norm_on_cpu = NormaliserOnline(
        Dict{String, Any}("max_accumulations" => 10000.0f0, "std_epsilon" => 1.0f-8,
            "acc_count" => 2000.0f0, "num_accumulations" => 200.0f0,
            "acc_sum" => [142.32f0, 63.24f0],
            "acc_sum_squared" => [20254.9824f0, 3999.2976f0]), cpu)
    norm_dict_cpu = Dict{String, Union{NormaliserOffline, NormaliserOnline}}(
        "norm_off" => norm_off, "norm_on" => norm_on_cpu)

    # inverse_data for NormaliserOfflineMinMax intentionally returns the input
    # unchanged (the decoder handles the output range).
    @test inverse_data(norm_off, [0.0f0]) == [0.0f0]
    @test inverse_data(norm_off, [-0.5f0, -0.25f0, 0.1f0, 0.75f0]) ==
          [-0.5f0, -0.25f0, 0.1f0, 0.75f0]
    hascuda && @test inverse_data(norm_off, gpu([-0.5f0, -0.25f0, 0.1f0, 0.75f0])) ==
          gpu([-0.5f0, -0.25f0, 0.1f0, 0.75f0])

    norm_off_default = NormaliserOfflineMinMax(
        Float32[-2, 0], Float32[2, 4], cpu)
    @test norm_off_default(Float32[-2 0; 2 4]) == Float32[0 0.5; 0.5 1]

    norm_mean_std = NormaliserOfflineMeanStd(Float32[2, -1], Float32[2, 4], cpu)
    mean_std_input = Float32[4, 3]
    @test norm_mean_std(mean_std_input) == Float32[1, 1]
    @test inverse_data(norm_mean_std, norm_mean_std(mean_std_input)) == mean_std_input

    norm_mean_std_epsilon = NormaliserOfflineMeanStd(Float32[2], Float32[0], 0.5f0)
    @test norm_mean_std_epsilon(Float32[3]) == Float32[2]
    @test inverse_data(norm_mean_std_epsilon, Float32[2]) == Float32[3]

    mean_std_data = GraphNetCore.serialize(norm_mean_std_epsilon)
    @test mean_std_data["std_epsilon"] == 0.5f0
    norm_mean_std_roundtrip = GraphNetCore.deserialize(mean_std_data, cpu)
    @test norm_mean_std_roundtrip.data_mean == norm_mean_std_epsilon.data_mean
    @test norm_mean_std_roundtrip.data_std == norm_mean_std_epsilon.data_std
    @test norm_mean_std_roundtrip.std_epsilon == norm_mean_std_epsilon.std_epsilon

    online = NormaliserOnline(Float32, 2, cpu; max_acc = 1.0f0, std_ep = 1.0f-6)
    online_input = Float32[1 3; 2 4]
    normalised = online(online_input)
    @test normalised ≈ Float32[-1 1; -1 1]
    @test inverse_data(online, normalised) ≈ online_input
    @test online.acc_count == 2.0f0
    @test online.num_accumulations == 1.0f0
    @test online.acc_sum == Float32[4, 6]
    @test online.acc_sum_squared == Float32[10, 20]

    accumulated_state = (online.acc_count, online.num_accumulations,
        copy(online.acc_sum), copy(online.acc_sum_squared))
    online(Float32[100 200; 300 400])
    @test (online.acc_count, online.num_accumulations,
        online.acc_sum, online.acc_sum_squared) == accumulated_state

    differentiable_online = NormaliserOnline(
        Float32, 2, cpu; max_acc = 1.0f0, std_ep = 1.0f-6)
    input_gradient = only(Zygote.gradient(
        x -> sum(differentiable_online(x)), online_input))
    @test input_gradient ≈ ones(Float32, size(online_input))
    @test differentiable_online.num_accumulations == 1.0f0

    constant_online = NormaliserOnline(
        Float32, 2, cpu; max_acc = 1.0f0, std_ep = 0.25f0)
    @test GraphNetCore.get_mean(constant_online) == zeros(Float32, 2)
    @test GraphNetCore.get_std_with_epsilon(constant_online) == fill(0.25f0, 2)
    constant_input = Float32[2 2; -1 -1]
    constant_normalised = constant_online(constant_input)
    @test constant_normalised == zeros(Float32, size(constant_input))
    @test GraphNetCore.get_std_with_epsilon(constant_online) == fill(0.25f0, 2)
    @test inverse_data(constant_online, constant_normalised) == constant_input

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
        online_gpu = NormaliserOnline(
            Float32, 2, gpu; max_acc = 1.0f0, std_ep = 1.0f-6)
        online_input_gpu = gpu(online_input)
        normalised_gpu = online_gpu(online_input_gpu)
        @test normalised_gpu ≈ gpu(Float32[-1 1; -1 1])
        @test inverse_data(online_gpu, normalised_gpu) ≈ online_input_gpu
        @test online_gpu.acc_count == 2.0f0
        @test online_gpu.num_accumulations == 1.0f0
        @test online_gpu.acc_sum == gpu(Float32[4, 6])
        @test online_gpu.acc_sum_squared == gpu(Float32[10, 20])

        norm_on_gpu = NormaliserOnline(
            Dict{String, Any}("max_accumulations" => 10000.0f0,
                "std_epsilon" => 1.0f-8, "acc_count" => 2000.0f0,
                "num_accumulations" => 200.0f0,
                "acc_sum" => gpu([142.32f0, 63.24f0]),
                "acc_sum_squared" => gpu([20254.9824f0, 3999.2976f0])), gpu)
        norm_dict_gpu = Dict{String, Union{NormaliserOffline, NormaliserOnline}}(
            "norm_off" => norm_off, "norm_on" => norm_on_gpu)
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
