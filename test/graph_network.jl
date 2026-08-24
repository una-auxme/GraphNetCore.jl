#
# Copyright (c) 2023 Julian Trommer
# Copyright (c) 2026 Josef Jouaux
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

struct TestGraphScaleLayer <: Lux.AbstractLuxLayer end

function (::TestGraphScaleLayer)(graph::FeatureGraph, ps, st)
    return ps.weight .* graph.nf, st
end
@testset "graph_network.jl" begin
    nf = [0.5f0 -0.25f0 -0.75f0
          -2.4f0 3.6f0 1.2f0]
    ef = [3.0f0 0.0f0 -3.0f0 0.0f0 -4.0f0 -4.0f0
          0.0f0 4.0f0 4.0f0 -3.0f0 0.0f0 3.0f0
          3.0f0 4.0f0 5.0f0 3.0f0 4.0f0 5.0f0]
    senders = [2, 3, 3, 1, 1, 2]
    receivers = [1, 1, 2, 2, 3, 3]
    output = [3.0774236f0 -0.37687588f0 0.2191811f0;
              -2.30065f0 -2.6680458f0 -1.881568f0]

    graph = FeatureGraph(nf, ef, senders, receivers)

    model = build_model(2, 2, 2, 1, 16, 1)
    ps, st = Lux.setup(MersenneTwister(1234), model)

    out, _ = model(graph, ps, st)

    @test out ≈ output

    mlp_with_norm = GraphNetCore.build_mlp(2, 4, 3, 1)
    mlp_without_norm = GraphNetCore.build_mlp(2, 4, 3, 1; layer_norm = false)
    @test last(Tuple(mlp_with_norm.layers)) isa Lux.LayerNorm
    @test !(last(Tuple(mlp_without_norm.layers)) isa Lux.LayerNorm)

    multi_step_model = build_model(2, 2, 2, 3, 8, 1)
    multi_step_layers = Tuple(multi_step_model.layers)
    @test length(multi_step_layers) == 5
    @test multi_step_layers[1] isa GraphNetCore.Encoder
    @test all(layer -> layer isa GraphNetCore.Processor,
        multi_step_layers[2:4])
    @test multi_step_layers[5] isa GraphNetCore.Decoder
    @test last(Tuple(multi_step_layers[1].node_layer.layers)) isa Lux.LayerNorm
    @test last(Tuple(multi_step_layers[1].edge_layer.layers)) isa Lux.LayerNorm
    @test all(
        processor -> last(Tuple(processor.node_layer.layers)) isa Lux.LayerNorm &&
                     last(Tuple(processor.edge_layer.layers)) isa Lux.LayerNorm,
        multi_step_layers[2:4])
    @test !(last(Tuple(multi_step_layers[5].decode_layer.layers)) isa Lux.LayerNorm)
    multi_ps, multi_st = Lux.setup(MersenneTwister(4321), multi_step_model)
    multi_out, _ = multi_step_model(graph, multi_ps, multi_st)
    @test size(multi_out) == (2, 3)

    norm_off = NormaliserOfflineMinMax(-1.0f0, 1.0f0, 0.0f0, 1.0f0)
    train_state = Lux.Training.TrainState(
        model, ps, st, Lux.Optimisers.Descent(1.0f-3))
    gn = GraphNetwork(train_state, norm_off,
        Dict{String, Union{NormaliserOffline, NormaliserOnline}}(),
        Dict{String, Union{NormaliserOffline, NormaliserOnline}}())

    target_delta = Float32[1 100 2; 3 200 4]
    target = out .+ target_delta
    node_mask = [1, 3]
    value_mask = Float32[1 1 0; 0 1 1]

    @test GraphNetCore.loss(ps, gn, graph, target, node_mask, value_mask) ≈ 8.5f0
    boolean_node_mask = Bool[true, false, true]
    @test GraphNetCore.loss(
        ps, gn, graph, target, boolean_node_mask, value_mask) ≈ 8.5f0
    @test GraphNetCore.loss(ps, gn, graph, target, node_mask,
        zeros(Float32, size(value_mask))) == 0.0f0

    gradients, train_loss = step!(gn, graph, target, node_mask, value_mask)
    @test train_loss ≈ 8.5f0
    @test length(gradients) == 1
    @test only(gradients) !== nothing
    expected_gradients = Zygote.gradient(
        parameters -> GraphNetCore.loss(
            parameters, gn, graph, target, node_mask, value_mask),
        ps)
    gradient_values = ComponentArray(only(gradients))
    @test gradient_values ≈ ComponentArray(only(expected_gradients))
    @test all(isfinite, gradient_values)
    @test any(!iszero, gradient_values)

    zero_mask_gradients,
    zero_mask_loss = step!(gn, graph, target, node_mask,
        zeros(Float32, size(value_mask)))
    @test zero_mask_loss == 0.0f0
    @test all(iszero, ComponentArray(only(zero_mask_gradients)))

    analytic_model = TestGraphScaleLayer()
    analytic_ps = (; weight = reshape(Float32[2, -1], 2, 1))
    analytic_train_state = Lux.Training.TrainState(
        analytic_model, analytic_ps, (;), Lux.Optimisers.Descent(1.0f-3))
    analytic_gn = GraphNetwork(analytic_train_state, norm_off,
        Dict{String, Union{NormaliserOffline, NormaliserOnline}}(),
        Dict{String, Union{NormaliserOffline, NormaliserOnline}}())
    analytic_graph = FeatureGraph(
        Float32[1 2 3; 4 5 6], zeros(Float32, 0, 0), Int[], Int[])
    analytic_gradients,
    analytic_loss = step!(analytic_gn, analytic_graph,
        zeros(Float32, 2, 3), [1, 3], ones(Float32, 2, 3))
    @test analytic_loss == 46.0f0
    @test only(analytic_gradients).weight ≈ reshape(Float32[20, -52], 2, 1)

    edge_online = NormaliserOnline(Float32, 1, cpu; max_acc = 10.0f0)
    node_online = NormaliserOnline(Float32, 1, cpu; max_acc = 20.0f0)
    output_online = NormaliserOnline(Float32, 1, cpu; max_acc = 30.0f0)
    edge_online.num_accumulations = 2.0f0
    node_online.num_accumulations = 3.0f0
    output_online.num_accumulations = 4.0f0
    gn.e_norm = edge_online
    gn.n_norm["online"] = node_online
    gn.n_norm["offline"] = norm_off
    gn.o_norm["online"] = output_online

    set_training!(gn, false)
    @test !gn.training
    @test edge_online.num_accumulations == 12.0f0
    @test node_online.num_accumulations == 23.0f0
    @test output_online.num_accumulations == 34.0f0
    set_training!(gn, false)
    @test edge_online.num_accumulations == 12.0f0
    @test node_online.num_accumulations == 23.0f0
    @test output_online.num_accumulations == 34.0f0
    set_training!(gn, true)
    @test gn.training
    @test edge_online.num_accumulations == 2.0f0
    @test node_online.num_accumulations == 3.0f0
    @test output_online.num_accumulations == 4.0f0
    set_training!(gn, true)
    @test edge_online.num_accumulations == 2.0f0
    @test node_online.num_accumulations == 3.0f0
    @test output_online.num_accumulations == 4.0f0

    if hascuda
        gpu_graph = FeatureGraph(
            gpu(nf), gpu(ef), gpu(senders), gpu(receivers))
        gpu_ps = gpu(ps)
        gpu_st = gpu(st)
        gpu_out, _ = model(gpu_graph, gpu_ps, gpu_st)
        @test gpu_out ≈ gpu(output)

        gpu_train_state = Lux.Training.TrainState(
            model, gpu_ps, gpu_st, Lux.Optimisers.Descent(1.0f-3))
        gpu_gn = GraphNetwork(gpu_train_state, norm_off,
            Dict{String, Union{NormaliserOffline, NormaliserOnline}}(),
            Dict{String, Union{NormaliserOffline, NormaliserOnline}}())
        gpu_target = gpu_out .+ gpu(target_delta)
        gpu_node_mask = gpu(Int32.(node_mask))
        gpu_value_mask = gpu(value_mask)

        @test GraphNetCore.loss(gpu_ps, gpu_gn, gpu_graph,
            gpu_target, gpu_node_mask, gpu_value_mask) ≈ 8.5f0
        gpu_gradients,
        gpu_train_loss = step!(gpu_gn, gpu_graph,
            gpu_target, gpu_node_mask, gpu_value_mask)
        gpu_gradient_values = ComponentArray(cpu(only(gpu_gradients)))
        @test gpu_train_loss ≈ train_loss
        @test all(isfinite, gpu_gradient_values)
        @test any(!iszero, gpu_gradient_values)
        @test isapprox(gpu_gradient_values, gradient_values;
            rtol = 1.0f-4, atol = 1.0f-5)
    end

    @testset "checkpoint round trip" begin
        mktempdir() do checkpoint_dir
            edge_norm = NormaliserOfflineMinMax(-1.0f0, 1.0f0, 0.0f0, 1.0f0)
            node_norms = Dict{String, Union{NormaliserOffline, NormaliserOnline}}(
                "node" => NormaliserOfflineMeanStd(Float32[0], Float32[1], cpu))
            output_norms = Dict{String, Union{NormaliserOffline, NormaliserOnline}}(
                "output" => NormaliserOfflineMeanStd(Float32[0], Float32[1], cpu))
            optimizer = Lux.Optimisers.Adam(1.0f-3)

            gn, df_train,
            df_valid = GraphNetCore.load_checkpoint(1, 2, edge_norm, node_norms,
                output_norms, 1, 1, 4, 0, optimizer, cpu, checkpoint_dir)
            @test isempty(df_train)
            @test isempty(df_valid)

            zero_gradients = zero(gn.train_state.parameters)
            gn.train_state = Lux.Training.apply_gradients(
                gn.train_state, zero_gradients)
            saved_optimizer_state = gn.train_state.optimizer_state
            set_training!(gn, false)

            for checkpoint_step in 1:7
                push!(df_train,
                    (; step = checkpoint_step, loss = Float32(checkpoint_step)))
                if isodd(checkpoint_step)
                    push!(df_valid,
                        (; step = checkpoint_step,
                            loss = Float32(checkpoint_step) / 10.0f0))
                end
                save_checkpoint!(gn, gn.train_state.optimizer_state,
                    df_train, df_valid, checkpoint_step, checkpoint_dir)
            end

            @test readlines(joinpath(checkpoint_dir, "checkpoints")) ==
                  string.(3:7)
            @test !isfile(joinpath(checkpoint_dir, "checkpoint_1.jld2"))
            @test !isfile(joinpath(checkpoint_dir, "checkpoint_2.jld2"))
            @test all(
                step -> isfile(
                    joinpath(checkpoint_dir, "checkpoint_$step.jld2")), 3:7)

            loaded_gn, loaded_train,
            loaded_valid = GraphNetCore.load_checkpoint(1, 2, edge_norm,
                node_norms, output_norms, 1, 1, 4, 0, optimizer, cpu, checkpoint_dir)
            @test loaded_gn.train_state.parameters ≈ gn.train_state.parameters
            @test loaded_gn.train_state.states == gn.train_state.states
            @test loaded_gn.train_state.optimizer_state == saved_optimizer_state
            @test loaded_gn.train_state.step == gn.train_state.step
            @test !loaded_gn.training
            @test loaded_gn.e_norm.data_min == gn.e_norm.data_min
            @test loaded_gn.n_norm["node"].data_mean == gn.n_norm["node"].data_mean
            @test loaded_gn.o_norm["output"].data_std == gn.o_norm["output"].data_std
            @test loaded_train == df_train
            @test loaded_valid == df_valid
            @test loaded_train.step[end] == 7
            @test loaded_valid.step == [1, 3, 5, 7]

            resumed_train_state = Lux.Training.apply_gradients(
                loaded_gn.train_state, zero(loaded_gn.train_state.parameters))
            @test resumed_train_state.step == gn.train_state.step + 1
        end

        if hascuda
            @testset "GPU checkpoint round trip" begin
                mktempdir() do checkpoint_dir
                    edge_norm = NormaliserOfflineMinMax(
                        Float32[-1], Float32[1], gpu)
                    node_norms = Dict{String, Union{
                        NormaliserOffline, NormaliserOnline}}(
                        "node" => NormaliserOfflineMeanStd(
                        Float32[0], Float32[1], gpu))
                    output_norms = Dict{String, Union{
                        NormaliserOffline, NormaliserOnline}}(
                        "output" => NormaliserOfflineMeanStd(
                        Float32[0], Float32[1], gpu))
                    optimizer = Lux.Optimisers.Adam(1.0f-3)

                    gpu_checkpoint_gn, df_train,
                    df_valid = GraphNetCore.load_checkpoint(1, 2, edge_norm,
                        node_norms, output_norms, 1, 1, 4, 0,
                        optimizer, gpu, checkpoint_dir)
                    gpu_checkpoint_gn.train_state = Lux.Training.apply_gradients(
                        gpu_checkpoint_gn.train_state,
                        zero(gpu_checkpoint_gn.train_state.parameters))
                    push!(df_train, (; step = 1, loss = 1.0f0))
                    save_checkpoint!(gpu_checkpoint_gn,
                        gpu_checkpoint_gn.train_state.optimizer_state,
                        df_train, df_valid, 1, checkpoint_dir)

                    loaded_gpu_gn, loaded_train,
                    loaded_valid = GraphNetCore.load_checkpoint(1, 2, edge_norm,
                        node_norms, output_norms, 1, 1, 4, 0,
                        optimizer, gpu, checkpoint_dir)
                    @test parent(loaded_gpu_gn.train_state.parameters) isa CUDA.CuArray
                    @test cpu(loaded_gpu_gn.train_state.parameters) ≈
                          cpu(gpu_checkpoint_gn.train_state.parameters)
                    @test cpu(loaded_gpu_gn.train_state.optimizer_state) ==
                          cpu(gpu_checkpoint_gn.train_state.optimizer_state)
                    @test loaded_gpu_gn.train_state.step ==
                          gpu_checkpoint_gn.train_state.step
                    @test loaded_train == df_train
                    @test loaded_valid == df_valid
                end
            end
        end
    end
end
