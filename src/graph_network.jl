# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
# Copyright (c) 2023 Julian Trommer
# Copyright (c) 2026 Josef Jouaux
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

import DataFrames: DataFrame
import JLD2: load, save
import Statistics: mean
import Setfield: @set!

include("feature_graph.jl")
include("graph_net_blocks.jl")

"""
    GraphNetwork(train_state, e_norm, n_norm, o_norm)

The central data structure that contains the neural network and the normalisers corresponding to the components of the GNN (edge features, node features and output).

## Arguments
- `train_state`: Lux training state containing the model, parameters, states, and optimiser.
- `e_norm`: Normaliser for the edge features of the GNN.
- `n_norm`: Normaliser for the node features of the GNN, whereas each feature has its own normaliser.
- `o_norm`: Normaliser for the output of the GNN, whereas each quantity of interest has its own normaliser.
"""
mutable struct GraphNetwork
    train_state::Lux.Training.TrainState
    e_norm::Union{NormaliserOffline, NormaliserOnline}
    n_norm::Dict{String, Union{NormaliserOffline, NormaliserOnline}}
    o_norm::Dict{String, Union{NormaliserOffline, NormaliserOnline}}
    training::Bool
end

function GraphNetwork(train_state::Lux.Training.TrainState,
        e_norm::Union{NormaliserOffline, NormaliserOnline},
        n_norm::Dict{String, Union{NormaliserOffline, NormaliserOnline}},
        o_norm::Dict{String, Union{NormaliserOffline, NormaliserOnline}})
    return GraphNetwork(train_state, e_norm, n_norm, o_norm, true)
end

"""
    build_mlp(input_size, latent_size, output_size, hidden_layers; layer_norm = true)

Constructs a MLP with the given parameters.

## Arguments
- `input_size`: Number of inputs of the MLP.
- `latent_size`: Dimension of the latent space and hidden layers.
- `output_size`: Number of outputs of the MLP.
- `hidden_layers`: Number of hidden layers.

## Keyword Arguments
- `layer_norm = true`: Wether a layer norm should be appended at the end of the MLP.

## Returns
- MLP constructed as a [Lux.jl](https://github.com/LuxDL/Lux.jl) Chain.
"""
function build_mlp(input_size::T, latent_size::T, output_size::T,
        hidden_layers::T; layer_norm = true) where {T <: Integer}
    mlp = Lux.Chain(Lux.Dense(input_size, latent_size, relu),
        collect(Lux.Dense(latent_size, latent_size, relu) for _ in 1:hidden_layers),
        Lux.Dense(latent_size, output_size))
    if layer_norm
        mlp = Lux.Chain(mlp.layers..., Lux.LayerNorm((output_size,)))
        # mlp = Lux.Chain(mlp.layers..., Lux.LayerNorm((output_size,); dims = 1))
    end
    return mlp
end

"""
    build_model(quantities_size, dims, output_size, mps, layer_size, hidden_layers)

Constructs the Encode-Process-Decode model as a [Lux.jl](https://github.com/LuxDL/Lux.jl) Chain with the given arguments.

## Arguments
- `quantities_size`: Sum of dimensions of each node feature.
- `dims`: Dimension of the mesh.
- `output_size`: Sum of dimensions of output quantities.
- `mps`: Number of message passing steps.
- `layer_size`: Size of hidden layers.
- `hidden_layers`: Number of hidden layers.

## Returns
- Encode-Process-Decode model as a [Lux.jl](https://github.com/LuxDL/Lux.jl) Chain.
"""
function build_model(quantities_size::Integer, dims, output_size::Integer,
        mps::Integer, layer_size::Integer, hidden_layers::Integer)
    encoder = Encoder(
        build_mlp(quantities_size, layer_size, layer_size, hidden_layers),
        build_mlp(dims + 1, layer_size, layer_size, hidden_layers))

    processors = Vector{Processor}()
    for _ in 1:mps
        push!(processors,
            Processor(
                build_mlp(2 * layer_size, layer_size, layer_size, hidden_layers),
                build_mlp(3 * layer_size, layer_size, layer_size, hidden_layers)))
    end

    decoder = Decoder(build_mlp(
        layer_size, layer_size, output_size, hidden_layers; layer_norm = false))

    model = Lux.Chain(encoder, processors..., decoder)

    return model
end

"""
    loss(ps, gn, graph, target, mask, loss_function)

Calculates the loss of the network based on the given loss function.

## Arguments
- `ps`: Parameters of the network.
- `gn`: [`GraphNetwork`](@ref) that contains the network.
- `graph`: [`FeatureGraph`](@ref) that contains the edge and node features.
- `target`: Ground truth data that is used for comparison to the network output.
- `mask`: Mask that filters which node types contribute to the loss.
- `loss_function`: Function used for calculating the loss.

## Returns
- Calculated Loss.
"""
function loss(ps, gn::GraphNetwork, graph::FeatureGraph, target::AbstractArray{Float32, 2},
        mask::AbstractArray{T, 1}, val_mask::AbstractArray{Float32, 2}) where {T <: Integer}
    output,
    _ = gn.train_state.model(
        graph, ps, gn.train_state.states)

    error = sum(abs2, (target - output) .* val_mask; dims = 1)[1, :]

    loss = mean(error[mask])

    return loss
end

"""
    step!(gn, graph, target_quantities_change, mask, loss_function)

## Arguments
- `gn`: [`GraphNetwork`](@ref) that is used.
- `graph`: Input data stored in a [`FeatureGraph`](@ref).
- `target_quantities_change`: Derivatives of quantities of interest (e.g. via finite differences from data).
- `mask`: Mask for excluding node types that should not be updated.
- `loss_function`: Loss function that is used to calculate the error.

## Returns
- Calculated gradients.
- Calculated training loss.
"""
function step!(gn, graph, target_quantities_change, mask, val_mask)
    train_loss,
    back = Zygote.pullback(
        ps -> loss(ps, gn, graph, target_quantities_change, mask, val_mask),
        gn.train_state.parameters)
    gs = back(one(train_loss))

    return gs, train_loss
end

"""
    set_training!(gn, training)

Enables or disables statistics accumulation for all online normalisers. Repeated calls with
the same mode are idempotent.
"""
function set_training!(gn::GraphNetwork, training::Bool)
    gn.training == training && return nothing

    direction = training ? -1 : 1
    if gn.e_norm isa NormaliserOnline
        gn.e_norm.num_accumulations += direction * gn.e_norm.max_accumulations
    end
    for nn in values(gn.n_norm)
        if nn isa NormaliserOnline
            nn.num_accumulations += direction * nn.max_accumulations
        end
    end
    for on in values(gn.o_norm)
        if on isa NormaliserOnline
            on.num_accumulations += direction * on.max_accumulations
        end
    end

    gn.training = training
    return nothing
end

"""
    save!(gn, opt_state, df_train, df_valid, step, train_loss, path; is_training = true)

Creates a checkpoint of the [`GraphNetwork`](@ref) at the given training step.

## Arguments
- `gn`: [`GraphNetwork`](@ref) that a checkpoint is created of.
- `opt_state`: State of the optimiser.
- `df_train`: [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) DataFrame that stores the train losses at the checkpoints.
- `df_valid`: [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) DataFrame that stores the validation losses at the checkpoints (only improvements are saved).
- `step`: Current training step where the checkpoint is created.
- `train_loss`: Current training loss.
- `path`: Path to the folder where checkpoints are saved.

## Keyword Arguments
- `is_training = true`: True if used in training, false otherwise (in validation).
"""
function save_checkpoint!(
        gn::GraphNetwork, opt_state, df_train::DataFrame, df_valid::DataFrame,
        step::Integer, path::String)
    ps_data = cpu_device()(getdata(gn.train_state.parameters))
    ps_axes = cpu_device()(getaxes(gn.train_state.parameters))
    st = cpu_device()(gn.train_state.states)

    save(joinpath(path, "checkpoint_$step.jld2"),
        Dict("ps_data" => ps_data, "ps_axes" => ps_axes,
            "st" => st, "e_norm" => serialize(gn.e_norm),
            "n_norm" => serialize(gn.n_norm), "o_norm" => serialize(gn.o_norm),
            "opt_state" => cpu_device()(opt_state), "df_train" => df_train,
            "df_valid" => df_valid, "train_state_step" => gn.train_state.step,
            "training" => gn.training))

    if isfile(joinpath(path, "checkpoints"))
        cps = readlines(joinpath(path, "checkpoints"))
    else
        cps = Vector{String}()
    end
    push!(cps, string(step))
    if length(cps) > 5
        rm(joinpath(path, "checkpoint_$(cps[1]).jld2"))
        deleteat!(cps, 1)
    end
    open(joinpath(path, "checkpoints"), "w") do f
        for cp in cps
            write(f, cp * "\n")
        end
    end
end

"""
    load(quantities, dims, norms, output, message_steps, ls, hl, opt, device, path)

Loads the [`GraphNetwork`](@ref) from the latest checkpoint at the given path.

## Arguments
- `quantities`: Sum of dimensions of each node feature.
- `dims`: Dimension of the mesh.
- `e_norms`: Normalisers for edge features.
- `n_norms`: Normalisers for node features.
- `o_norms`: Normalisers for output features.
- `output`: Sum of dimensions of output quantities.
- `message_steps`: Number of message passing steps.
- `ls`: Size of hidden layers.
- `hl`: Number of hidden layers.
- `opt`: Optimiser that is used for training. Set this to `nothing` if you want to use the optimiser from the checkpoint.
- `device`: Device where the model should be loaded (see [Lux GPU Management](https://lux.csail.mit.edu/dev/manual/gpu_management#gpu-management)).
- `path`: Path to the folder where the checkpoint is.

## Returns
- [`GraphNetwork`](@ref) that is loaded from the checkpoint.
- Loaded Optimiser state. Is nothing if no checkpoint was found or an optimiser was passed as an argument.
- [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) DataFrame containing the train losses at the checkpoints.
- [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) DataFrame containing the validation losses at the checkpoints (only improvements are saved).
"""
function load_checkpoint(
        quantities, dims, e_norms::Union{NormaliserOffline, NormaliserOnline},
        n_norms::Dict{String, Union{NormaliserOffline, NormaliserOnline}},
        o_norms::Dict{String, Union{NormaliserOffline, NormaliserOnline}},
        output, message_steps, ls, hl, opt, device::Function, path::String)
    if isfile(joinpath(path, "checkpoints"))
        step = parse(Int, readlines(joinpath(path, "checkpoints"))[end])
        checkpoint = load(joinpath(path, "checkpoint_$step.jld2"))
        ps_data = checkpoint["ps_data"]
        ps_axes = checkpoint["ps_axes"]
        st = checkpoint["st"]
        e_norm = checkpoint["e_norm"]
        n_norm = checkpoint["n_norm"]
        o_norm = checkpoint["o_norm"]
        opt_state = checkpoint["opt_state"]
        df_train = checkpoint["df_train"]
        df_valid = checkpoint["df_valid"]
        train_state_step = get(checkpoint, "train_state_step", 0)
        training = get(checkpoint, "training", true)
        ps = ComponentArray(ps_data, ps_axes)
        model = build_model(quantities, dims, output, message_steps, ls, hl)

        en = deserialize(e_norm, device)
        nn = deserialize(n_norm, device)
        on = deserialize(o_norm, device)

        ps = ps |> device
        st = st |> device

        train_state = Lux.Training.TrainState(model, ps, st, opt)
        @set! train_state.optimizer_state = device(opt_state)
        @set! train_state.step = train_state_step

        gn = GraphNetwork(train_state, en, nn, on, training)

        return gn, df_train, df_valid
    else
        model = build_model(quantities, dims, output, message_steps, ls, hl)
        ps, st = Lux.setup(Random.default_rng(), model)

        ps = ComponentArray(ps) |> device
        st = st |> device

        train_state = Lux.Training.TrainState(model, ps, st, opt)

        gn = GraphNetwork(train_state, e_norms, n_norms, o_norms)

        df_train = DataFrame(; step = Integer[], loss = Float32[])
        df_valid = DataFrame(; step = Integer[], loss = Float32[])

        return gn, df_train, df_valid
    end
end
