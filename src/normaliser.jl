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

abstract type NormaliserOffline end

"""
    NormaliserOfflineMinMax(data_min, data_max, target_min = 0.0f0, target_max = 0.0f0)

Offline normalization if the minimum and maximum of the quantity is known (e.g. from the training data).
It is recommended to use offline normalization since the minimum and maximum do not need to be inferred from data.

## Arguments
- `data_min`: Minimum of the quantity in the dataset.
- `data_max`: Maximum of the quantity in the dataset.
- `target_min`: Minimum of the target of normalization.
- `target_max`: Maximum of the target of normalization.
"""
mutable struct NormaliserOfflineMinMax{AT, T} <: NormaliserOffline
    data_min::AT
    data_max::AT
    target_min::T
    target_max::T
end
function NormaliserOfflineMinMax(
        data_min::AT, data_max::AT, device::Function) where {AT}
    NormaliserOfflineMinMax(device(data_min), device(data_max), 0.0f0, 1.0f0)
end

function NormaliserOfflineMinMax(
        data_min::AT, data_max::AT, target_min::T, target_max::T, device::Function) where {
        AT, T}
    NormaliserOfflineMinMax(device(data_min), device(data_max), target_min, target_max)
end

function NormaliserOfflineMinMax(d::Dict{String, Any}, device::Function)
    NormaliserOfflineMinMax(
        device(d["data_min"]), device(d["data_max"]), d["target_min"], d["target_max"])
end

function (n::NormaliserOfflineMinMax)(F)
    minmaxnorm(F, n.data_min, n.data_max, n.target_min, n.target_max)
end

"""
    inverse_data(n, data)

Inverses the normalised data.

## Arguments
- `n`: Used [`NormaliserOfflineMinMax`](@ref).
- `data`: Data to be converted back.

## Returns
- Converted data.
"""
function inverse_data(n::NormaliserOfflineMinMax, data)
    # Since the minmax of the output is not known, we let the decoder handle it
    return data
    # return minmaxnorm(data, n.target_min, n.target_max, n.data_min, n.data_max)
end

"""
    NormaliserOfflineMeanStd(data_mean, data_std)

Offline normalization if the mean and standard deviation of the quantity is known (e.g. from the training data).
It is recommended to use offline normalization since the minimum and maximum do not need to be inferred from data.

## Arguments
- `data_mean`: Mean of the quantity in the dataset.
- `data_std`: Standard deviation of the quantity in the dataset.
"""
mutable struct NormaliserOfflineMeanStd{AT, T} <: NormaliserOffline
    data_mean::AT
    data_std::AT
    std_epsilon::T
end

function NormaliserOfflineMeanStd(
        data_mean::T, data_std::T, device::Function) where {T}
    NormaliserOfflineMeanStd(device(data_mean), device(data_std), eps(eltype(T)))
end

function NormaliserOfflineMeanStd(d::Dict{String, Any}, device::Function)
    NormaliserOfflineMeanStd(
        device(d["data_mean"]), device(d["data_std"]),
        haskey(d, "std_epsilon") ? d["std_epsilon"] : eps(eltype(d["data_mean"])))
end

function (n::NormaliserOfflineMeanStd)(F)
    (F .- n.data_mean) ./ max.(n.data_std, n.std_epsilon)
end

"""
    inverse_data(n, data)

Inverses the normalised data.

## Arguments
- `n`: Used [`NormaliserOfflineMeanStd`](@ref).
- `data`: Data to be converted back.

## Returns
- Converted data.
"""
function inverse_data(n::NormaliserOfflineMeanStd, data)
    return data .* max.(n.data_std, n.std_epsilon) .+ n.data_mean
end

"""
    NormaliserOnline(max_accumulations, std_epsilon, acc_count, num_accumulations, acc_sum, acc_sum_squared)

Online normalization if the minimum and maximum of the quantity is not known.
It is recommended to use offline normalization since the minimum and maximum do not need to be inferred from data.

## Arguments
- `max_accumulations`: Maximum number of accumulation steps.
- `std_epsilon`: Epsilon for caluclating the standard deviation.
- `acc_count`: Sum of dimensions of quantities in each accumulation step.
- `num_accumulations`: Current number of accumulation steps.
- `acc_sum`: Sum of quantities in each step.
- `acc_sum_squared`: Sum of quantities squared in each step.
"""
mutable struct NormaliserOnline{T, AT}
    max_accumulations::T
    std_epsilon::T
    acc_count::T
    num_accumulations::T
    acc_sum::AT
    acc_sum_squared::AT
end

"""
    NormaliserOnline(dim, device; max_acc = 10f6, std_ep = 1f-8)

Online normalization if the minimum and maximum of the quantity is not known.
It is recommended to use offline normalization since the minimum and maximum do not need to be inferred from data.

## Arguments
- `dims`: Dimension of the quantity to normalize.
- `device`: Device where the normaliser should be loaded (see [Lux GPU Management](https://lux.csail.mit.edu/dev/manual/gpu_management#gpu-management)).

## Keyword Arguments
- `max_acc = 10f6`: Maximum number of accumulation steps.
- `std_epsilon = 1f-8`: Epsilon for caluclating the standard deviation.
"""
function NormaliserOnline(::Type{T}, dim::Integer, device::Function;
        max_acc::T = 10.0f6, std_ep::T = eps(T)) where {T}
    if device == reactant_device()
        NormaliserOnline(Reactant.to_rarray(max_acc; track_numbers = true),
            Reactant.to_rarray(std_ep; track_numbers = true),
            Reactant.to_rarray(zero(T); track_numbers = true),
            Reactant.to_rarray(zero(T); track_numbers = true),
            device(zeros(T, dim)), device(zeros(T, dim)))
    else
        NormaliserOnline(
            max_acc, std_ep, zero(T), zero(T), device(zeros(T, dim)), device(zeros(T, dim)))
    end
end

"""
    NormaliserOnline(d, device)

Online normalization if the minimum and maximum of the quantity is not known.
It is recommended to use offline normalization since the minimum and maximum do not need to be inferred from data.

## Arguments
- `d`: Dictionary containing the fields of the struct [`NormaliserOnline`](@ref).
- `device`: Device where the normaliser should be loaded (see [Lux GPU Management](https://lux.csail.mit.edu/dev/manual/gpu_management#gpu-management)).
"""
function NormaliserOnline(d::Dict{String, Any}, device::Function)
    if device == reactant_device()
        NormaliserOnline(Reactant.to_rarray(d["max_accumulations"]; track_numbers = true),
            Reactant.to_rarray(d["std_epsilon"]; track_numbers = true),
            Reactant.to_rarray(d["acc_count"]; track_numbers = true),
            Reactant.to_rarray(d["num_accumulations"]; track_numbers = true),
            device(d["acc_sum"]), device(d["acc_sum_squared"]))
    else
        NormaliserOnline(d["max_accumulations"], d["std_epsilon"],
            d["acc_count"], d["num_accumulations"],
            device(d["acc_sum"]), device(d["acc_sum_squared"]))
    end
end

function (n::NormaliserOnline)(F::AbstractArray)
    @trace if n.num_accumulations < n.max_accumulations
        accumulate_stats!(n, F)
    end

    return (F .- get_mean(n)) ./ get_std_with_epsilon(n)
end

"""
    inverse_data(n, data)

Inverses the normalised data.

## Arguments
- `n`: Used [`NormaliserOnline`](@ref).
- `data`: Data to be converted back.

## Returns
- Converted data.
"""
function inverse_data(n::NormaliserOnline, data)
    return data .* get_std_with_epsilon(n) .+ get_mean(n)
end

function accumulate_stats!(n::NormaliserOnline, F)
    n.acc_count += size(F)[2]
    n.acc_sum += tullio_reducesum(F, 2)
    n.acc_sum_squared += tullio_reducesum(F .^ 2, 2)
    n.num_accumulations += 1.0f0
end

function accumulate_stats!(n::NormaliserOnline, F::Reactant.TracedRArray)
    n.acc_count += size(F)[2]
    n.acc_sum += reduce(+, F; dims = 2)[:, 1]
    n.acc_sum_squared += reduce(+, F .^ 2; dims = 2)[:, 1]
    n.num_accumulations += 1.0f0
end

# function accumulate_stats!(n::NormaliserOnline, F::Reactant.TracedRArray)
#     @set n.acc_count = n.acc_count + size(F)[2]
#     @set n.acc_sum = n.acc_sum + reduce(+, F; dims = 2)[:, 1]
#     @set n.acc_sum_squared = n.acc_sum_squared + reduce(+, F .^ 2; dims = 2)[:, 1]
#     @set n.num_accumulations = n.num_accumulations + 1.0f0
# end

function get_mean(n::NormaliserOnline)
    safe_count = max(n.acc_count, 1.0f0)

    return n.acc_sum / safe_count
end

function get_std_with_epsilon(n::NormaliserOnline)
    safe_count = max(n.acc_count, 1.0f0)
    std = get_sqrt.(n.acc_sum_squared / safe_count - get_mean(n) .^ 2)

    return max.(std, n.std_epsilon)
end

function get_sqrt(n)
    return sqrt(max(0.0f0, n))
end

function serialize(ns::Dict{String, Union{NormaliserOffline, NormaliserOnline}})
    result = Dict{String, Any}()
    for (k, n) in ns
        result[k] = serialize(n)
    end

    return result
end

function serialize(n::NormaliserOnline)
    return Dict{String, Any}(
        "max_accumulations" => cpu_device()(n.max_accumulations),
        "std_epsilon" => cpu_device()(n.std_epsilon),
        "acc_count" => cpu_device()(n.acc_count),
        "num_accumulations" => cpu_device()(n.num_accumulations),
        "acc_sum" => cpu_device()(n.acc_sum),
        "acc_sum_squared" => cpu_device()(n.acc_sum_squared)
    )
end

function serialize(n::NormaliserOfflineMinMax)
    return Dict{String, Any}(
        "data_min" => cpu_device()(n.data_min),
        "data_max" => cpu_device()(n.data_max),
        "target_min" => n.target_min,
        "target_max" => n.target_max
    )
end

function serialize(n::NormaliserOfflineMeanStd)
    return Dict{String, Any}(
        "data_mean" => cpu_device()(n.data_mean),
        "data_std" => cpu_device()(n.data_std)
    )
end

function deserialize(n::Dict{String, Any}, device::Function)
    if haskey(n, "max_accumulations")
        return NormaliserOnline(n, device)
    elseif haskey(n, "data_min")
        return NormaliserOfflineMinMax(n, device)
    elseif haskey(n, "data_mean")
        return NormaliserOfflineMeanStd(n, device)
    else
        features = keys(n)
        norms = deserialize.(values(n), device)
        return Dict{String, Union{NormaliserOffline, NormaliserOnline}}(features .=> norms)
    end
end
