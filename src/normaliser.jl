#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

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
mutable struct NormaliserOfflineMinMax <: NormaliserOffline
    data_min::Float32
    data_max::Float32
    target_min::Float32
    target_max::Float32
end

function NormaliserOfflineMinMax(data_min::Float32, data_max::Float32)
    NormaliserOfflineMinMax(data_min, data_max, 0.0f0, 1.0f0)
end

function NormaliserOfflineMinMax(d::Dict{String, Any})
    NormaliserOfflineMinMax(d["data_min"], d["data_max"], d["target_min"], d["target_max"])
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
    return minmaxnorm(data, n.target_min, n.target_max, n.data_min, n.data_max)
end

"""
    NormaliserOfflineMeanStd(data_mean, data_std)

Offline normalization if the mean and standard deviation of the quantity is known (e.g. from the training data).
It is recommended to use offline normalization since the minimum and maximum do not need to be inferred from data.

## Arguments
- `data_mean`: Mean of the quantity in the dataset.
- `data_std`: Standard deviation of the quantity in the dataset.
"""
mutable struct NormaliserOfflineMeanStd <: NormaliserOffline
    data_mean::Float32
    data_std::Float32
    std_epsilon::Float32
end

function NormaliserOfflineMeanStd(data_mean::Float32, data_std::Float32)
    NormaliserOfflineMeanStd(data_mean, data_std, 1.0f-8)
end

function NormaliserOfflineMeanStd(d::Dict{String, Any})
    NormaliserOfflineMeanStd(
        d["data_mean"], d["data_std"], haskey(d, "std_epsilon") ? d["std_epsilon"] : 1.0f-8)
end

(n::NormaliserOfflineMeanStd)(F) = (F .- n.data_mean) ./ max(n.data_std, n.std_epsilon)

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
    return data .* max(n.data_std, n.std_epsilon) .+ n.data_mean
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
mutable struct NormaliserOnline{T <: AbstractArray{Float32}}
    max_accumulations::Float32
    std_epsilon::Float32
    acc_count::Float32
    num_accumulations::Float32
    acc_sum::T
    acc_sum_squared::T
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
function NormaliserOnline(
        dim::Integer, device::Function; max_acc::Float32 = 10.0f6, std_ep::Float32 = 1.0f-8)
    NormaliserOnline(max_acc, std_ep, 0.0f0, 0.0f0,
        device(zeros(Float32, dim)), device(zeros(Float32, dim)))
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
    NormaliserOnline(d["max_accumulations"], d["std_epsilon"], d["acc_count"],
        d["num_accumulations"], device(d["acc_sum"]), device(d["acc_sum_squared"]))
end

function (n::NormaliserOnline)(F, acc = true::Bool)
    if acc
        if n.num_accumulations < n.max_accumulations
            accumulate_stats!(n, F)
        end
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
    if n < 0.0f0
        return convert(typeof(n), -Inf32)
    else
        return sqrt(n)
    end
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
        "max_accumulations" => n.max_accumulations,
        "std_epsilon" => n.std_epsilon,
        "acc_count" => n.acc_count,
        "num_accumulations" => n.num_accumulations,
        "acc_sum" => cpu_device()(n.acc_sum),
        "acc_sum_squared" => cpu_device()(n.acc_sum_squared)
    )
end

function serialize(n::NormaliserOfflineMinMax)
    return Dict{String, Any}(
        "data_min" => n.data_min,
        "data_max" => n.data_max,
        "target_min" => n.target_min,
        "target_max" => n.target_max
    )
end

function serialize(n::NormaliserOfflineMeanStd)
    return Dict{String, Any}(
        "data_mean" => n.data_mean,
        "data_std" => n.data_std
    )
end

function deserialize(n::Dict{String, Any}, device::Function)
    if haskey(n, "max_accumulations")
        return NormaliserOnline(n, device)
    elseif haskey(n, "data_min")
        return NormaliserOfflineMinMax(n)
    elseif haskey(n, "data_mean")
        return NormaliserOfflineMeanStd(n)
    else
        features = keys(n)
        norms = deserialize.(values(n), device)
        return Dict{String, Union{NormaliserOffline, NormaliserOnline}}(features .=> norms)
    end
end
