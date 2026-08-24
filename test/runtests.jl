#
# Copyright (c) 2023 Julian Trommer
# Copyright (c) 2026 Josef Jouaux
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using GraphNetCore
using Test
using Aqua

using CUDA
using ComponentArrays
using Lux
using Zygote

import Random: MersenneTwister

const hascuda = CUDA.has_cuda()
const cpu = cpu_device()
const gpu = hascuda ? gpu_device() : nothing

hascuda || @warn "No CUDA installation detected! Skipping GPU tests..."

@testset "GraphNetCore.jl" begin
    include("aqua.jl")
    include("utils.jl")
    include("normaliser.jl")
    include("feature_graph.jl")
    include("graph_net_blocks.jl")
    include("graph_network.jl")
end
