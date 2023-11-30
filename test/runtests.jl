#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using GraphNetCore
using Test

using CUDA, cuDNN

@testset "GraphNetCore.jl" begin

    hascuda = CUDA.has_cuda()

    !hascuda && @warn "No CUDA installation detected! Skipping GPU tests..."

    @testset "utils.jl" begin
        #   3 - 4
        #  / \ /
        # 1 - 2
        faces = [1 2
                 2 4
                 3 3]
        @test triangles_to_edges(faces) == ([2, 4, 3, 4, 3, 1, 2, 2, 3, 1],
                                            [1, 2, 2, 3, 1, 2, 4, 3, 4, 3])

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

        @test minmaxnorm([2.0], 1.0, 1.0) == [0.0]
        hascuda && @test minmaxnorm(cu([1.0, 2.0]), 1.0, 1.0) == cu([0.0, 0.0])
        @test minmaxnorm([1.4, 2.3, 3.9, 4.0], -4.0, 4.0, 0.0, 1.0) == [0.675, 0.7875, 0.9875, 1.0]
        
        mse_reduce_target = [2.0 1.3
                             0.6 1.7]
        mse_reduce_output = [2.0 1.5
                             0.2 1.8]
        @test mse_reduce(mse_reduce_target, mse_reduce_output) ≈ [0.16, 0.05]
        hascuda && @test mse_reduce(cu(mse_reduce_target), cu(mse_reduce_output)) ≈ cu([0.16, 0.05])

        reduce_arr = [1.0 2.0
                      3.0 4.0]
        @test GraphNetCore.tullio_reducesum(reduce_arr, 1) == [4.0 6.0]
        @test GraphNetCore.tullio_reducesum(reduce_arr, 2) == [3.0, 7.0]
    end

end
