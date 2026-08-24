#
# Copyright (c) 2023 Julian Trommer
# Copyright (c) 2026 Josef Jouaux
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

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
    @test one_hot([0, 1, 4], 3) == Float32[0 1 0
                                           0 0 0
                                           0 0 0]

    @test_throws AssertionError minmaxnorm([2.0f0], 1.0f0, 1.0f0)
    hascuda &&
        @test_throws AssertionError minmaxnorm(gpu([1.0f0, 2.0f0]), 1.0f0, 1.0f0)
    @test minmaxnorm([1.4f0, 2.3f0, 3.9f0, 4.0f0], -4.0f0, 4.0f0, 0.0f0, 1.0f0) ==
          [0.675f0, 0.7875f0, 0.9875f0, 1.0f0]
    @test minmaxnorm(Float32[-2 0; 2 4], Float32[-2, 0], Float32[2, 4]) ==
          Float32[0 0.5; 0.5 1]
    @test_throws AssertionError minmaxnorm([2.0f0], 1.5f0, 0.5f0)
    @test_throws AssertionError minmaxnorm([2.0f0], 1.0f0, 1.0f0, 1.5f0, 0.5f0)
end
