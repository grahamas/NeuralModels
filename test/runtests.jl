push!(LOAD_PATH, "@stdlib")
using Pkg
Pkg.activate(@__DIR__)

using Test
using Simulation73
using NeuralModels


n_points_dx1 = 101
extent_dx1 = 100.0
mid_point_dx1 = floor(Int, n_points_dx1 / 2) + (n_points_dx1 % 2)
circle_dx1 = PeriodicLattice{Float64,1}(; n_points=n_points_dx1, extent=extent_dx1)
line_dx1 = CompactLattice{Float64,1}(; n_points=n_points_dx1, extent=extent_dx1)
empty_circle_dx1 = zeros(circle_dx1)
empty_line_dx1 = zeros(line_dx1)


@testset "Stimulus" begin
    @testset "Non-stimulus" begin
        nostim = NoStimulusParameter{Float64,1}()
        nostim_action = nostim(circle_dx1)
        empty_circle_nostim_test = copy(empty_circle_dx1)
        nostim_action(empty_circle_nostim_test)
        @test all(empty_circle_nostim_test .== empty_circle_dx1)
    end
    @testset "Sharp Bump" begin
        whole_space_stim_param = SharpBumpStimulusParameter(; center=(0.0,), strength=10.0, width=101.0, time_windows=[(0.0,45.0)])
        whole_space_stim = whole_space_stim_param(line_dx1)
        whole_line_bump = copy(empty_line_dx1)
        whole_space_stim(whole_line_bump, whole_line_bump, 0.0)
        @test all(whole_line_bump .== (empty_line_dx1 .+ 10.0))

        manual_sharp_bump_width_dx1 = 20.0
        half_width_dx1 = round(Int,manual_sharp_bump_width_dx1 / 2)
        manual_sharp_bump_dx1 = copy(empty_line_dx1)
        manual_sharp_bump_dx1[mid_point_dx1-half_width_dx1:mid_point_dx1+half_width_dx1] .= 1.0

        sharp_bump_stim_param = SharpBumpStimulusParameter(; center=(0.0,), strength=1.0, width=manual_sharp_bump_width_dx1, time_windows=[(0.0,45.0)])
        sharp_bump_stim = sharp_bump_stim_param(line_dx1)
        sharp_bump_test = copy(empty_line_dx1)
        sharp_bump_stim(sharp_bump_test, sharp_bump_test, 0.0)
        @show sharp_bump_test |> sum
        @show manual_sharp_bump_dx1 |> sum
        @test all(sharp_bump_test .== manual_sharp_bump_dx1)
        sharp_bump_stim(sharp_bump_test, sharp_bump_test, 55.0)
        @test all(sharp_bump_test .== manual_sharp_bump_dx1)
    end
    @testset "Array" begin

    end
end

@testset "Connectivity" begin
    @testset "Exponentially decaying connectivity" begin
        conn_param = ExpSumSqDecayingConnectivityParameter(amplitude=1.0, spread=(20.0,))
        conn_action = conn_param(line_dx1)
        @testset "FFT" begin
            fft_conn_param = FFTParameter(conn_param)
            fft_conn_action = fft_conn_param(line_dx1)

            fft_square_bump = copy(empty_line_dx1)
            naive_square_bump = copy(empty_line_dx1)
            fft_square_bump[mid_point_dx1-10:mid_point_dx1+10] .= 1.0
            naive_square_bump[mid_point_dx1-10:mid_point_dx1+10] .= 1.0

            fft_conn_action(fft_square_bump, fft_square_bump, 0.0)
            conn_action(naive_square_bump, naive_square_bump, 0.0)
            @show fft_square_bump
            @show naive_square_bump
            @show fft_square_bump .- naive_square_bump
            @test all(fft_square_bump .≈ naive_square_bump)

        end
    end
end
