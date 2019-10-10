push!(LOAD_PATH, "@stdlib")
using Pkg
Pkg.activate(@__DIR__)

using Test
using Simulation73
using NeuralModels
using DSP
using SpecialFunctions


n_points_dx1 = 1001
extent_dx1 = 300.0
dx = extent_dx1 / n_points_dx1
mid_point = floor(Int, n_points_dx1 / 2) + (n_points_dx1 % 2)
circle_dx1 = PeriodicLattice{Float64,1}(; n_points=(n_points_dx1,), extent=(extent_dx1,))
line_dx1 = CompactLattice{Float64,1}(; n_points=(n_points_dx1,), extent=(extent_dx1,))
empty_circle_dx1 = zeros(size(circle_dx1)...)
empty_line_dx1 = zeros(size(line_dx1)...)


@testset "Stimulus" begin
    @testset "Sharp Bump" begin
        whole_space_stim_param = SharpBumpStimulusParameter(; center=(0.0,), strength=10.0, width=extent_dx1, time_windows=[(0.0,45.0)])
        whole_space_stim = whole_space_stim_param(line_dx1)
        whole_line_bump = copy(empty_line_dx1)
        whole_space_stim(whole_line_bump, whole_line_bump, 0.0)
        @test all(whole_line_bump .== (empty_line_dx1 .+ 10.0))

        manual_sharp_bump_width = 20.0
        half_width = round(Int,manual_sharp_bump_width / dx / 2)
        manual_sharp_bump = copy(empty_line_dx1)
        manual_sharp_bump[mid_point-half_width:mid_point+half_width] .= 1.0

        sharp_bump_stim_param = SharpBumpStimulusParameter(; center=(0.0,), strength=1.0, width=manual_sharp_bump_width, time_windows=[(0.0,45.0)])
        sharp_bump_stim = sharp_bump_stim_param(line_dx1)
        sharp_bump_test = copy(empty_line_dx1)
        sharp_bump_stim(sharp_bump_test, sharp_bump_test, 0.0)
        @test all(sharp_bump_test .== manual_sharp_bump)
        sharp_bump_stim(sharp_bump_test, sharp_bump_test, 55.0)
        @test all(sharp_bump_test .== manual_sharp_bump)
    end
    @testset "Array" begin
        wide = SharpBumpStimulusParameter(; 
                      strength = 1.0,
                      width = 20.0,
                      time_windows = [(0.0, 20.0)])
        thin = SharpBumpStimulusParameter(; 
                      strength = 1.0,
                      width = 10.0,
                      time_windows = [(0.0, 30.0)])
        wide_test = copy(empty_line_dx1)
        thin_test = copy(empty_line_dx1)
        wide_stim = wide(line_dx1)
        thin_stim = wide(line_dx1)
        wide_stim(wide_test, wide_test, 1.0)
        thin_stim(thin_test, thin_test, 1.0)
        summed_test = wide_test .+ thin_test
        combined = [wide, thin]
        combined_stim = combined(line_dx1)
        combined_test_early = copy(empty_line_dx1)
        combined_test_late = copy(empty_line_dx1)
        combined_stim(combined_test_early, combined_test_early, 1.0)
        combined_stim(combined_test_late, combined_test_late, 25.0)
        @test all(combined_test_early .== summed_test)
        @test all(combined_test_late .== thin_test)
    end
end

@testset "Connectivity" begin
    manual_bump = copy(empty_circle_dx1)
    half_width = 30.0
    half_width_dx = floor(Int, half_width / dx)
    manual_bump[mid_point-half_width_dx:mid_point+half_width_dx] .= 1.0
    @testset "Exponentially decaying connectivity" begin
        σ = 20.0
        sq_conn_param = GaussianConnectivityParameter(amplitude=1.0, spread=(σ,))
        dsp_sq_output = conv(NeuralModels.kernel(sq_conn_param, circle_dx1), manual_bump)
        abs_conn_param = ExpSumAbsDecayingConnectivityParameter(amplitude=1.0, spread=(σ,))
        dsp_sq_output = conv(NeuralModels.kernel(abs_conn_param, circle_dx1), manual_bump)
        @test all(directed_weights(sq_conn_param, circle_dx1, (0.0,)) .≈ directed_weights(sq_conn_param, circle_dx1, (extent_dx1,)))
        @testset "FFT" begin
            sq_conn_action = sq_conn_param(circle_dx1)
            abs_conn_action = abs_conn_param(circle_dx1)
            naive_sq_output = zeros(size(manual_bump)...)
            sq_conn_action(naive_sq_output, manual_bump, 0.0)

            fft_sq_conn_param = FFTParameter(sq_conn_param)
            fft_sq_conn_action = fft_sq_conn_param(circle_dx1)

            fft_sq_output = zeros(size(manual_bump)...)
            fft_sq_conn_action(fft_sq_output, manual_bump, 0.0)

            # Assumes analytical form of Gaussian normalizes to 1
            theory_sq_conv(x) =  (erf((half_width-x[1])/σ) + erf((x[1]+half_width)/σ)) / (2)
            theory_sq_output = theory_sq_conv.(coordinates(circle_dx1))

            @test all(isapprox.(fft_sq_output,theory_sq_output, atol=0.01, rtol=0.1))
            @test all(isapprox.(naive_sq_output,theory_sq_output, atol=0.01, rtol=0.1))
        end
        @testset "FFT pops" begin
        end
    end
end

@testset "Nonlinearity" begin
    @testset "Sigmoid" begin
        sn = SigmoidNonlinearity(a=1.0, θ=5.0)
        test_vals = [-1.0, 0.0, 0.01, 5.0, 200.0]
        sn(test_vals)
        @test test_vals[1] .== 0.0
        @test test_vals[2] .== 0.0
        @test isapprox(test_vals[3], 0.0, atol=0.0001)
        @test isapprox(test_vals[4], 0.5, atol=0.01)
        @test isapprox(test_vals[5], 1.0, atol=0.01)
    end 
    @testset "Sech2" begin
        sn = Sech2Nonlinearity(a=1.0, θ=5.0)
        test_vals = [-100.0, 5.0, 200.0]
        sn(test_vals)
        @test isapprox(test_vals[1], 0.0, atol=0.001)
        @test isapprox(test_vals[2], 1.0, atol=0.001)
        @test isapprox(test_vals[3], 0.0, atol=0.001)
    end
    @testset "Gaussian" begin
        gn = GaussianNonlinearity(sd=1.0, θ=5.0)
        test_vals = [-100.0, 5.0, 200.0]
        gn(test_vals)
        @test isapprox(test_vals[1], 0.0, atol=0.001)
        @test isapprox(test_vals[2], 1.0, atol=0.001)
        @test isapprox(test_vals[3], 0.0, atol=0.001)
    end
end
