push!(LOAD_PATH, "@stdlib")
using Pkg
Pkg.activate(@__DIR__)

using Test
using Simulation73
using NeuralModels
using DSP
using SpecialFunctions

function make_testing_lattice(; n_points=1001, extent=300.0, type=CompactLattice{Float64,1})
    dx = extent / n_points
    mid_point = floor(Int, n_points / 2) + (n_points % 2)
    lattice = type(; n_points = n_points, extent = extent)
    lattice_zeros = zeros(size(lattice)...)
    return (n_points, extent, dx, mid_point, lattice, lattice_zeros)
end


@testset "Stimulus" begin
    @testset "Sharp Bump" begin
        n_points, extent, dx, mid_point, lattice, lattice_zeros = make_testing_lattice()
        whole_lattice_stim_param = SharpBumpStimulusParameter(; center=(0.0,), strength=10.0, width=extent, time_windows=[(0.0,45.0)])
        whole_lattice_stim = whole_lattice_stim_param(lattice)
        whole_lattice_bump = copy(lattice_zeros)
        whole_lattice_stim(whole_lattice_bump, whole_lattice_bump, 0.0)
        @test all(whole_lattice_bump .== (lattice_zeros .+ 10.0))

        manual_sharp_bump_width = 20.0
        half_width = round(Int,manual_sharp_bump_width / dx / 2)
        manual_sharp_bump = copy(lattice_zeros)
        manual_sharp_bump[mid_point-half_width:mid_point+half_width] .= 1.0

        sharp_bump_stim_param = SharpBumpStimulusParameter(; center=(0.0,), strength=1.0, width=manual_sharp_bump_width, time_windows=[(0.0,45.0)])
        sharp_bump_stim = sharp_bump_stim_param(lattice)
        sharp_bump_test = copy(lattice_zeros)
        sharp_bump_stim(sharp_bump_test, sharp_bump_test, 0.0)
        @test all(sharp_bump_test .== manual_sharp_bump)
        sharp_bump_stim(sharp_bump_test, sharp_bump_test, 55.0)
        @test all(sharp_bump_test .== manual_sharp_bump)
    end
    @testset "Array" begin
        n_points, extent, dx, mid_point, lattice, lattice_zeros = make_testing_lattice(n_points=100, extent=100.0)
        wide = SharpBumpStimulusParameter(; 
                      strength = 1.0,
                      width = 20.0,
                      time_windows = [(0.0, 20.0)])
        thin = SharpBumpStimulusParameter(; 
                      strength = 1.0,
                      width = 10.0,
                      time_windows = [(0.0, 30.0)])
        wide_test = copy(lattice_zeros)
        thin_test = copy(lattice_zeros)
        wide_stim = wide(lattice)
        thin_stim = thin(lattice)
        wide_stim(wide_test, wide_test, 1.0)
        thin_stim(thin_test, thin_test, 1.0)
        summed_test = wide_test .+ thin_test
        combined = [wide, thin]
        combined_stim = combined(lattice)
        combined_test_early = copy(lattice_zeros)
        combined_test_late = copy(lattice_zeros)
        combined_stim(combined_test_early, combined_test_early, 1.0)
        combined_stim(combined_test_late, combined_test_late, 25.0)
        @show combined_test_early
        @show summed_test
        @show combined_test_late
        @show thin_test
        @test all(combined_test_early .== summed_test)
        @test all(combined_test_late .== thin_test)
    end
end

@testset "Connectivity" begin
    @testset "Odd n_points" begin
        n_points, extent, dx, mid_point, circle, circle_zeros = make_testing_lattice(type=PeriodicLattice{Float64,1})
        manual_bump = copy(circle_zeros)
        half_width = 30.0
        half_width_dx = floor(Int, half_width / dx)
        manual_bump[mid_point-half_width_dx:mid_point+half_width_dx] .= 1.0
        @testset "Exponentially decaying connectivity" begin
            σ = 20.0
            sq_conn_param = GaussianConnectivityParameter(amplitude=1.0, spread=(σ,))
            dsp_sq_output = conv(NeuralModels.kernel(sq_conn_param, circle), manual_bump)
            abs_conn_param = ExpSumAbsDecayingConnectivityParameter(amplitude=1.0, spread=(σ,))
            dsp_sq_output = conv(NeuralModels.kernel(abs_conn_param, circle), manual_bump)
            @test all(directed_weights(sq_conn_param, circle, (0.0,)) .≈ directed_weights(sq_conn_param, circle, (extent,)))
            @testset "FFT" begin
                sq_conn_action = sq_conn_param(circle)
                abs_conn_action = abs_conn_param(circle)
                naive_sq_output = zeros(size(manual_bump)...)
                sq_conn_action(naive_sq_output, manual_bump, 0.0)

                fft_sq_conn_param = FFTParameter(sq_conn_param)
                fft_sq_conn_action = fft_sq_conn_param(circle)

                fft_sq_output = zeros(size(manual_bump)...)
                fft_sq_conn_action(fft_sq_output, manual_bump, 0.0)

                # Assumes analytical form of Gaussian normalizes to 1
                theory_sq_conv(x) =  (erf((half_width-x[1])/σ) + erf((x[1]+half_width)/σ)) / (2)
                theory_sq_output = theory_sq_conv.(coordinates(circle))

                @test all(isapprox.(fft_sq_output,theory_sq_output, atol=0.01, rtol=0.1))
                @test all(isapprox.(naive_sq_output,theory_sq_output, atol=0.01, rtol=0.1))
            end
            @testset "FFT pops" begin
            end
        end
    end
    @testset "Even n_points" begin
        n_points, extent, dx, mid_point, circle, circle_zeros = make_testing_lattice(n_points=1000, type=PeriodicLattice{Float64,1})
        manual_bump = copy(circle_zeros)
        half_width = 30.0
        half_width_dx = floor(Int, half_width / dx)
        manual_bump[mid_point-half_width_dx:mid_point+half_width_dx] .= 1.0
        @testset "Exponentially decaying connectivity" begin
            σ = 20.0
            sq_conn_param = GaussianConnectivityParameter(amplitude=1.0, spread=(σ,))
            dsp_sq_output = conv(NeuralModels.kernel(sq_conn_param, circle), manual_bump)
            abs_conn_param = ExpSumAbsDecayingConnectivityParameter(amplitude=1.0, spread=(σ,))
            dsp_sq_output = conv(NeuralModels.kernel(abs_conn_param, circle), manual_bump)
            @test all(directed_weights(sq_conn_param, circle, (0.0,)) .≈ directed_weights(sq_conn_param, circle, (extent,)))
            @testset "FFT" begin
                sq_conn_action = sq_conn_param(circle)
                abs_conn_action = abs_conn_param(circle)
                naive_sq_output = zeros(size(manual_bump)...)
                sq_conn_action(naive_sq_output, manual_bump, 0.0)

                fft_sq_conn_param = FFTParameter(sq_conn_param)
                fft_sq_conn_action = fft_sq_conn_param(circle)

                fft_sq_output = zeros(size(manual_bump)...)
                fft_sq_conn_action(fft_sq_output, manual_bump, 0.0)

                # Assumes analytical form of Gaussian normalizes to 1
                theory_sq_conv(x) =  (erf((half_width-x[1])/σ) + erf((x[1]+half_width)/σ)) / (2)
                theory_sq_output = theory_sq_conv.(coordinates(circle))

                @test all(isapprox.(fft_sq_output,theory_sq_output, atol=0.01, rtol=0.1))
                @test all(isapprox.(naive_sq_output,theory_sq_output, atol=0.01, rtol=0.1))
            end
            @testset "FFT pops" begin
            end
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
