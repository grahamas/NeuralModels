push!(LOAD_PATH, "@stdlib")
using Pkg
Pkg.activate(@__DIR__)

using Test
using Simulation73
using NeuralModels
using DSP
using SpecialFunctions
using Plots; unicodeplots()

include("src/test_sanity.jl")

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
        whole_lattice_stim_param = CircleStimulusParameter(; center=(0.0,), strength=10.0, radius=extent/2, time_windows=[(0.0,45.0)])
        whole_lattice_stim = whole_lattice_stim_param(lattice)
        whole_lattice_bump = copy(lattice_zeros)
        whole_lattice_stim(whole_lattice_bump, whole_lattice_bump, 0.0)
        @test all(whole_lattice_bump .== (lattice_zeros .+ 10.0))

        manual_sharp_bump_radius = 10.0
        half_width = round(Int,manual_sharp_bump_radius / dx)
        manual_sharp_bump = copy(lattice_zeros)
        manual_sharp_bump[mid_point-half_width:mid_point+half_width] .= 1.0

        sharp_bump_stim_param = CircleStimulusParameter(; center=(0.0,), strength=1.0, radius=manual_sharp_bump_radius, time_windows=[(0.0,45.0)])
        sharp_bump_stim = sharp_bump_stim_param(lattice)
        sharp_bump_test = copy(lattice_zeros)
        sharp_bump_stim(sharp_bump_test, sharp_bump_test, 0.0)
        @test all(sharp_bump_test .== manual_sharp_bump)
        sharp_bump_stim(sharp_bump_test, sharp_bump_test, 55.0)
        @test all(sharp_bump_test .== manual_sharp_bump)
    end
    @testset "Array" begin
        n_points, extent, dx, mid_point, lattice, lattice_zeros = make_testing_lattice(n_points=100, extent=100.0)
        wide = CircleStimulusParameter(;
                      strength = 1.0,
                      radius = 20.0,
                      time_windows = [(0.0, 20.0)])
        thin = CircleStimulusParameter(;
                      strength = 1.0,
                      radius = 10.0,
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


function fft_n_points_test(n_points, atol=0.05, rtol=0.05)
    n_points, extent, dx, mid_point, circle, circle_zeros = make_testing_lattice(n_points=n_points, type=PeriodicLattice{Float64,1})
    radius = extent/2
    manual_bump = copy(circle_zeros)
    stim_radius = 30.0
    manual_bump_fn(x) = -stim_radius <= only(x) <= stim_radius ? 1.0 : 0.0
    manual_bump = manual_bump_fn.(coordinates(circle))
    fft_gaussian_output, naive_gaussian_output, analytical_gaussian_output = nothing, nothing, nothing
    @testset "Exponentially decaying connectivity" begin
        σ = 20.0
        gaussian_conn_param = GaussianConnectivityParameter(amplitude=1.0, spread=(σ,))
        gaussian_dsp_output = conv(NeuralModels.kernel(gaussian_conn_param, circle), manual_bump)
        abs_conn_param = ExpAbsSumDecayingConnectivityParameter(amplitude=1.0, spread=(σ,))
        abs_dsp_output = conv(NeuralModels.kernel(abs_conn_param, circle), manual_bump)
        @test directed_weights(gaussian_conn_param, circle, (-radius,)) ≈ directed_weights(gaussian_conn_param, circle, (radius,))
        plt = plot(directed_weights(gaussian_conn_param, circle, (-radius,)))
        plot!(plt, directed_weights(gaussian_conn_param, circle, (radius,)))
        display(plt)
        @testset "FFT" begin
            gaussian_conn_action = gaussian_conn_param(circle)
            abs_conn_action = abs_conn_param(circle)

            naive_gaussian_output = zeros(size(manual_bump)...)
            gaussian_conn_action(naive_gaussian_output, manual_bump, 0.0)

            fft_gaussian_conn_param = FFTParameter(gaussian_conn_param)
            fft_gaussian_conn_action = fft_gaussian_conn_param(circle)

            fft_gaussian_output = zeros(size(manual_bump)...)
            fft_gaussian_conn_action(fft_gaussian_output, manual_bump, 0.0)

            # Assumes analytical form of Gaussian normalizes to 1
            # https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node228.html ish
            analytical_gaussian_conv(x) =  (erf(( stim_radius - x[1])/ (sqrt(2) * σ)) + erf((x[1]+stim_radius)/(sqrt(2) * σ))) / (2)
            analytical_gaussian_output = analytical_gaussian_conv.(coordinates(circle))

            @test isapprox(fft_gaussian_output,analytical_gaussian_output, atol=atol, rtol=rtol)
            plt = plot(fft_gaussian_output)
            plot!(plt, analytical_gaussian_output)
            display(plt)
            @test isapprox(naive_gaussian_output,analytical_gaussian_output, atol = atol, rtol=rtol)
            plt = plot(naive_gaussian_output)
            plot!(plt, analytical_gaussian_output)
            display(plt)
        end
        @testset "FFT pops" begin
            sigmas = [20.0 10.0; 30.0 40.0] .|> (x) -> (x,)
            amplitudes = [1.0 -1.0; 1.0 1.0]
            gaussian_conn_pop_params = pops(GaussianConnectivityParameter;
                amplitude=amplitudes, spread=sigmas)


            pops_bump = population_repeat(manual_bump, 2)
            manual_input = copy(pops_bump)
            manual_output = zeros(size(pops_bump))
            fft_input = copy(pops_bump)
            fft_output = zeros(size(pops_bump))
            plot(manual_input) |> display

            for i in 1:2
                for j in 1:2
                    manual_conn_action = FFTParameter(GaussianConnectivityParameter(amplitude=amplitudes[i,j], spread=sigmas[i,j]))(circle)
                    manual_conn_action(population(manual_output,i), population(manual_input,j), 0.0)
                end
            end

            fft_gaussian_conn_pop_params = FFTParameter(gaussian_conn_pop_params)
            fft_pops_action = fft_gaussian_conn_pop_params(circle)
            plot(fft_input) |> display
            fft_pops_action(fft_output, fft_input, 0.0)

            plot(fft_output) |> display
            plot(manual_output) |> display
            @test all(fft_output .== manual_output) # Manual output just unrolls what the pops should be doing; no approximation
        end
    end
    return (analytical_gaussian_output, naive_gaussian_output, fft_gaussian_output)
end

@testset "Connectivity" begin
    @testset "Odd n_points" begin
        (analytical_gaussian_output, naive_gaussian_output, fft_gaussian_output) = fft_n_points_test(101, 0.01, 0.1)
    end
    @testset "Even n_points" begin
        (analytical_gaussian_output, naive_gaussian_output, fft_gaussian_output) = fft_n_points_test(100)
    end
end

@testset "Nonlinearity" begin
    n_points, extent, dx, mid_point, circle, circle_zeros = make_testing_lattice(n_points=100, type=PeriodicLattice{Float64,1})
    @testset "Binary switch" begin
        @test binary_switch_off_on(0., 1.) == 0.
        @test binary_switch_off_on(2., 1.) == 1.
        @test binary_switch_off_on_off(0., 1., 3.) == 0.
        @test binary_switch_off_on_off(2., 1., 3.) == 1.
        @test binary_switch_off_on_off(4., 1., 3.) == 0.
    end
    @testset "Sigmoid" begin
        sn = RectifiedSigmoidNonlinearity(a=1.0, θ=5.0)
        test_vals = [-100.0, 0.0, 0.01, 5.0, 200.0]
        sn(test_vals)
        @test isapprox(test_vals[1], 0.0, atol=sqrt(eps()))
        @test isapprox(test_vals[2], 0.0, atol=0.01)
        @test isapprox(test_vals[3], 0.0, atol=0.01)
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
    @testset "Difference of Sigmoids" begin
        fθ = 7.0
        bθ = 12.0
        fsig = RectifiedSigmoidNonlinearity(a=5.0, θ=fθ)
        bsig = RectifiedSigmoidNonlinearity(a=5.0, θ=bθ)
        dosp = DifferenceOfSigmoidsParameter(fsig, bsig) 
        test_vals = [-100.0, 0.0, 0.01, fθ, 200.0, (fθ + bθ) / 2, bθ]
        dos! = DifferenceOfSigmoids(dosp, zero(test_vals))
        dos!(test_vals)
        @test isapprox(test_vals[1], 0.0, atol=sqrt(eps()))
        @test isapprox(test_vals[2], 0.0, atol=0.01)
        @test isapprox(test_vals[3], 0.0, atol=0.01)
        @test isapprox(test_vals[4], 0.5, atol=0.01)
        @test isapprox(test_vals[5], 0.0, atol=0.01) # returned to zero
        @test isapprox(test_vals[6], 1.0, atol=0.01)
        @test isapprox(test_vals[7], 0.5, atol=0.01)
    end
    @testset "Normed Difference of Sigmoids" begin
        fθ = 7.0
        bθ = 12.0
        fsig = RectifiedSigmoidNonlinearity(a=5.0, θ=fθ)
        bsig = RectifiedSigmoidNonlinearity(a=5.0, θ=bθ)
        ndosp = NormedDifferenceOfSigmoidsParameter(fsig, bsig)
        test_vals = [-100.0, 0.0, 0.01, fθ, 200.0, (fθ + bθ) / 2, bθ]
        ndos! = NormedDifferenceOfSigmoids(DifferenceOfSigmoids(ndosp.dosp, zero(test_vals)), NeuralModels.calc_norm_factor(ndosp.dosp))
        ndos!(test_vals)
        # TODO better to have tested values written into tests
        @test isapprox(test_vals[1], 0.0, atol=sqrt(eps()))
        @test isapprox(test_vals[2], 0.0, atol=0.01)
        @test isapprox(test_vals[3], 0.0, atol=0.01)
        @test isapprox(test_vals[4], 0.5, atol=0.01)
        @test isapprox(test_vals[5], 0.0, atol=0.01) # back to zero
        @test isapprox(test_vals[6], 1.0, atol=0.01)
        @test isapprox(test_vals[7], 0.5, atol=0.01)
    end
end
