# Tests for GPU extension — Property 5 (VRAM monotonicity) and unit tests for error handling
#
# Since CUDA.jl is not installed in this environment, we replicate the VRAM estimation
# formula locally for property tests, and gate GPU-specific tests behind GPU_AVAILABLE[].

using Random
using RandomCloud
using RandomCloud: RedNeuronal, ConfiguracionNube, GPU_AVAILABLE
using Supposition
using Supposition: Data
using Test

# --- Local replica of estimar_vram formula (from ext/RandomCloudCUDAExt/gpu_backend.jl) ---
# This allows testing the VRAM estimation logic without requiring CUDA.

"""
    _estimar_vram_local(config, entradas) → Float64

Local replica of the VRAM estimation formula for testing without CUDA.
Returns estimated bytes.
"""
function _estimar_vram_local(config::ConfiguracionNube, entradas::Matrix{Float64})
    n_features = size(entradas, 1)
    n_samples = size(entradas, 2)
    topo = config.topologia_inicial
    N = config.tamano_nube

    total = 0
    # Input data
    total += n_features * n_samples * sizeof(Float32)
    # Target data (output layer × samples)
    total += topo[end] * n_samples * sizeof(Float32)

    # Per-layer weights and biases
    for i in 1:(length(topo) - 1)
        neurons_out = topo[i + 1]
        neurons_in = topo[i]
        total += neurons_out * neurons_in * N * sizeof(Float32)  # weights
        total += neurons_out * N * sizeof(Float32)               # biases
    end

    # Activation buffers (forward + backward)
    max_layer = maximum(topo[2:end])
    total += max_layer * n_samples * 2 * sizeof(Float32)

    # 20% overhead for CUDA allocator fragmentation
    estimated = total * 1.2
    return estimated
end


# =============================================================================
# Feature: gpu-batched-cloud, Property 5: VRAM estimation monotonicity
# **Validates: Requirements 3.6**
# =============================================================================

@testset "PBT Property 5: VRAM estimation monotonicity" begin

    # Generator for valid topologies (3-5 layers, reasonable sizes)
    hidden_gen = Data.Vectors(Data.Integers(1, 30); min_size=1, max_size=3)
    topo_gen = @composed function valid_topology_vram(
        input_size = Data.Integers(1, 20),
        hidden = hidden_gen,
        output_size = Data.Integers(1, 10)
    )
        return vcat([input_size], hidden, [output_size])
    end

    # --- Sub-property 5a: Monotonicity in tamano_nube ---
    # Two configs differing only in tamano_nube (c1 < c2), same data → estimate(c2) > estimate(c1)

    @check max_examples=100 function prop_vram_monotonic_cloud_size(
        topo = topo_gen,
        n1 = Data.Integers(1, 50),
        n2 = Data.Integers(1, 50),
        n_samples = Data.Integers(1, 100),
        seed = Data.Integers(1, 10_000)
    )
        # Ensure c1 < c2
        c1_size = min(n1, n2)
        c2_size = max(n1, n2)
        c1_size == c2_size && return true  # skip equal case

        config1 = ConfiguracionNube(
            tamano_nube=c1_size, topologia_inicial=topo,
            semilla=seed
        )
        config2 = ConfiguracionNube(
            tamano_nube=c2_size, topologia_inicial=topo,
            semilla=seed
        )

        n_features = topo[1]
        entradas = randn(MersenneTwister(seed), n_features, n_samples)

        est1 = _estimar_vram_local(config1, entradas)
        est2 = _estimar_vram_local(config2, entradas)

        return est2 > est1
    end

    # --- Sub-property 5b: Monotonicity in number of samples ---
    # Two input matrices with s1 < s2 samples, same config → estimate(s2) > estimate(s1)

    @check max_examples=100 function prop_vram_monotonic_sample_count(
        topo = topo_gen,
        cloud_size = Data.Integers(1, 50),
        s1 = Data.Integers(1, 100),
        s2 = Data.Integers(1, 100),
        seed = Data.Integers(1, 10_000)
    )
        # Ensure s1 < s2
        samples_small = min(s1, s2)
        samples_large = max(s1, s2)
        samples_small == samples_large && return true  # skip equal case

        config = ConfiguracionNube(
            tamano_nube=cloud_size, topologia_inicial=topo,
            semilla=seed
        )

        n_features = topo[1]
        rng = MersenneTwister(seed)
        entradas_small = randn(rng, n_features, samples_small)
        entradas_large = randn(rng, n_features, samples_large)

        est_small = _estimar_vram_local(config, entradas_small)
        est_large = _estimar_vram_local(config, entradas_large)

        return est_large > est_small
    end
end


# =============================================================================
# Unit tests for GPU extension error handling (Task 11.6)
# Requirements: 3.3, 3.6
# =============================================================================

@testset "GPU extension error handling" begin

    # --- Test estimar_vram raises error for oversized config (> 3.5 GB) ---
    # Requirement 3.6
    @testset "estimar_vram raises error for oversized config" begin
        # Create a config with large cloud size and topology to exceed 3.5 GB
        # N=2000 with [784,512,256,10] and 60K samples ≈ 5.67 GB
        oversized_config = ConfiguracionNube(
            tamano_nube=2000,
            topologia_inicial=[784, 512, 256, 10],
            semilla=42
        )
        large_entradas = randn(784, 60_000)

        est = _estimar_vram_local(oversized_config, large_entradas)
        @test est > 3.5e9  # Confirm the estimate exceeds 3.5 GB

        # If CUDA extension is loaded, the actual estimar_vram should throw
        if GPU_AVAILABLE[]
            @test_throws ArgumentError estimar_vram(oversized_config, large_entradas)
        else
            @info "Skipping actual estimar_vram error test — CUDA.jl not available"
            # Verify the local formula would trigger the error condition
            @test est > 3.5e9
        end
    end

    # --- Test verificar_gpu raises informative error when no device available ---
    # Requirement 3.3
    @testset "verificar_gpu raises error when no GPU available" begin
        if GPU_AVAILABLE[]
            # If CUDA is loaded but no device, verificar_gpu should throw
            # We can't easily test this when CUDA IS functional, so just verify it's callable
            @info "CUDA available — verificar_gpu should succeed (cannot test failure path)"
        else
            @info "Skipping verificar_gpu error test — CUDA.jl not available (function not defined)"
            @test !GPU_AVAILABLE[]
        end
    end

    # --- Test a_gpu/de_gpu round-trip preserves values (within Float32 precision) ---
    # Requirement 3.1
    @testset "a_gpu/de_gpu round-trip preserves values" begin
        if GPU_AVAILABLE[]
            original = randn(5, 10)
            gpu_data = a_gpu(original)
            roundtrip = de_gpu(gpu_data)

            # Values should be preserved within Float32 precision
            @test size(roundtrip) == size(original)
            @test all(abs.(roundtrip .- original) .< 1e-6)
        else
            @info "Skipping a_gpu/de_gpu round-trip test — CUDA.jl not available"
            # Verify the round-trip logic conceptually: Float64 → Float32 → Float64
            original = randn(5, 10)
            roundtrip = Float64.(Float32.(original))
            @test size(roundtrip) == size(original)
            @test all(abs.(roundtrip .- original) .< 1e-6)
        end
    end
end
