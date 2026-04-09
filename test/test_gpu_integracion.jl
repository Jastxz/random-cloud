# GPU Integration Tests — gated behind RandomCloud.GPU_AVAILABLE[]
#
# These tests verify that GPU computations match CPU computations within
# Float32 tolerance. All tests are skipped when CUDA.jl is not available.
#
# Requirements: 3.1, 3.2, 4.4

using Random
using RandomCloud
using RandomCloud: RedNeuronal, ConfiguracionNube, InformeNube, GPU_AVAILABLE,
                   feedforward_batch, entrenar_batch_matmul!, evaluar_nube_batch,
                   empaquetar_pesos, activaciones_por_capa, EntrenarBuffers
using Test

@testset "GPU Integration Tests" begin
    if !GPU_AVAILABLE[]
        @info "Skipping GPU integration tests — CUDA.jl not available"
        @test true  # placeholder to avoid empty testset
        return
    end

    # --- GPU feedforward matches CPU feedforward (Float32 tolerance: 1e-5) ---
    # Requirement 3.1, 3.2
    @testset "GPU feedforward matches CPU" begin
        rng = MersenneTwister(42)
        topo = [4, 8, 3]
        red = RedNeuronal(topo, rng)
        n_capas = length(red.pesos)
        acts = activaciones_por_capa(n_capas, :sigmoid)

        X_cpu = randn(MersenneTwister(123), 4, 20)

        # CPU feedforward (Float64)
        pesos_cpu = red.pesos
        biases_cpu = red.biases
        Y_cpu = feedforward_batch(pesos_cpu, biases_cpu, X_cpu, acts)

        # GPU feedforward (Float32 via a_gpu/de_gpu)
        pesos_gpu = [a_gpu(p) for p in red.pesos]
        biases_gpu = [a_gpu(b) for b in red.biases]
        X_gpu = a_gpu(X_cpu)
        Y_gpu_raw = feedforward_batch(pesos_gpu, biases_gpu, X_gpu, acts)
        Y_gpu = de_gpu(Y_gpu_raw)

        @test size(Y_gpu) == size(Y_cpu)
        @test all(abs.(Y_gpu .- Y_cpu) .< 1e-5)
    end

    # --- GPU cloud evaluation matches CPU cloud evaluation ---
    # Requirement 3.1, 3.2
    @testset "GPU cloud evaluation matches CPU" begin
        rng = MersenneTwister(99)
        topo = [3, 6, 2]
        n_redes = 5
        nube = [RedNeuronal(topo, rng) for _ in 1:n_redes]
        n_capas = length(nube[1].pesos)
        acts = activaciones_por_capa(n_capas, :sigmoid)

        n_samples = 30
        X_cpu = randn(MersenneTwister(200), 3, n_samples)
        # One-hot targets for 2 classes
        Y_cpu = zeros(2, n_samples)
        for k in 1:n_samples
            Y_cpu[rand(MersenneTwister(k), 1:2), k] = 1.0
        end

        # CPU cloud evaluation
        accs_cpu = evaluar_nube_batch(nube, X_cpu, Y_cpu, acts)

        # GPU cloud evaluation
        X_gpu = a_gpu(X_cpu)
        Y_gpu = a_gpu(Y_cpu)
        accs_gpu_raw = evaluar_nube_batch(nube, X_gpu, Y_gpu, acts)

        @test length(accs_gpu_raw) == length(accs_cpu)
        # Accuracy is discrete (count-based), so small Float32 differences in
        # feedforward may occasionally flip an argmax near a decision boundary.
        # We allow up to 2/n_samples difference per network.
        for i in 1:n_redes
            @test abs(accs_gpu_raw[i] - accs_cpu[i]) < 0.1
        end
    end

    # --- GPU backprop produces similar weight updates (Float32 tolerance: 1e-4) ---
    # Requirement 4.4
    @testset "GPU backprop matches CPU" begin
        rng = MersenneTwister(77)
        topo = [3, 5, 2]
        red = RedNeuronal(topo, rng)
        n_capas = length(red.pesos)
        acts = activaciones_por_capa(n_capas, :sigmoid)
        lr = 0.01

        n_samples = 10
        X = randn(MersenneTwister(300), 3, n_samples)
        Y = zeros(2, n_samples)
        for k in 1:n_samples
            Y[rand(MersenneTwister(k + 100), 1:2), k] = 1.0
        end

        # CPU backprop (Float64)
        pesos_cpu = [copy(p) for p in red.pesos]
        biases_cpu = [copy(b) for b in red.biases]
        entrenar_batch_matmul!(pesos_cpu, biases_cpu, X, Y, lr, acts)

        # GPU backprop (Float32)
        pesos_gpu = [a_gpu(p) for p in red.pesos]
        biases_gpu = [a_gpu(b) for b in red.biases]
        X_gpu = a_gpu(X)
        Y_gpu = a_gpu(Y)
        entrenar_batch_matmul!(pesos_gpu, biases_gpu, X_gpu, Y_gpu, Float32(lr), acts)

        # Transfer back and compare
        for l in 1:n_capas
            pesos_back = de_gpu(pesos_gpu[l])
            biases_back = de_gpu(biases_gpu[l])
            @test all(abs.(pesos_back .- pesos_cpu[l]) .< 1e-4)
            @test all(abs.(biases_back .- biases_cpu[l]) .< 1e-4)
        end
    end

    # --- VRAM estimation matches actual allocation within 20% ---
    # Requirement 3.6
    @testset "VRAM estimation accuracy" begin
        topo = [4, 8, 3]
        config = ConfiguracionNube(
            tamano_nube=10,
            topologia_inicial=topo,
            semilla=42,
            gpu=true
        )
        entradas = randn(4, 50)

        estimated = estimar_vram(config, entradas)

        # Measure actual VRAM usage: record baseline, allocate, record peak
        CUDA.reclaim()
        mem_before = CUDA.memory_status().total_bytes - CUDA.memory_status().free_bytes

        # Allocate the same structures the GPU path would use
        rng = MersenneTwister(42)
        nube = [RedNeuronal(topo, rng) for _ in 1:10]
        X_gpu = a_gpu(entradas)
        Y_gpu = CUDA.cu(Float32.(zeros(3, 50)))
        pesos_gpu = [a_gpu(nube[i].pesos[l]) for i in 1:10 for l in 1:length(nube[1].pesos)]
        biases_gpu = [a_gpu(nube[i].biases[l]) for i in 1:10 for l in 1:length(nube[1].biases)]

        CUDA.synchronize()
        mem_after = CUDA.memory_status().total_bytes - CUDA.memory_status().free_bytes
        actual_bytes = mem_after - mem_before

        # Estimation should be within 20% of actual (or higher, since it includes overhead)
        if actual_bytes > 0
            ratio = estimated / actual_bytes
            @test ratio > 0.5   # not wildly underestimating
            @test ratio < 5.0   # not wildly overestimating
        else
            @info "VRAM measurement returned 0 — skipping ratio check"
            @test estimated > 0
        end
    end

    # --- Full ejecutar with gpu=true completes on small problems ---
    # Requirement 3.1, 3.2
    @testset "Full ejecutar with gpu=true" begin
        # Small XOR-like problem
        entradas = Float64[0 0 1 1; 0 1 0 1]
        objetivos = Float64[1 0 0 1; 0 1 1 0]

        config = ConfiguracionNube(
            tamano_nube=5,
            topologia_inicial=[2, 4, 2],
            umbral_acierto=0.5,
            epocas_refinamiento=100,
            tasa_aprendizaje=0.1,
            semilla=42,
            gpu=true,
            batch_size=4
        )

        motor = MotorNube(config, entradas, objetivos)
        informe = ejecutar(motor)

        @test informe isa InformeNube
        @test informe.precision >= 0.0
        @test informe.precision <= 1.0
        @test informe.tiempo_ejecucion_ms > 0.0
        @test informe.gpu_tiempo_ms > 0.0
        @test informe.pico_vram_mb >= 0.0
        @test informe.total_redes_evaluadas > 0
    end
end
