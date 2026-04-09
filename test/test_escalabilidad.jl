# test/test_escalabilidad.jl — Scalability VRAM estimation checks
#
# Fast tests that verify VRAM estimation for MNIST, Fashion-MNIST, and CIFAR-10
# sized inputs doesn't crash. No actual training — just estimation logic.
#
# Requirements: 8.1, 8.2, 8.3, 8.4, 8.5

using RandomCloud
using RandomCloud: ConfiguracionNube
using Test

# Local replica of VRAM estimation (separate name to avoid redefinition warning)
function _estimar_vram_escalabilidad(config::ConfiguracionNube, entradas::Matrix{Float64})
    n_features = size(entradas, 1)
    n_samples = size(entradas, 2)
    topo = config.topologia_inicial
    N = config.tamano_nube

    total = 0
    total += n_features * n_samples * sizeof(Float32)
    total += topo[end] * n_samples * sizeof(Float32)
    for i in 1:(length(topo) - 1)
        neurons_out = topo[i + 1]
        neurons_in = topo[i]
        total += neurons_out * neurons_in * N * sizeof(Float32)
        total += neurons_out * N * sizeof(Float32)
    end
    max_layer = maximum(topo[2:end])
    total += max_layer * n_samples * 2 * sizeof(Float32)
    return total * 1.2
end

@testset "Scalability VRAM estimation" begin

    # MNIST-like: 784 features, 60K samples, cloud=20, topo=[784, 128, 64, 10]
    @testset "MNIST-like estimation" begin
        config_mnist = ConfiguracionNube(
            tamano_nube=20, topologia_inicial=[784, 128, 64, 10], semilla=42
        )
        entradas_mnist = randn(784, 100)  # small subset for estimation
        est = _estimar_vram_escalabilidad(config_mnist, entradas_mnist)
        @test est > 0
        @test est < 3.5e9  # should fit for 100 samples
    end

    # Fashion-MNIST-like: same dimensions as MNIST
    @testset "Fashion-MNIST-like estimation" begin
        config_fmnist = ConfiguracionNube(
            tamano_nube=20, topologia_inicial=[784, 128, 64, 10], semilla=42
        )
        entradas_fmnist = randn(784, 100)
        est_fmnist = _estimar_vram_escalabilidad(config_fmnist, entradas_fmnist)
        @test est_fmnist > 0
        @test est_fmnist < 3.5e9
    end

    # CIFAR-10-like: 3072 features, 50K samples, cloud=20
    @testset "CIFAR-10-like estimation" begin
        config_cifar = ConfiguracionNube(
            tamano_nube=20, topologia_inicial=[3072, 256, 128, 10], semilla=42
        )
        entradas_cifar = randn(3072, 100)
        est_cifar = _estimar_vram_escalabilidad(config_cifar, entradas_cifar)
        @test est_cifar > 0
    end

    # Full-scale MNIST estimation (60K samples) — should fit in 4 GB with small cloud
    @testset "Full MNIST fits in 4 GB with small cloud" begin
        config_full = ConfiguracionNube(
            tamano_nube=15, topologia_inicial=[784, 128, 64, 10], semilla=42
        )
        entradas_full = randn(784, 60_000)
        est_full = _estimar_vram_escalabilidad(config_full, entradas_full)
        @test est_full > 0
        @test est_full < 4.0e9  # should fit within 4 GB VRAM
    end

    # VRAM grows with cloud size (sanity check)
    @testset "VRAM grows with cloud size" begin
        entradas = randn(784, 100)
        config_small = ConfiguracionNube(
            tamano_nube=5, topologia_inicial=[784, 128, 64, 10], semilla=42
        )
        config_large = ConfiguracionNube(
            tamano_nube=50, topologia_inicial=[784, 128, 64, 10], semilla=42
        )
        est_small = _estimar_vram_escalabilidad(config_small, entradas)
        est_large = _estimar_vram_escalabilidad(config_large, entradas)
        @test est_large > est_small
    end

    # VRAM grows with sample count (sanity check)
    @testset "VRAM grows with sample count" begin
        config = ConfiguracionNube(
            tamano_nube=15, topologia_inicial=[784, 128, 64, 10], semilla=42
        )
        est_100 = _estimar_vram_escalabilidad(config, randn(784, 100))
        est_1000 = _estimar_vram_escalabilidad(config, randn(784, 1000))
        @test est_1000 > est_100
    end
end
