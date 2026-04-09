# =============================================================================
# Scalability Harness: GPU-Accelerated Random Cloud on MNIST, Fashion-MNIST, CIFAR-10
# =============================================================================
#
# Ejecutar con:
#   julia --project=. examples/escalabilidad_gpu.jl
#
# Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
#
# For each dataset:
#   1. Run batched CPU mode, record accuracy and wall-clock time
#   2. If GPU available, run GPU mode, record accuracy, wall-clock time, GPU time, VRAM
#   3. Report speedup factor (CPU time / GPU time)
#   4. For CIFAR-10, catch VRAM error if it occurs
#
# Uses small cloud sizes (10-20) and few epochs (10-20) to keep runtime reasonable.
# =============================================================================

using RandomCloud
using RandomCloud: RedNeuronal, ConfiguracionNube, InformeNube, GPU_AVAILABLE,
                   evaluar, activaciones_por_capa
using MLDatasets: MNIST, FashionMNIST, CIFAR10
using Random

const SEED = 42

# ─── Dataset loaders ───────────────────────────────────────────────────────────

function cargar_mnist()
    println("  Cargando MNIST...")
    dataset = MNIST(Float64, :train)
    features = reshape(dataset[:].features, 784, 60_000)
    labels = dataset[:].targets
    Y = zeros(Float64, 10, 60_000)
    for k in 1:60_000
        Y[labels[k] + 1, k] = 1.0
    end
    return (nombre="MNIST", X=features, Y=Y,
            topo=[784, 128, 64, 10], n_samples=60_000, n_features=784, n_classes=10)
end

function cargar_fashion_mnist()
    println("  Cargando Fashion-MNIST...")
    dataset = FashionMNIST(Float64, :train)
    features = reshape(dataset[:].features, 784, 60_000)
    labels = dataset[:].targets
    Y = zeros(Float64, 10, 60_000)
    for k in 1:60_000
        Y[labels[k] + 1, k] = 1.0
    end
    return (nombre="Fashion-MNIST", X=features, Y=Y,
            topo=[784, 128, 64, 10], n_samples=60_000, n_features=784, n_classes=10)
end

function cargar_cifar10()
    println("  Cargando CIFAR-10...")
    dataset = CIFAR10(Float64, :train)
    raw = dataset[:].features  # 32×32×3×50000
    features = reshape(raw, 3072, 50_000)
    labels = dataset[:].targets
    Y = zeros(Float64, 10, 50_000)
    for k in 1:50_000
        Y[labels[k] + 1, k] = 1.0
    end
    return (nombre="CIFAR-10", X=features, Y=Y,
            topo=[3072, 256, 128, 10], n_samples=50_000, n_features=3072, n_classes=10)
end

# ─── VRAM estimation (local replica for reporting without CUDA) ────────────────

function _estimar_vram_local(config::ConfiguracionNube, entradas::Matrix{Float64})
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

# ─── Run helpers ───────────────────────────────────────────────────────────────

function run_cpu(data, seed; cloud_size=15, epochs=15)
    config = ConfiguracionNube(
        tamano_nube=cloud_size,
        topologia_inicial=data.topo,
        umbral_acierto=0.1,
        neuronas_eliminar=1,
        epocas_refinamiento=epochs,
        tasa_aprendizaje=0.1,
        semilla=seed,
        batch_size=size(data.X, 2),
        gpu=false
    )
    motor = MotorNube(config, data.X, data.Y)
    t0 = time_ns()
    informe = RandomCloud._ejecutar_batched(motor)
    wall_ms = (time_ns() - t0) / 1_000_000.0
    return (precision=informe.precision, exitoso=informe.exitoso, wall_ms=wall_ms)
end

function run_gpu(data, seed; cloud_size=15, epochs=15)
    config = ConfiguracionNube(
        tamano_nube=cloud_size,
        topologia_inicial=data.topo,
        umbral_acierto=0.1,
        neuronas_eliminar=1,
        epocas_refinamiento=epochs,
        tasa_aprendizaje=0.1,
        semilla=seed,
        batch_size=size(data.X, 2),
        gpu=true
    )
    motor = MotorNube(config, data.X, data.Y)
    t0 = time_ns()
    informe = ejecutar(motor)
    wall_ms = (time_ns() - t0) / 1_000_000.0
    return (precision=informe.precision, exitoso=informe.exitoso,
            wall_ms=wall_ms, gpu_ms=informe.gpu_tiempo_ms,
            vram_mb=informe.pico_vram_mb)
end

# ─── Main ──────────────────────────────────────────────────────────────────────

function main()
    println("=" ^ 90)
    println("  SCALABILITY HARNESS: GPU Random Cloud — MNIST, Fashion-MNIST, CIFAR-10")
    println("=" ^ 90)
    println()
    println("  GPU available: $(GPU_AVAILABLE[])")
    if !GPU_AVAILABLE[]
        println("  ⚠  CUDA not available — GPU benchmarks will be skipped.")
        println("     Install CUDA.jl and ensure a CUDA-capable GPU is present for GPU tests.")
    end
    println("  Seed: $SEED")
    println("  Cloud size: 15 | Epochs: 15 (small for reasonable runtime)")
    println()

    datasets = [cargar_mnist, cargar_fashion_mnist, cargar_cifar10]
    results = []

    for loader in datasets
        data = loader()
        nombre = data.nombre
        println("-" ^ 90)
        println("  Dataset: $nombre")
        println("    $(data.n_features) features × $(data.n_samples) samples, $(data.n_classes) classes")
        println("    Topology: $(data.topo)")

        # VRAM estimate
        config_est = ConfiguracionNube(
            tamano_nube=15, topologia_inicial=data.topo, semilla=SEED
        )
        vram_est = _estimar_vram_local(config_est, data.X)
        println("    Estimated VRAM: $(round(vram_est / 1e9, digits=3)) GB")
        println("-" ^ 90)

        # --- CPU batched mode ---
        print("  CPU (batched): running... ")
        r_cpu = run_cpu(data, SEED)
        println("done")
        println("    Accuracy:  $(round(r_cpu.precision * 100, digits=2))%")
        println("    Wall time: $(round(r_cpu.wall_ms / 1000, digits=2))s")
        println("    Exitoso:   $(r_cpu.exitoso)")

        # --- GPU mode ---
        r_gpu = nothing
        gpu_error = nothing
        speedup = NaN

        if GPU_AVAILABLE[]
            print("  GPU: running... ")
            try
                r_gpu = run_gpu(data, SEED)
                println("done")
                println("    Accuracy:  $(round(r_gpu.precision * 100, digits=2))%")
                println("    Wall time: $(round(r_gpu.wall_ms / 1000, digits=2))s")
                println("    GPU time:  $(round(r_gpu.gpu_ms / 1000, digits=2))s")
                println("    GPU frac:  $(round(r_gpu.gpu_ms / r_gpu.wall_ms * 100, digits=1))%")
                println("    Peak VRAM: $(round(r_gpu.vram_mb, digits=2)) MB")

                speedup = r_cpu.wall_ms / r_gpu.wall_ms
                println("    Speedup:   $(round(speedup, digits=2))× (CPU/GPU wall-clock)")

                # Check VRAM fits within 4 GB for MNIST and Fashion-MNIST
                if nombre in ("MNIST", "Fashion-MNIST")
                    if r_gpu.vram_mb < 4096.0
                        println("    ✓ VRAM within 4 GB limit")
                    else
                        println("    ✗ VRAM exceeds 4 GB limit! ($(round(r_gpu.vram_mb, digits=2)) MB)")
                    end
                end
            catch e
                if e isa ArgumentError
                    gpu_error = e
                    println("VRAM limit error (expected for large datasets)")
                    println("    Error: $(e.msg)")
                    if nombre == "CIFAR-10"
                        println("    ✓ CIFAR-10 correctly raised memory-limit error (Req 3.6)")
                    end
                else
                    rethrow(e)
                end
            end
        else
            println("  GPU: skipped (CUDA not available)")
        end

        push!(results, (
            nombre=nombre,
            n_features=data.n_features,
            n_samples=data.n_samples,
            vram_est_gb=vram_est / 1e9,
            cpu_prec=r_cpu.precision,
            cpu_wall_ms=r_cpu.wall_ms,
            cpu_exitoso=r_cpu.exitoso,
            gpu_prec=r_gpu !== nothing ? r_gpu.precision : NaN,
            gpu_wall_ms=r_gpu !== nothing ? r_gpu.wall_ms : NaN,
            gpu_time_ms=r_gpu !== nothing ? r_gpu.gpu_ms : NaN,
            gpu_vram_mb=r_gpu !== nothing ? r_gpu.vram_mb : NaN,
            speedup=speedup,
            gpu_error=gpu_error
        ))
        println()
    end

    # ─── Summary table ─────────────────────────────────────────────────────────
    println("=" ^ 90)
    println("  SUMMARY — Scalability Results")
    println("=" ^ 90)
    println()

    # Header
    hdr = rpad("Dataset", 16) * rpad("Samples", 8) * rpad("Features", 9) *
          rpad("VRAM Est", 10) * rpad("CPU Acc%", 9) * rpad("CPU Time", 10)
    if GPU_AVAILABLE[]
        hdr *= rpad("GPU Acc%", 9) * rpad("GPU Time", 10) *
               rpad("VRAM MB", 9) * rpad("Speedup", 8)
    end
    println("  $hdr")
    println("  " * "-" ^ length(hdr))

    for r in results
        row = rpad(r.nombre, 16)
        row *= rpad("$(r.n_samples)", 8)
        row *= rpad("$(r.n_features)", 9)
        row *= rpad("$(round(r.vram_est_gb, digits=3))G", 10)
        row *= rpad("$(round(r.cpu_prec * 100, digits=1))%", 9)
        row *= rpad("$(round(r.cpu_wall_ms / 1000, digits=1))s", 10)
        if GPU_AVAILABLE[]
            if r.gpu_error !== nothing
                row *= rpad("VRAM ERR", 9) * rpad("-", 10) * rpad("-", 9) * rpad("-", 8)
            elseif isnan(r.gpu_prec)
                row *= rpad("N/A", 9) * rpad("N/A", 10) * rpad("N/A", 9) * rpad("N/A", 8)
            else
                row *= rpad("$(round(r.gpu_prec * 100, digits=1))%", 9)
                row *= rpad("$(round(r.gpu_wall_ms / 1000, digits=1))s", 10)
                row *= rpad("$(round(r.gpu_vram_mb, digits=1))", 9)
                row *= rpad("$(round(r.speedup, digits=2))×", 8)
            end
        end
        println("  $row")
    end

    println()
    println("  Notes:")
    println("    - Cloud size: 15, Epochs: 15 (small for benchmarking)")
    println("    - VRAM Est = estimated GPU memory with 20% overhead")
    println("    - Speedup = CPU wall-clock / GPU wall-clock")
    if !GPU_AVAILABLE[]
        println("    - GPU columns omitted (CUDA not available)")
    end
    println("=" ^ 90)
end

main()
