# =============================================================================
# Validation Script: Batched CPU vs Legacy vs GPU on 7 Standard Datasets
# =============================================================================
#
# Ejecutar con:
#   julia --project=. examples/validacion_batched_gpu.jl
#
# Requirements: 7.1, 7.2, 7.3, 7.4
#
# Runs the Random Cloud Method on 7 datasets in three modes:
#   1. Legacy (sample-by-sample)
#   2. Batched CPU (matrix operations)
#   3. GPU (CUDA, if available)
#
# Asserts:
#   - Batched CPU accuracy within ±1 pp of legacy per dataset
#   - GPU accuracy within ±1 pp of batched CPU per dataset (when CUDA available)
# Reports:
#   - Wall-clock time for each mode and dataset
#   - Peak GPU memory usage when running in GPU mode
# =============================================================================

using RandomCloud
using RandomCloud: RedNeuronal, ConfiguracionNube, InformeNube, GPU_AVAILABLE,
                   evaluar, activaciones_por_capa
using MLDatasets: Iris, Wine
import DataFrames
using Random
using Downloads: download

const SEED = 42

# ─── Dataset loaders ───────────────────────────────────────────────────────────

function normalizar!(features::Matrix{Float64})
    for fila in axes(features, 1)
        mn, mx = extrema(@view features[fila, :])
        if mx > mn
            features[fila, :] .= (features[fila, :] .- mn) ./ (mx - mn)
        end
    end
    features
end

function cargar_iris()
    dataset = Iris(as_df=false)
    features = Float64.(dataset.features)
    labels = vec(dataset.targets)
    normalizar!(features)
    clases = sort(unique(labels))
    idx = Dict(c => i for (i, c) in enumerate(clases))
    n = size(features, 2)
    Y = zeros(Float64, length(clases), n)
    for k in 1:n; Y[idx[labels[k]], k] = 1.0; end
    return (nombre="Iris", X=features, Y=Y, topo=[4, 8, 3])
end

function cargar_wine()
    dataset = Wine(as_df=false)
    features = Float64.(dataset.features)
    labels = vec(dataset.targets)
    normalizar!(features)
    nc = length(unique(labels))
    n = size(features, 2)
    Y = zeros(Float64, nc, n)
    for k in 1:n; Y[labels[k], k] = 1.0; end
    return (nombre="Wine", X=features, Y=Y, topo=[13, 8, 3])
end

function _descargar_cache(url, cache)
    if !isfile(cache)
        println("  Descargando $cache...")
        download(url, cache)
    end
    return cache
end

function cargar_breastcancer()
    cache = _descargar_cache(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
        joinpath(@__DIR__, "..", ".cache_breastcancer.csv"))
    lines = filter(l -> !isempty(strip(l)), readlines(cache))
    n = length(lines)
    features = zeros(Float64, 30, n)
    labels = zeros(Int, n)
    for (k, line) in enumerate(lines)
        parts = split(line, ',')
        labels[k] = parts[2] == "M" ? 1 : 0
        for j in 1:30; features[j, k] = parse(Float64, parts[j+2]); end
    end
    normalizar!(features)
    Y = zeros(Float64, 2, n)
    for k in 1:n; Y[labels[k]+1, k] = 1.0; end
    return (nombre="BreastCancer", X=features, Y=Y, topo=[30, 16, 2])
end

function cargar_ionosphere()
    cache = _descargar_cache(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data",
        joinpath(@__DIR__, "..", ".cache_ionosphere.csv"))
    lines = filter(l -> !isempty(strip(l)), readlines(cache))
    n = length(lines)
    features = zeros(Float64, 34, n)
    labels = zeros(Int, n)
    for (k, line) in enumerate(lines)
        parts = split(line, ',')
        for j in 1:34; features[j, k] = parse(Float64, parts[j]); end
        labels[k] = parts[35] == "g" ? 1 : 0
    end
    normalizar!(features)
    Y = zeros(Float64, 2, n)
    for k in 1:n; Y[labels[k]+1, k] = 1.0; end
    return (nombre="Ionosphere", X=features, Y=Y, topo=[34, 16, 2])
end

function cargar_sonar()
    cache = _descargar_cache(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data",
        joinpath(@__DIR__, "..", ".cache_sonar.csv"))
    lines = filter(l -> !isempty(strip(l)), readlines(cache))
    n = length(lines)
    features = zeros(Float64, 60, n)
    labels = zeros(Int, n)
    for (k, line) in enumerate(lines)
        parts = split(line, ',')
        for j in 1:60; features[j, k] = parse(Float64, parts[j]); end
        labels[k] = parts[61] == "M" ? 1 : 0
    end
    normalizar!(features)
    Y = zeros(Float64, 2, n)
    for k in 1:n; Y[labels[k]+1, k] = 1.0; end
    return (nombre="Sonar", X=features, Y=Y, topo=[60, 16, 2])
end

function cargar_digits()
    cache_train = _descargar_cache(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra",
        joinpath(@__DIR__, "..", ".cache_digits_train.csv"))
    lines = filter(l -> !isempty(strip(l)), readlines(cache_train))
    n = length(lines)
    features = zeros(Float64, 64, n)
    labels = zeros(Int, n)
    for (k, line) in enumerate(lines)
        parts = split(line, ',')
        for j in 1:64; features[j, k] = parse(Float64, parts[j]); end
        labels[k] = parse(Int, parts[65])
    end
    features ./= 16.0
    Y = zeros(Float64, 10, n)
    for k in 1:n; Y[labels[k]+1, k] = 1.0; end
    return (nombre="Digits", X=features, Y=Y, topo=[64, 32, 10])
end

function cargar_adult()
    cache = _descargar_cache(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        joinpath(@__DIR__, "..", ".cache_adult_train.csv"))
    lines = readlines(cache)
    lines = filter(l -> !isempty(strip(l)) && !occursin("?", l), lines)
    num_cols = [1, 3, 5, 11, 12, 13]
    nf = length(num_cols)
    n = length(lines)
    features = zeros(Float64, nf, n)
    labels = zeros(Int, n)
    for (i, line) in enumerate(lines)
        parts = [strip(p) for p in split(line, ',')]
        length(parts) < 15 && continue
        for (j, c) in enumerate(num_cols)
            features[j, i] = parse(Float64, parts[c])
        end
        labels[i] = occursin(">50K", parts[15]) ? 1 : 0
    end
    normalizar!(features)
    Y = zeros(Float64, 2, n)
    for k in 1:n; Y[labels[k]+1, k] = 1.0; end
    return (nombre="Adult", X=features, Y=Y, topo=[nf, 16, 2])
end

# ─── Run validation ────────────────────────────────────────────────────────────

function run_mode(mode::Symbol, features, objetivos, topo, seed; batch_size=0)
    gpu = mode == :gpu
    config = ConfiguracionNube(
        tamano_nube=30,
        topologia_inicial=topo,
        umbral_acierto=0.3,
        neuronas_eliminar=1,
        epocas_refinamiento=100,
        tasa_aprendizaje=0.1,
        semilla=seed,
        gpu=gpu,
        batch_size=batch_size
    )
    motor = MotorNube(config, features, objetivos)
    t0 = time_ns()
    if mode == :legacy
        informe = RandomCloud._ejecutar_legacy(motor)
    elseif mode == :batched
        informe = RandomCloud._ejecutar_batched(motor)
    else
        informe = ejecutar(motor)
    end
    wall_ms = (time_ns() - t0) / 1_000_000.0
    return (precision=informe.precision, exitoso=informe.exitoso,
            wall_ms=wall_ms, gpu_ms=informe.gpu_tiempo_ms,
            vram_mb=informe.pico_vram_mb)
end

function main()
    println("=" ^ 90)
    println("  VALIDATION: Batched CPU vs Legacy vs GPU — 7 Standard Datasets")
    println("=" ^ 90)
    println()
    println("  GPU available: $(GPU_AVAILABLE[])")
    println("  Seed: $SEED")
    println()

    loaders = [
        cargar_iris, cargar_wine, cargar_breastcancer,
        cargar_ionosphere, cargar_sonar, cargar_digits, cargar_adult
    ]

    results = []
    all_pass = true

    for loader in loaders
        data = loader()
        nombre = data.nombre
        println("-" ^ 90)
        println("  Dataset: $nombre ($(size(data.X, 1)) features × $(size(data.X, 2)) samples, topo=$(data.topo))")
        println("-" ^ 90)

        # Legacy
        r_legacy = run_mode(:legacy, data.X, data.Y, data.topo, SEED)
        println("  Legacy:  prec=$(round(r_legacy.precision*100, digits=1))%  time=$(round(r_legacy.wall_ms, digits=1))ms  exitoso=$(r_legacy.exitoso)")

        # Batched CPU
        r_batched = run_mode(:batched, data.X, data.Y, data.topo, SEED)
        println("  Batched: prec=$(round(r_batched.precision*100, digits=1))%  time=$(round(r_batched.wall_ms, digits=1))ms  exitoso=$(r_batched.exitoso)")

        diff_cpu = abs(r_legacy.precision - r_batched.precision)
        pass_cpu = diff_cpu <= 0.01
        status_cpu = pass_cpu ? "✓ PASS" : "✗ FAIL"
        println("  → CPU diff: $(round(diff_cpu*100, digits=2))pp  $status_cpu (±1pp tolerance)")
        all_pass &= pass_cpu

        # GPU (if available)
        if GPU_AVAILABLE[]
            r_gpu = run_mode(:gpu, data.X, data.Y, data.topo, SEED; batch_size=size(data.X, 2))
            println("  GPU:     prec=$(round(r_gpu.precision*100, digits=1))%  time=$(round(r_gpu.wall_ms, digits=1))ms  gpu_time=$(round(r_gpu.gpu_ms, digits=1))ms  vram=$(round(r_gpu.vram_mb, digits=2))MB")

            diff_gpu = abs(r_batched.precision - r_gpu.precision)
            pass_gpu = diff_gpu <= 0.01
            status_gpu = pass_gpu ? "✓ PASS" : "✗ FAIL"
            println("  → GPU diff: $(round(diff_gpu*100, digits=2))pp  $status_gpu (±1pp tolerance)")
            all_pass &= pass_gpu

            push!(results, (nombre=nombre, legacy=r_legacy.precision, batched=r_batched.precision,
                            gpu=r_gpu.precision, legacy_ms=r_legacy.wall_ms, batched_ms=r_batched.wall_ms,
                            gpu_ms=r_gpu.wall_ms, vram_mb=r_gpu.vram_mb))
        else
            push!(results, (nombre=nombre, legacy=r_legacy.precision, batched=r_batched.precision,
                            gpu=NaN, legacy_ms=r_legacy.wall_ms, batched_ms=r_batched.wall_ms,
                            gpu_ms=NaN, vram_mb=NaN))
        end
        println()
    end

    # Summary table
    println("=" ^ 90)
    println("  SUMMARY")
    println("=" ^ 90)
    println()
    header = rpad("Dataset", 15) * rpad("Legacy%", 10) * rpad("Batched%", 10)
    if GPU_AVAILABLE[]
        header *= rpad("GPU%", 10) * rpad("VRAM(MB)", 10)
    end
    header *= rpad("Legacy(ms)", 12) * rpad("Batched(ms)", 12)
    if GPU_AVAILABLE[]
        header *= rpad("GPU(ms)", 10)
    end
    println("  $header")
    println("  " * "-" ^ length(header))

    for r in results
        row = rpad(r.nombre, 15)
        row *= rpad("$(round(r.legacy*100, digits=1))", 10)
        row *= rpad("$(round(r.batched*100, digits=1))", 10)
        if GPU_AVAILABLE[]
            row *= rpad(isnan(r.gpu) ? "N/A" : "$(round(r.gpu*100, digits=1))", 10)
            row *= rpad(isnan(r.vram_mb) ? "N/A" : "$(round(r.vram_mb, digits=2))", 10)
        end
        row *= rpad("$(round(r.legacy_ms, digits=1))", 12)
        row *= rpad("$(round(r.batched_ms, digits=1))", 12)
        if GPU_AVAILABLE[]
            row *= rpad(isnan(r.gpu_ms) ? "N/A" : "$(round(r.gpu_ms, digits=1))", 10)
        end
        println("  $row")
    end

    println()
    if all_pass
        println("  ✓ ALL VALIDATIONS PASSED")
    else
        println("  ✗ SOME VALIDATIONS FAILED — check details above")
    end
    println("=" ^ 90)
end

main()
