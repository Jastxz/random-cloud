# Validation suite: batched CPU vs legacy vs GPU on standard datasets
# Requirements: 7.1, 7.2, 7.3, 7.4
#
# This test validates that the batched CPU path produces accuracy within ±1 percentage
# point of the legacy sample-by-sample path on real and synthetic datasets.
# GPU validation is included when CUDA is available.
#
# Uses small cloud sizes and few epochs to keep test runtime reasonable.

using Random
using RandomCloud
using RandomCloud: RedNeuronal, ConfiguracionNube, InformeNube, GPU_AVAILABLE,
                   evaluar, activaciones_por_capa
using MLDatasets
import DataFrames
using Test

const VALIDATION_SEED = 42

# ─── Dataset loaders ───────────────────────────────────────────────────────────

function _normalizar_features!(features::Matrix{Float64})
    for fila in axes(features, 1)
        mn, mx = extrema(@view features[fila, :])
        if mx > mn
            features[fila, :] .= (features[fila, :] .- mn) ./ (mx - mn)
        end
    end
    features
end

function _cargar_iris()
    try
        dataset = Iris(as_df=false)
        features = Float64.(dataset.features)  # 4×150
        labels = vec(dataset.targets)
        _normalizar_features!(features)
        clases = sort(unique(labels))
        clase_idx = Dict(c => i for (i, c) in enumerate(clases))
        n = size(features, 2)
        objetivos = zeros(Float64, length(clases), n)
        for k in 1:n
            objetivos[clase_idx[labels[k]], k] = 1.0
        end
        return (nombre="Iris", features=features, objetivos=objetivos,
                n_features=4, n_clases=length(clases), topo=[4, 8, 3])
    catch e
        @info "Could not load Iris: $e"
        return nothing
    end
end

function _cargar_wine()
    try
        dataset = Wine(as_df=false)
        features = Float64.(dataset.features)  # 13×178
        labels = vec(dataset.targets)
        _normalizar_features!(features)
        n_clases = length(unique(labels))
        n = size(features, 2)
        objetivos = zeros(Float64, n_clases, n)
        for k in 1:n
            objetivos[labels[k], k] = 1.0
        end
        return (nombre="Wine", features=features, objetivos=objetivos,
                n_features=13, n_clases=n_clases, topo=[13, 8, 3])
    catch e
        @info "Could not load Wine: $e"
        return nothing
    end
end

function _cargar_csv_cache(nombre, cache_path, n_features, parse_fn)
    if !isfile(cache_path)
        return nothing
    end
    lines = filter(l -> !isempty(strip(l)), readlines(cache_path))
    return parse_fn(lines, n_features)
end

function _cargar_breastcancer()
    cache = joinpath(@__DIR__, "..", ".cache_breastcancer.csv")
    _cargar_csv_cache("BreastCancer", cache, 30, function(lines, n_feat)
        n = length(lines)
        features = zeros(Float64, n_feat, n)
        labels = zeros(Int, n)
        for (k, line) in enumerate(lines)
            parts = split(line, ',')
            length(parts) < 32 && continue
            labels[k] = parts[2] == "M" ? 1 : 0
            for j in 1:n_feat
                features[j, k] = parse(Float64, parts[j + 2])
            end
        end
        _normalizar_features!(features)
        objetivos = zeros(Float64, 2, n)
        for k in 1:n
            objetivos[labels[k] + 1, k] = 1.0
        end
        return (nombre="BreastCancer", features=features, objetivos=objetivos,
                n_features=n_feat, n_clases=2, topo=[30, 8, 2])
    end)
end

function _cargar_ionosphere()
    cache = joinpath(@__DIR__, "..", ".cache_ionosphere.csv")
    _cargar_csv_cache("Ionosphere", cache, 34, function(lines, n_feat)
        n = length(lines)
        features = zeros(Float64, n_feat, n)
        labels = zeros(Int, n)
        for (k, line) in enumerate(lines)
            parts = split(line, ',')
            length(parts) < 35 && continue
            for j in 1:n_feat
                features[j, k] = parse(Float64, parts[j])
            end
            labels[k] = parts[35] == "g" ? 1 : 0
        end
        _normalizar_features!(features)
        objetivos = zeros(Float64, 2, n)
        for k in 1:n
            objetivos[labels[k] + 1, k] = 1.0
        end
        return (nombre="Ionosphere", features=features, objetivos=objetivos,
                n_features=n_feat, n_clases=2, topo=[34, 8, 2])
    end)
end

function _cargar_sonar()
    cache = joinpath(@__DIR__, "..", ".cache_sonar.csv")
    _cargar_csv_cache("Sonar", cache, 60, function(lines, n_feat)
        n = length(lines)
        features = zeros(Float64, n_feat, n)
        labels = zeros(Int, n)
        for (k, line) in enumerate(lines)
            parts = split(line, ',')
            length(parts) < 61 && continue
            for j in 1:n_feat
                features[j, k] = parse(Float64, parts[j])
            end
            labels[k] = parts[61] == "M" ? 1 : 0
        end
        _normalizar_features!(features)
        objetivos = zeros(Float64, 2, n)
        for k in 1:n
            objetivos[labels[k] + 1, k] = 1.0
        end
        return (nombre="Sonar", features=features, objetivos=objetivos,
                n_features=n_feat, n_clases=2, topo=[60, 16, 2])
    end)
end

function _cargar_digits()
    cache_train = joinpath(@__DIR__, "..", ".cache_digits_train.csv")
    cache_test = joinpath(@__DIR__, "..", ".cache_digits_test.csv")
    if !isfile(cache_train)
        return nothing
    end
    lines = filter(l -> !isempty(strip(l)), readlines(cache_train))
    n = min(length(lines), 500)  # Use subset for speed
    features = zeros(Float64, 64, n)
    labels = zeros(Int, n)
    for (k, line) in enumerate(lines)
        k > n && break
        parts = split(line, ',')
        for j in 1:64
            features[j, k] = parse(Float64, parts[j])
        end
        labels[k] = parse(Int, parts[65])
    end
    features ./= 16.0
    objetivos = zeros(Float64, 10, n)
    for k in 1:n
        objetivos[labels[k] + 1, k] = 1.0
    end
    return (nombre="Digits", features=features, objetivos=objetivos,
            n_features=64, n_clases=10, topo=[64, 16, 10])
end

function _cargar_adult()
    cache_train = joinpath(@__DIR__, "..", ".cache_adult_train.csv")
    if !isfile(cache_train)
        return nothing
    end
    # Simplified: use only numeric columns for speed in tests
    lines = readlines(cache_train)
    lines = filter(l -> !isempty(strip(l)) && !occursin("?", l), lines)
    n = min(length(lines), 500)  # Use subset for speed
    # Use 6 numeric columns: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
    num_cols = [1, 3, 5, 11, 12, 13]
    n_feat = length(num_cols)
    features = zeros(Float64, n_feat, n)
    labels = zeros(Int, n)
    for i in 1:n
        parts = [strip(p) for p in split(lines[i], ',')]
        length(parts) < 15 && continue
        for (j, c) in enumerate(num_cols)
            features[j, i] = parse(Float64, parts[c])
        end
        labels[i] = occursin(">50K", parts[15]) ? 1 : 0
    end
    _normalizar_features!(features)
    objetivos = zeros(Float64, 2, n)
    for k in 1:n
        objetivos[labels[k] + 1, k] = 1.0
    end
    return (nombre="Adult", features=features, objetivos=objetivos,
            n_features=n_feat, n_clases=2, topo=[n_feat, 8, 2])
end

# ─── Helper: run legacy and batched paths, compare accuracy ───────────────────

function _ejecutar_y_comparar(nombre, features, objetivos, topo, seed)
    config = ConfiguracionNube(
        tamano_nube=10,
        topologia_inicial=topo,
        umbral_acierto=0.3,
        neuronas_eliminar=1,
        epocas_refinamiento=50,
        tasa_aprendizaje=0.1,
        semilla=seed
    )

    # Legacy path
    motor_legacy = MotorNube(config, features, objetivos)
    t0 = time_ns()
    informe_legacy = RandomCloud._ejecutar_legacy(motor_legacy)
    t_legacy = (time_ns() - t0) / 1_000_000.0

    # Batched CPU path
    motor_batched = MotorNube(config, features, objetivos)
    t0 = time_ns()
    informe_batched = RandomCloud._ejecutar_batched(motor_batched)
    t_batched = (time_ns() - t0) / 1_000_000.0

    return (nombre=nombre,
            legacy_prec=informe_legacy.precision,
            legacy_exitoso=informe_legacy.exitoso,
            legacy_ms=t_legacy,
            batched_prec=informe_batched.precision,
            batched_exitoso=informe_batched.exitoso,
            batched_ms=t_batched)
end

# ─── Test suite ────────────────────────────────────────────────────────────────

@testset "Validation Suite: Batched CPU vs Legacy" begin

    # Synthetic XOR — always available, fast baseline
    @testset "XOR" begin
        X = Float64[0 0 1 1; 0 1 0 1]
        Y = Float64[1 0 0 1; 0 1 1 0]
        result = _ejecutar_y_comparar("XOR", X, Y, [2, 4, 2], VALIDATION_SEED)
        @test abs(result.legacy_prec - result.batched_prec) <= 0.01
        @test result.legacy_exitoso == result.batched_exitoso
        @info "XOR: legacy=$(round(result.legacy_prec*100, digits=1))% batched=$(round(result.batched_prec*100, digits=1))% | legacy=$(round(result.legacy_ms, digits=1))ms batched=$(round(result.batched_ms, digits=1))ms"
    end

    # Real datasets — skip if cache/MLDatasets not available
    datasets_loaders = [
        ("Iris", _cargar_iris),
        ("Wine", _cargar_wine),
        ("BreastCancer", _cargar_breastcancer),
        ("Ionosphere", _cargar_ionosphere),
        ("Sonar", _cargar_sonar),
        ("Digits", _cargar_digits),
        ("Adult", _cargar_adult),
    ]

    for (nombre, loader) in datasets_loaders
        @testset "$nombre" begin
            data = loader()
            if data === nothing
                @info "Skipping $nombre — data not available (cache file or MLDatasets missing)"
                @test true  # placeholder
                continue
            end

            result = _ejecutar_y_comparar(
                data.nombre, data.features, data.objetivos, data.topo, VALIDATION_SEED
            )

            # Requirement 7.1: batched CPU accuracy within ±1 percentage point of legacy
            @test abs(result.legacy_prec - result.batched_prec) <= 0.01
            @test result.legacy_exitoso == result.batched_exitoso

            @info "$nombre: legacy=$(round(result.legacy_prec*100, digits=1))% batched=$(round(result.batched_prec*100, digits=1))% | legacy=$(round(result.legacy_ms, digits=1))ms batched=$(round(result.batched_ms, digits=1))ms"
        end
    end
end

# ─── GPU Validation (when CUDA available) ─────────────────────────────────────

@testset "Validation Suite: GPU vs Batched CPU" begin
    if !GPU_AVAILABLE[]
        @info "Skipping GPU validation — CUDA.jl not available"
        @test true
        return
    end

    # Small synthetic dataset for GPU validation
    @testset "XOR GPU" begin
        X = Float64[0 0 1 1; 0 1 0 1]
        Y = Float64[1 0 0 1; 0 1 1 0]

        config_cpu = ConfiguracionNube(
            tamano_nube=10, topologia_inicial=[2, 4, 2],
            umbral_acierto=0.3, neuronas_eliminar=1,
            epocas_refinamiento=50, tasa_aprendizaje=0.1,
            semilla=VALIDATION_SEED, batch_size=4
        )
        config_gpu = ConfiguracionNube(
            tamano_nube=10, topologia_inicial=[2, 4, 2],
            umbral_acierto=0.3, neuronas_eliminar=1,
            epocas_refinamiento=50, tasa_aprendizaje=0.1,
            semilla=VALIDATION_SEED, gpu=true, batch_size=4
        )

        motor_cpu = MotorNube(config_cpu, X, Y)
        informe_cpu = RandomCloud._ejecutar_batched(motor_cpu)

        motor_gpu = MotorNube(config_gpu, X, Y)
        informe_gpu = ejecutar(motor_gpu)

        # Requirement 7.2: GPU accuracy within ±1 percentage point of batched CPU
        @test abs(informe_cpu.precision - informe_gpu.precision) <= 0.01

        # Requirement 7.3: wall-clock time reported
        @test informe_gpu.tiempo_ejecucion_ms > 0.0
        @test informe_gpu.gpu_tiempo_ms > 0.0

        # Requirement 7.4: peak GPU memory reported
        @test informe_gpu.pico_vram_mb >= 0.0

        @info "XOR GPU: cpu=$(round(informe_cpu.precision*100, digits=1))% gpu=$(round(informe_gpu.precision*100, digits=1))% | gpu_time=$(round(informe_gpu.gpu_tiempo_ms, digits=1))ms vram=$(round(informe_gpu.pico_vram_mb, digits=2))MB"
    end
end
