# =============================================================================
# Fase 2: Resultados con train/test split (80/20 estratificado)
# =============================================================================
#
# Para el paper: precisión en TEST set, no train.
# julia --project=. examples/fase2_test_set.jl

using RandomCloud
using RandomCloud: RedNeuronal, evaluar
using MLDatasets: Iris
import DataFrames
using Random
using DelimitedFiles

const SEMILLA = 42

# ─── Utilidades ───

function normalizar_minmax!(X::Matrix{Float64})
    for fila in axes(X, 1)
        mn, mx = extrema(@view X[fila, :])
        if mx > mn
            X[fila, :] .= (X[fila, :] .- mn) ./ (mx - mn)
        end
    end
end

function onehot(labels, clases)
    n = length(labels)
    nc = length(clases)
    clase_idx = Dict(c => i for (i, c) in enumerate(clases))
    Y = zeros(Float64, nc, n)
    for k in 1:n
        Y[clase_idx[labels[k]], k] = 1.0
    end
    return Y
end

function split_estratificado(labels, ratio_train=0.8; seed=SEMILLA)
    rng = MersenneTwister(seed)
    clases = sort(unique(labels))
    train_idx = Int[]
    test_idx = Int[]
    for c in clases
        idx_clase = findall(==(c), labels)
        perm = shuffle(rng, idx_clase)
        n_train = round(Int, ratio_train * length(perm))
        append!(train_idx, perm[1:n_train])
        append!(test_idx, perm[n_train+1:end])
    end
    shuffle!(rng, train_idx)
    shuffle!(rng, test_idx)
    return train_idx, test_idx
end

function contar_parametros(topo::Vector{Int})
    sum(topo[i+1] * topo[i] + topo[i+1] for i in 1:length(topo)-1)
end

function _descargar_cache(nombre, url)
    cache = ".cache_$(nombre).csv"
    if !isfile(cache)
        run(`curl -sL -o $cache $url`)
    end
    return cache
end

# ─── Datasets ───

function cargar_two_moons(; n=400, ruido=0.15)
    rng = MersenneTwister(SEMILLA)
    n_half = n ÷ 2
    theta1 = range(0, π, length=n_half)
    x1 = cos.(theta1) .+ ruido .* randn(rng, n_half)
    y1 = sin.(theta1) .+ ruido .* randn(rng, n_half)
    theta2 = range(0, π, length=n - n_half)
    x2 = 1.0 .- cos.(theta2) .+ ruido .* randn(rng, n - n_half)
    y2 = 1.0 .- sin.(theta2) .- 0.5 .+ ruido .* randn(rng, n - n_half)
    X = Float64.(hcat(vcat(x1, x2), vcat(y1, y2))')
    labels = vcat(fill(0, n_half), fill(1, n - n_half))
    clases = [0, 1]
    Y = onehot(labels, clases)
    return X, Y, labels, "Two Moons"
end

function cargar_iris()
    dataset = Iris(as_df=false)
    X = Float64.(dataset.features)
    labels = vec(dataset.targets)
    normalizar_minmax!(X)
    clases = sort(unique(labels))
    Y = onehot(labels, clases)
    return X, Y, labels, "Iris"
end

function cargar_wine()
    cache = _descargar_cache("wine", "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data")
    data = readdlm(cache, ',', Float64)
    labels = Int.(data[:, 1])
    X = Float64.(data[:, 2:end]')
    normalizar_minmax!(X)
    clases = sort(unique(labels))
    Y = onehot(labels, clases)
    return X, Y, labels, "Wine"
end

function cargar_breastcancer()
    cache = _descargar_cache("breastcancer",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data")
    lines = readlines(cache)
    n = length(lines)
    X = zeros(Float64, 30, n)
    labels = Vector{String}(undef, n)
    for (i, line) in enumerate(lines)
        parts = split(line, ',')
        labels[i] = String(parts[2])
        for j in 1:30
            X[j, i] = parse(Float64, parts[j+2])
        end
    end
    normalizar_minmax!(X)
    clases = sort(unique(labels))
    Y = onehot(labels, clases)
    return X, Y, labels, "Breast Cancer"
end

function cargar_ionosphere()
    cache = _descargar_cache("ionosphere",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data")
    lines = readlines(cache)
    n = length(lines)
    X = zeros(Float64, 34, n)
    labels = Vector{String}(undef, n)
    for (i, line) in enumerate(lines)
        parts = split(line, ',')
        labels[i] = String(parts[end])
        for j in 1:34
            X[j, i] = parse(Float64, parts[j])
        end
    end
    normalizar_minmax!(X)
    clases = sort(unique(labels))
    Y = onehot(labels, clases)
    return X, Y, labels, "Ionosphere"
end

function cargar_sonar()
    cache = _descargar_cache("sonar",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")
    lines = readlines(cache)
    n = length(lines)
    X = zeros(Float64, 60, n)
    labels = Vector{String}(undef, n)
    for (i, line) in enumerate(lines)
        parts = split(line, ',')
        labels[i] = String(parts[end])
        for j in 1:60
            X[j, i] = parse(Float64, parts[j])
        end
    end
    normalizar_minmax!(X)
    clases = sort(unique(labels))
    Y = onehot(labels, clases)
    return X, Y, labels, "Sonar"
end

# ─── Configuraciones ───

datasets_configs = [
    (cargar_two_moons,    [2, 32, 16, 2],      50, 0.6, 2, 500, 0.1),
    (cargar_iris,         [4, 16, 8, 3],        50, 0.4, 1, 200, 0.1),
    (cargar_wine,         [13, 24, 12, 3],      50, 0.5, 1, 300, 0.1),
    (cargar_breastcancer, [30, 32, 16, 2],      30, 0.7, 2, 200, 0.1),
    (cargar_ionosphere,   [34, 24, 12, 2],      30, 0.5, 1, 300, 0.1),
    (cargar_sonar,        [60, 32, 16, 2],      30, 0.5, 1, 300, 0.1),
]

# ─── Ejecución ───

println("=" ^ 95)
println("  FASE 2: Resultados en TEST SET (80/20 estratificado)")
println("=" ^ 95)
println()

resultados = []

for (cargar_fn, topo, nube_size, umbral, elim, epocas, lr) in datasets_configs
    X, Y, labels, nombre = cargar_fn()
    n_features = size(X, 1)
    n_muestras = size(X, 2)

    train_idx, test_idx = split_estratificado(labels)
    train_X, train_Y = X[:, train_idx], Y[:, train_idx]
    test_X, test_Y = X[:, test_idx], Y[:, test_idx]

    println("  $nombre: $(n_muestras) total, $(length(train_idx)) train, $(length(test_idx)) test")

    # Fase 1 sola
    config_f1 = ConfiguracionNube(
        tamano_nube=nube_size, topologia_inicial=topo,
        umbral_acierto=umbral, neuronas_eliminar=elim,
        epocas_refinamiento=epocas, tasa_aprendizaje=lr,
        semilla=SEMILLA, explorar_estructura=false
    )
    motor_f1 = MotorNube(config_f1, train_X, train_Y)
    inf_f1 = ejecutar(motor_f1)

    # Evaluar en test
    acc_test_f1 = 0.0
    topo_f1 = inf_f1.topologia_final
    params_f1 = 0
    if inf_f1.mejor_red !== nothing
        acts_f1 = RandomCloud.activaciones_por_capa(length(inf_f1.mejor_red.pesos), config_f1.activacion)
        acc_test_f1 = evaluar(inf_f1.mejor_red, test_X, test_Y; acts=acts_f1)
        params_f1 = contar_parametros(topo_f1)
    end

    # Fase 1 + Fase 2
    config_f2 = ConfiguracionNube(
        tamano_nube=nube_size, topologia_inicial=topo,
        umbral_acierto=umbral, neuronas_eliminar=elim,
        epocas_refinamiento=epocas, tasa_aprendizaje=lr,
        semilla=SEMILLA, explorar_estructura=true,
        max_profundidad_split=2, ancho_minimo_split=4,
        n_candidatos_estructura=5
    )
    motor_f2 = MotorNube(config_f2, train_X, train_Y)
    inf_f2 = ejecutar(motor_f2)

    acc_test_f2 = 0.0
    topo_f2 = inf_f2.topologia_final
    params_f2 = 0
    op = "-"
    if inf_f2.mejor_red !== nothing
        acts_f2 = RandomCloud.activaciones_por_capa(length(inf_f2.mejor_red.pesos), config_f2.activacion)
        acc_test_f2 = evaluar(inf_f2.mejor_red, test_X, test_Y; acts=acts_f2)
        params_f2 = contar_parametros(topo_f2)
    end
    if inf_f2.fase2_resultado !== nothing && inf_f2.fase2_resultado.operacion !== nothing
        op = string(inf_f2.fase2_resultado.operacion)
    end

    diff = (acc_test_f2 - acc_test_f1) * 100
    println("    Fase 1:   $(round(acc_test_f1*100, digits=1))% test | $(topo_f1) | $(params_f1) params")
    println("    Fase 1+2: $(round(acc_test_f2*100, digits=1))% test | $(topo_f2) | $(params_f2) params | $op")
    println("    Δ = $(diff >= 0 ? "+" : "")$(round(diff, digits=1)) pp")
    println()

    push!(resultados, (nombre=nombre, acc_f1=round(acc_test_f1*100, digits=1),
                       acc_f2=round(acc_test_f2*100, digits=1), diff=round(diff, digits=1),
                       topo_f1=topo_f1, topo_f2=topo_f2, params_f1=params_f1,
                       params_f2=params_f2, op=op))
end

# ─── Tabla resumen ───

println("=" ^ 95)
println("  TABLA PARA EL PAPER (test accuracy)")
println("=" ^ 95)
println()
println("  Dataset          | Phase 1 (test) | Phase 1+2 (test) | Δ      | Params F1 | Params F1+2 | Op")
println("  -----------------+----------------+------------------+--------+-----------+-------------+----------")
for r in resultados
    d = r.diff >= 0 ? "+$(r.diff)" : "$(r.diff)"
    println("  $(rpad(r.nombre, 17))| $(rpad("$(r.acc_f1)%", 15))| $(rpad("$(r.acc_f2)%", 17))| $(rpad(d, 7))| $(rpad(r.params_f1, 10))| $(rpad(r.params_f2, 12))| $(r.op)")
end
println()
