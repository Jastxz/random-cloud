# =============================================================================
# Comparativa completa: Fase 1 sola vs Fase 1 + Fase 2 en múltiples datasets
# =============================================================================
#
# Ejecutar con:
#   julia --project=. examples/comparativa_fase2_completa.jl
#
# Datasets: Two Moons, Iris, Wine, Breast Cancer, Ionosphere, Sonar
# =============================================================================

using RandomCloud
using RandomCloud: RedNeuronal, evaluar
using MLDatasets: Iris
import DataFrames
using Random
using DelimitedFiles

const SEMILLA = 42

# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────

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

function contar_parametros(topologia::Vector{Int})
    sum(topologia[i+1] * topologia[i] + topologia[i+1] for i in 1:length(topologia)-1)
end

function _descargar_cache(nombre, url)
    cache = ".cache_$(nombre).csv"
    if !isfile(cache)
        run(`curl -sL -o $cache $url`)
    end
    return cache
end

# ─────────────────────────────────────────────────────────────────────────────
# Carga de datasets
# ─────────────────────────────────────────────────────────────────────────────

function cargar_two_moons(; n=300, ruido=0.15)
    rng = MersenneTwister(SEMILLA)
    n_half = n ÷ 2
    theta1 = range(0, π, length=n_half)
    x1 = cos.(theta1) .+ ruido .* randn(rng, n_half)
    y1 = sin.(theta1) .+ ruido .* randn(rng, n_half)
    theta2 = range(0, π, length=n - n_half)
    x2 = 1.0 .- cos.(theta2) .+ ruido .* randn(rng, n - n_half)
    y2 = 1.0 .- sin.(theta2) .- 0.5 .+ ruido .* randn(rng, n - n_half)
    X = hcat(vcat(x1, x2), vcat(y1, y2))'
    labels = vcat(zeros(Int, n_half), ones(Int, n - n_half))
    Y = zeros(2, n)
    for i in 1:n
        Y[labels[i] + 1, i] = 1.0
    end
    return Float64.(X), Y, "Two Moons"
end

function cargar_iris()
    dataset = Iris(as_df=false)
    X = Float64.(dataset.features)
    labels = vec(dataset.targets)
    normalizar_minmax!(X)
    clases = sort(unique(labels))
    Y = onehot(labels, clases)
    return X, Y, "Iris"
end

function cargar_wine()
    cache = _descargar_cache("wine", "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data")
    data = readdlm(cache, ',', Float64)
    labels = Int.(data[:, 1])
    X = data[:, 2:end]'
    X = Float64.(X)
    normalizar_minmax!(X)
    clases = sort(unique(labels))
    Y = onehot(labels, clases)
    return X, Y, "Wine"
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
    return X, Y, "Breast Cancer"
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
    return X, Y, "Ionosphere"
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
    return X, Y, "Sonar"
end

# ─────────────────────────────────────────────────────────────────────────────
# Configuraciones por dataset (topologías con 2-3 capas ocultas)
# ─────────────────────────────────────────────────────────────────────────────

datasets_configs = [
    (cargar_two_moons,    [2, 32, 16, 2],      50, 0.6, 2, 500, 0.1),
    (cargar_iris,         [4, 16, 8, 3],        50, 0.4, 1, 200, 0.1),
    (cargar_wine,         [13, 24, 12, 3],      50, 0.5, 1, 300, 0.1),
    (cargar_breastcancer, [30, 32, 16, 2],      30, 0.7, 2, 200, 0.1),
    (cargar_ionosphere,   [34, 24, 12, 2],      30, 0.5, 1, 300, 0.1),
    (cargar_sonar,        [60, 32, 16, 2],      30, 0.5, 1, 300, 0.1),
]

# ─────────────────────────────────────────────────────────────────────────────
# Ejecución
# ─────────────────────────────────────────────────────────────────────────────

println("=" ^ 90)
println("  COMPARATIVA: Fase 1 sola vs Fase 1 + Fase 2 (exploración estructural)")
println("=" ^ 90)
println()

resultados = []

for (cargar_fn, topo, nube_size, umbral, elim, epocas, lr) in datasets_configs
    X, Y, nombre = cargar_fn()
    n_features = size(X, 1)
    n_muestras = size(X, 2)
    n_clases = size(Y, 1)

    println("─" ^ 90)
    println("  $nombre: $(n_muestras) muestras, $(n_features) features, $(n_clases) clases")
    println("  Topología: $topo | Nube: $nube_size | Umbral: $umbral | Épocas: $epocas")
    println("─" ^ 90)

    # Fase 1 sola
    config_f1 = ConfiguracionNube(
        tamano_nube=nube_size, topologia_inicial=topo,
        umbral_acierto=umbral, neuronas_eliminar=elim,
        epocas_refinamiento=epocas, tasa_aprendizaje=lr,
        semilla=SEMILLA, explorar_estructura=false
    )
    motor_f1 = MotorNube(config_f1, X, Y)
    inf_f1 = ejecutar(motor_f1)

    acc_f1 = round(inf_f1.precision * 100, digits=1)
    topo_f1 = inf_f1.topologia_final
    params_f1 = topo_f1 !== nothing ? contar_parametros(topo_f1) : 0
    println("  Fase 1:   $(inf_f1.exitoso ? "✓" : "✗")  $(acc_f1)%  topo=$(topo_f1)  params=$(params_f1)  $(round(inf_f1.tiempo_ejecucion_ms, digits=0))ms")

    # Fase 1 + Fase 2
    config_f2 = ConfiguracionNube(
        tamano_nube=nube_size, topologia_inicial=topo,
        umbral_acierto=umbral, neuronas_eliminar=elim,
        epocas_refinamiento=epocas, tasa_aprendizaje=lr,
        semilla=SEMILLA, explorar_estructura=true,
        max_profundidad_split=2, ancho_minimo_split=4,
        n_candidatos_estructura=5
    )
    motor_f2 = MotorNube(config_f2, X, Y)
    inf_f2 = ejecutar(motor_f2)

    acc_f2 = round(inf_f2.precision * 100, digits=1)
    topo_f2 = inf_f2.topologia_final
    params_f2 = topo_f2 !== nothing ? contar_parametros(topo_f2) : 0
    mejoro = inf_f2.fase2_mejoro
    op = inf_f2.fase2_resultado !== nothing ? string(inf_f2.fase2_resultado.operacion) : "-"
    n_cands = inf_f2.fase2_resultado !== nothing ? inf_f2.fase2_resultado.candidatos_evaluados : 0

    println("  Fase 1+2: $(inf_f2.exitoso ? "✓" : "✗")  $(acc_f2)%  topo=$(topo_f2)  params=$(params_f2)  $(round(inf_f2.tiempo_ejecucion_ms, digits=0))ms")
    println("  → Mejoró: $(mejoro)  Op: $(op)  Candidatos: $(n_cands)")

    diff = acc_f2 - acc_f1
    println("  → Δ: $(diff >= 0 ? "+" : "")$(round(diff, digits=1)) pp")
    println()

    push!(resultados, (nombre=nombre, acc_f1=acc_f1, acc_f2=acc_f2, diff=diff,
                       topo_f1=topo_f1, topo_f2=topo_f2, params_f1=params_f1,
                       params_f2=params_f2, mejoro=mejoro, op=op))
end

# ─────────────────────────────────────────────────────────────────────────────
# Tabla resumen
# ─────────────────────────────────────────────────────────────────────────────

println("=" ^ 90)
println("  RESUMEN")
println("=" ^ 90)
println()
println("  $(rpad("Dataset", 16)) │ $(rpad("Fase 1", 8)) │ $(rpad("Fase 1+2", 8)) │ $(rpad("Δ", 8)) │ Mejoró │ Operación")
println("  $("─"^16)─┼─$("─"^8)─┼─$("─"^8)─┼─$("─"^8)─┼─$("─"^6)─┼─$("─"^12)")
for r in resultados
    d = r.diff >= 0 ? "+$(r.diff)" : "$(r.diff)"
    m = r.mejoro ? "  SI  " : "  no  "
    println("  $(rpad(r.nombre, 16)) | $(rpad("$(r.acc_f1)%", 8)) | $(rpad("$(r.acc_f2)%", 8)) | $(rpad(d, 8)) |$m| $(r.op)")
end
println()

n_mejoras = count(r -> r.mejoro, resultados)
println("  Fase 2 mejoró en $n_mejoras/$(length(resultados)) datasets.")
println("=" ^ 90)
