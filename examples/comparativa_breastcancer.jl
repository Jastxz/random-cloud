# =============================================================================
# Comparativa: Nube Aleatoria vs Clásico en Breast Cancer Wisconsin
# =============================================================================
#
# Ejecutar con:
#   julia --project=. -t auto examples/comparativa_breastcancer.jl
#
# Breast Cancer Wisconsin (Diagnostic): 569 muestras, 30 features, 2 clases.
# Datos descargados del UCI ML Repository.
# =============================================================================

using RandomCloud
using RandomCloud: RedNeuronal, feedforward, entrenar!, EntrenarBuffers, evaluar
using Random
using Downloads: download

const SEMILLA = 42
const EPOCAS = 100
const LR = 0.1


# --- Descargar y parsear datos ---
println("Descargando Breast Cancer Wisconsin...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
cache_path = joinpath(@__DIR__, "..", ".cache_breastcancer.csv")

if !isfile(cache_path)
    download(url, cache_path)
    println("  Descargado y cacheado en $cache_path")
else
    println("  Usando cache local")
end

# Parsear CSV: ID, Diagnosis (M/B), 30 features
lines = readlines(cache_path)
n_total = length(lines)
features = zeros(Float64, 30, n_total)
labels = zeros(Int, n_total)

for (k, line) in enumerate(lines)
    parts = split(line, ',')
    labels[k] = parts[2] == "M" ? 1 : 0   # Maligno=1, Benigno=0
    for j in 1:30
        features[j, k] = parse(Float64, parts[j + 2])
    end
end

# Normalizar features a [0,1]
for fila in 1:30
    mn, mx = extrema(@view features[fila, :])
    if mx > mn
        features[fila, :] .= (features[fila, :] .- mn) ./ (mx - mn)
    end
end

# One-hot: 2 clases
objetivos = zeros(Float64, 2, n_total)
for k in 1:n_total
    objetivos[labels[k] + 1, k] = 1.0
end

println("  Muestras: $n_total | Features: 30 | Clases: 2 (B/M)")
println("  Distribución: $(sum(labels .== 0)) benigno, $(sum(labels .== 1)) maligno")
println()


# --- Train/Test split 80/20 estratificado ---
rng = MersenneTwister(SEMILLA)
train_idx = Int[]
test_idx = Int[]
for c in [0, 1]
    idx_clase = findall(==(c), labels)
    perm = shuffle(rng, idx_clase)
    n_train = round(Int, 0.8 * length(perm))
    append!(train_idx, perm[1:n_train])
    append!(test_idx, perm[n_train+1:end])
end
shuffle!(rng, train_idx)
shuffle!(rng, test_idx)

train_X = features[:, train_idx]
train_Y = objetivos[:, train_idx]
test_X = features[:, test_idx]
test_Y = objetivos[:, test_idx]

println("  Train: $(length(train_idx)) | Test: $(length(test_idx))")
println()

# --- Funciones auxiliares ---
function contar_parametros(topologia::Vector{Int})
    sum(topologia[i+1] * topologia[i] + topologia[i+1] for i in 1:length(topologia)-1)
end

function entrenar_clasica(topologia, entradas, objetivos, epocas, lr, semilla)
    rng = MersenneTwister(semilla)
    red = RedNeuronal(topologia, rng)
    bufs = EntrenarBuffers(topologia)
    n = size(entradas, 2)
    t0 = time_ns()
    for _ in 1:epocas
        @inbounds for k in 1:n
            entrenar!(red, @view(entradas[:, k]), @view(objetivos[:, k]), lr, bufs)
        end
    end
    t1 = time_ns()
    return red, (t1 - t0) / 1_000_000.0
end


# --- Comparativa ---
topologias = [
    [30, 8, 2],
    [30, 16, 2],
    [30, 16, 8, 2],
    [30, 32, 16, 2],
]

println("=" ^ 70)
println("  COMPARATIVA: Nube Aleatoria vs Clásico — Breast Cancer")
println("=" ^ 70)
println()
println("  Épocas: $EPOCAS | LR: $LR | Threads: $(Threads.nthreads())")
println("  Nube: 50 redes, umbral=0.7, eliminar=1")
println()

resultados = []

for topo in topologias
    params = contar_parametros(topo)
    println("-" ^ 70)
    println("  Topología: $topo ($params parámetros)")
    println("-" ^ 70)

    print("  Clásico:  ")
    red_c, t_c = entrenar_clasica(topo, train_X, train_Y, EPOCAS, LR, SEMILLA)
    prec_train_c = evaluar(red_c, train_X, train_Y)
    prec_test_c = evaluar(red_c, test_X, test_Y)
    println("$(round(t_c, digits=1))ms — train=$(round(prec_train_c*100,digits=1))% test=$(round(prec_test_c*100,digits=1))%")

    # Nube con reintentos
    semilla_nube = SEMILLA
    tiempo_total = 0.0
    for intento in 1:3
        label = intento > 1 ? "  Nube(#$intento): " : "  Nube:     "
        print(label)
        config = ConfiguracionNube(
            tamano_nube = 50,
            topologia_inicial = topo,
            umbral_acierto = 0.7,
            neuronas_eliminar = 1,
            epocas_refinamiento = EPOCAS,
            tasa_aprendizaje = LR,
            semilla = semilla_nube
        )
        motor = MotorNube(config, train_X, train_Y)
        informe = ejecutar(motor)
        tiempo_total += informe.tiempo_ejecucion_ms

        if informe.exitoso
            prec_test_n = evaluar(informe.mejor_red, test_X, test_Y)
            topo_f = informe.topologia_final
            params_f = contar_parametros(topo_f)
            red_pct = round((1.0 - params_f / params) * 100, digits=1)
            println("$(round(informe.tiempo_ejecucion_ms, digits=1))ms (s=$semilla_nube) — train=$(round(informe.precision*100,digits=1))% test=$(round(prec_test_n*100,digits=1))% → $topo_f (-$(red_pct)%)")
            push!(resultados, (topo=topo, params=params,
                c_train=prec_train_c, c_test=prec_test_c,
                n_train=informe.precision, n_test=prec_test_n,
                n_topo=topo_f, n_params=params_f, reduccion=red_pct, exitoso=true, intentos=intento))
            break
        else
            println("$(round(informe.tiempo_ejecucion_ms, digits=1))ms (s=$semilla_nube) — no viable")
            semilla_nube += 1000
            if intento == 3
                push!(resultados, (topo=topo, params=params,
                    c_train=prec_train_c, c_test=prec_test_c,
                    n_train=0.0, n_test=0.0,
                    n_topo=Int[], n_params=0, reduccion=0.0, exitoso=false, intentos=3))
            end
        end
    end
    println()
end

println("=" ^ 70)
println("  RESUMEN")
println("=" ^ 70)
println()
for r in resultados
    ct = "$(round(r.c_test * 100, digits=1))%"
    if r.exitoso
        nt = "$(round(r.n_test * 100, digits=1))%"
        s = r.intentos > 1 ? " (#$(r.intentos))" : ""
        println("  $(rpad(string(r.topo), 20)) │ Clásico: $(rpad(ct, 7)) │ Nube: $(rpad(nt, 7)) → $(r.n_topo) (-$(r.reduccion)%)$s")
    else
        println("  $(rpad(string(r.topo), 20)) │ Clásico: $(rpad(ct, 7)) │ Nube: FAIL (3 intentos)")
    end
end
println()
println("=" ^ 70)
