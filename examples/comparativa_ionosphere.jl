# =============================================================================
# Comparativa: Nube Aleatoria vs Clásico en Ionosphere
# =============================================================================
#
# Ejecutar con:
#   julia --project=. -t auto examples/comparativa_ionosphere.jl
#
# Ionosphere: 351 muestras, 34 features, 2 clases (good/bad radar returns).
# Clásico en papers de NAS y poda de redes neuronales.
# =============================================================================

using RandomCloud
using RandomCloud: RedNeuronal, entrenar!, EntrenarBuffers, evaluar
using Random
using Downloads: download

const SEMILLA = 42
const EPOCAS = 200
const LR = 0.1

# --- Descargar y parsear ---
println("Descargando Ionosphere...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
cache = joinpath(@__DIR__, "..", ".cache_ionosphere.csv")

if !isfile(cache)
    download(url, cache)
    println("  Descargado")
else
    println("  Cache local")
end

lines = filter(l -> !isempty(strip(l)), readlines(cache))
n_total = length(lines)
features = zeros(Float64, 34, n_total)
labels = zeros(Int, n_total)

for (k, line) in enumerate(lines)
    parts = split(line, ',')
    for j in 1:34
        features[j, k] = parse(Float64, parts[j])
    end
    labels[k] = parts[35] == "g" ? 1 : 0  # good=1, bad=0
end


# Normalizar a [0,1]
for fila in 1:34
    mn, mx = extrema(@view features[fila, :])
    if mx > mn
        features[fila, :] .= (features[fila, :] .- mn) ./ (mx - mn)
    end
end

# One-hot
objetivos = zeros(Float64, 2, n_total)
for k in 1:n_total
    objetivos[labels[k] + 1, k] = 1.0
end

pct_good = round(sum(labels) / n_total * 100, digits=1)
println("  Muestras: $n_total | Features: 34 | Clases: 2 (good/bad)")
println("  Distribución: $(pct_good)% good, $(round(100-pct_good, digits=1))% bad")
println()

# --- Train/Test split 80/20 estratificado ---
rng = MersenneTwister(SEMILLA)
train_idx = Int[]
test_idx = Int[]
for c in [0, 1]
    idx_c = findall(==(c), labels)
    perm = shuffle(rng, idx_c)
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
function contar_parametros(topo::Vector{Int})
    sum(topo[i+1] * topo[i] + topo[i+1] for i in 1:length(topo)-1)
end

function entrenar_clasica(topo, X, Y, epocas, lr, semilla)
    rng = MersenneTwister(semilla)
    red = RedNeuronal(topo, rng)
    bufs = EntrenarBuffers(topo)
    n = size(X, 2)
    t0 = time_ns()
    for _ in 1:epocas
        @inbounds for k in 1:n
            entrenar!(red, @view(X[:, k]), @view(Y[:, k]), lr, bufs)
        end
    end
    t1 = time_ns()
    return red, (t1 - t0) / 1_000_000.0
end


# --- Comparativa ---
topologias = [
    [34, 8, 2],
    [34, 16, 2],
    [34, 16, 8, 2],
    [34, 32, 16, 2],
    [34, 32, 16, 8, 2],
]

println("=" ^ 75)
println("  COMPARATIVA: Nube Aleatoria vs Clásico — Ionosphere")
println("=" ^ 75)
println()
println("  Épocas: $EPOCAS | LR: $LR | Threads: $(Threads.nthreads())")
println("  Nube: 50 redes, umbral=0.6, eliminar=1")
println()

resultados = []

for topo in topologias
    params = contar_parametros(topo)
    println("-" ^ 75)
    println("  Topología: $topo ($params params)")
    println("-" ^ 75)

    print("  Clásico:  ")
    red_c, t_c = entrenar_clasica(topo, train_X, train_Y, EPOCAS, LR, SEMILLA)
    ct_train = evaluar(red_c, train_X, train_Y)
    ct_test = evaluar(red_c, test_X, test_Y)
    println("$(round(t_c, digits=1))ms — train=$(round(ct_train*100,digits=1))% test=$(round(ct_test*100,digits=1))%")

    semilla_nube = SEMILLA
    tiempo_total = 0.0
    for intento in 1:3
        label = intento > 1 ? "  Nube(#$intento): " : "  Nube:     "
        print(label)
        config = ConfiguracionNube(
            tamano_nube=50, topologia_inicial=topo, umbral_acierto=0.6,
            neuronas_eliminar=1, epocas_refinamiento=EPOCAS,
            tasa_aprendizaje=LR, semilla=semilla_nube
        )
        motor = MotorNube(config, train_X, train_Y)
        informe = ejecutar(motor)
        tiempo_total += informe.tiempo_ejecucion_ms

        if informe.exitoso
            nt_test = evaluar(informe.mejor_red, test_X, test_Y)
            topo_f = informe.topologia_final
            pf = contar_parametros(topo_f)
            red_pct = round((1.0 - pf / params) * 100, digits=1)
            println("$(round(informe.tiempo_ejecucion_ms, digits=1))ms (s=$semilla_nube) — train=$(round(informe.precision*100,digits=1))% test=$(round(nt_test*100,digits=1))% → $topo_f (-$(red_pct)%)")
            push!(resultados, (topo=topo, params=params, c_test=ct_test, n_test=nt_test,
                n_topo=topo_f, n_params=pf, reduccion=red_pct, exitoso=true, intentos=intento))
            break
        else
            println("$(round(informe.tiempo_ejecucion_ms, digits=1))ms (s=$semilla_nube) — no viable")
            semilla_nube += 1000
            if intento == 3
                push!(resultados, (topo=topo, params=params, c_test=ct_test, n_test=0.0,
                    n_topo=Int[], n_params=0, reduccion=0.0, exitoso=false, intentos=3))
            end
        end
    end
    println()
end

println("=" ^ 75)
println("  RESUMEN — Ionosphere")
println("=" ^ 75)
println()
for r in resultados
    ct = "$(round(r.c_test * 100, digits=1))%"
    if r.exitoso
        nt = "$(round(r.n_test * 100, digits=1))%"
        s = r.intentos > 1 ? " (#$(r.intentos))" : ""
        println("  $(rpad(string(r.topo), 22)) │ Clásico: $(rpad(ct, 7)) │ Nube: $(rpad(nt, 7)) → $(r.n_topo) (-$(r.reduccion)%)$s")
    else
        println("  $(rpad(string(r.topo), 22)) │ Clásico: $(rpad(ct, 7)) │ Nube: FAIL (3 intentos)")
    end
end
println()
println("=" ^ 75)
