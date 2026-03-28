# =============================================================================
# Comparativa: Método de la Nube Aleatoria vs Entrenamiento Clásico en Iris
# =============================================================================
#
# Ejecutar con:
#   julia --project=. examples/comparativa_iris.jl
#
# Iris: 150 muestras, 4 features, 3 clases (setosa, versicolor, virginica).
# Dataset pequeño → ideal para el método de la nube.
# =============================================================================

using RandomCloud
using RandomCloud: RedNeuronal, feedforward, entrenar!, EntrenarBuffers, evaluar
using MLDatasets: Iris
import DataFrames
using Random

const SEMILLA = 42
const EPOCAS = 100
const LR = 0.1

# --- Cargar Iris ---
println("Cargando Iris...")
dataset = Iris(as_df=false)
features = Float64.(dataset.features)   # 4×150
labels = vec(dataset.targets)            # Vector de strings

# Normalizar features a [0,1] por columna (min-max)
for fila in 1:size(features, 1)
    mn, mx = extrema(@view features[fila, :])
    if mx > mn
        features[fila, :] .= (features[fila, :] .- mn) ./ (mx - mn)
    end
end

# One-hot encoding: "Iris-setosa"→1, "Iris-versicolor"→2, "Iris-virginica"→3
clases = sort(unique(labels))
clase_idx = Dict(c => i for (i, c) in enumerate(clases))
n_muestras = size(features, 2)
objetivos = zeros(Float64, 3, n_muestras)
for k in 1:n_muestras
    objetivos[clase_idx[labels[k]], k] = 1.0
end

println("  Features: $(size(features)) — 4 atributos × $n_muestras muestras")
println("  Clases:   $(join(clases, ", "))")
println()


# --- Train/Test split (80/20) estratificado ---
rng = MersenneTwister(SEMILLA)
train_idx = Int[]
test_idx = Int[]
for c in clases
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

println("  Train: $(size(train_X, 2)) muestras")
println("  Test:  $(size(test_X, 2)) muestras")
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


# --- Comparativa con distintas topologías ---
topologias = [
    [4, 8, 3],
    [4, 8, 4, 3],
    [4, 16, 8, 3],
    [4, 16, 8, 4, 3],
]

println("=" ^ 80)
println("  COMPARATIVA: Nube Aleatoria vs Clásico — Iris")
println("=" ^ 80)
println()
println("  Épocas: $EPOCAS | LR: $LR | Nube: 50 redes, umbral=0.4, eliminar=1")
println("  Train: $(size(train_X, 2)) muestras | Test: $(size(test_X, 2)) muestras")
println()

resultados = []

for topo in topologias
    params = contar_parametros(topo)
    println("-" ^ 80)
    println("  Topología: $topo ($params parámetros)")
    println("-" ^ 80)

    # Clásico
    print("  Clásico:  ")
    red_c, t_c = entrenar_clasica(topo, train_X, train_Y, EPOCAS, LR, SEMILLA)
    prec_train_c = evaluar(red_c, train_X, train_Y)
    prec_test_c = evaluar(red_c, test_X, test_Y)
    println("$(round(t_c, digits=1))ms — train=$(round(prec_train_c*100,digits=1))% test=$(round(prec_test_c*100,digits=1))%")

    # Nube — con política de reintentos (hasta 3 semillas distintas)
    informe = nothing
    semilla_nube = SEMILLA
    tiempo_total_nube = 0.0
    for intento in 1:3
        if intento > 1
            print("  Nube(#$intento): ")
        else
            print("  Nube:     ")
        end
        config = ConfiguracionNube(
            tamano_nube = 50,
            topologia_inicial = topo,
            umbral_acierto = 0.4,
            neuronas_eliminar = 1,
            epocas_refinamiento = EPOCAS,
            tasa_aprendizaje = LR,
            semilla = semilla_nube
        )
        motor = MotorNube(config, train_X, train_Y)
        informe = ejecutar(motor)
        tiempo_total_nube += informe.tiempo_ejecucion_ms

        if informe.exitoso
            prec_test_n = evaluar(informe.mejor_red, test_X, test_Y)
            topo_f = informe.topologia_final
            params_f = contar_parametros(topo_f)
            red_pct = round((1.0 - params_f / params) * 100, digits=1)
            println("$(round(informe.tiempo_ejecucion_ms, digits=1))ms (semilla=$semilla_nube) — train=$(round(informe.precision*100,digits=1))% test=$(round(prec_test_n*100,digits=1))% → $topo_f ($params_f params, -$(red_pct)%)")

            push!(resultados, (topo=topo, params=params,
                c_train=prec_train_c, c_test=prec_test_c, c_tiempo=t_c,
                n_train=informe.precision, n_test=prec_test_n, n_tiempo=tiempo_total_nube,
                n_topo=topo_f, n_params=params_f, reduccion=red_pct, exitoso=true,
                intentos=intento))
            break
        else
            println("$(round(informe.tiempo_ejecucion_ms, digits=1))ms (semilla=$semilla_nube) — no viable")
            semilla_nube += 1000  # siguiente semilla
            if intento == 3
                push!(resultados, (topo=topo, params=params,
                    c_train=prec_train_c, c_test=prec_test_c, c_tiempo=t_c,
                    n_train=0.0, n_test=0.0, n_tiempo=tiempo_total_nube,
                    n_topo=Int[], n_params=0, reduccion=0.0, exitoso=false,
                    intentos=3))
            end
        end
    end
    println()
end

# --- Tabla resumen ---
println("=" ^ 80)
println("  RESUMEN")
println("=" ^ 80)
println()

for r in resultados
    ct = "$(round(r.c_test * 100, digits=1))%"
    if r.exitoso
        nt = "$(round(r.n_test * 100, digits=1))%"
        intentos_str = r.intentos > 1 ? " (intento #$(r.intentos))" : ""
        println("  $(rpad(string(r.topo), 20)) │ Clásico: $(rpad(ct, 7)) │ Nube: $(rpad(nt, 7)) → $(r.n_topo) (-$(r.reduccion)%)$intentos_str")
    else
        println("  $(rpad(string(r.topo), 20)) │ Clásico: $(rpad(ct, 7)) │ Nube: FAIL (3 intentos)")
    end
end

println()
println("=" ^ 80)
