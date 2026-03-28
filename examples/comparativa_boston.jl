# =============================================================================
# Comparativa: Nube Aleatoria vs Clásico en Boston Housing (Regresión)
# =============================================================================
#
# Ejecutar con:
#   julia --project=. -t auto examples/comparativa_boston.jl
#
# Boston Housing: 506 muestras, 13 features, regresión (precio vivienda).
# Primer test del método con regresión usando evaluar_regresion (R²).
# =============================================================================

using RandomCloud
using RandomCloud: RedNeuronal, entrenar!, EntrenarBuffers, evaluar_regresion
using MLDatasets: BostonHousing
import DataFrames
using Random

const SEMILLA = 42
const EPOCAS = 500
const LR = 0.01

# --- Cargar datos ---
println("Cargando Boston Housing...")
dataset = BostonHousing(as_df=false)
features = Float64.(dataset.features)    # 13×506
targets = Float64.(dataset.targets)      # 1×506

n_muestras = size(features, 2)


# Normalizar features a [0,1]
for fila in 1:size(features, 1)
    mn, mx = extrema(@view features[fila, :])
    if mx > mn
        features[fila, :] .= (features[fila, :] .- mn) ./ (mx - mn)
    end
end

# Normalizar targets a [0,1] (para que sigmoid de salida sea compatible)
t_min, t_max = extrema(targets)
targets_norm = (targets .- t_min) ./ (t_max - t_min)

# Reshape targets a matriz 1×N
objetivos = reshape(targets_norm, 1, n_muestras)

println("  Muestras: $n_muestras | Features: 13 | Target: precio (normalizado)")
println()

# --- Train/Test split 80/20 ---
rng = MersenneTwister(SEMILLA)
perm = randperm(rng, n_muestras)
n_train = round(Int, 0.8 * n_muestras)
train_idx = perm[1:n_train]
test_idx = perm[n_train+1:end]

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
    [13, 8, 1],
    [13, 16, 1],
    [13, 16, 8, 1],
    [13, 32, 16, 1],
    [13, 64, 32, 1],
    [13, 64, 32, 16, 1],
    [13, 128, 64, 1],
    [13, 128, 64, 32, 1],
]

println("=" ^ 70)
println("  COMPARATIVA: Nube Aleatoria vs Clásico — Boston Housing (R²)")
println("=" ^ 70)
println()
println("  Épocas: $EPOCAS | LR: $LR | Threads: $(Threads.nthreads())")
println("  Nube: 100 redes, umbral=0.05, eliminar=1")
println("  Métrica: R² (coeficiente de determinación)")
println()

resultados = []

for topo in topologias
    params = contar_parametros(topo)
    println("-" ^ 70)
    println("  Topología: $topo ($params parámetros)")
    println("-" ^ 70)

    # Clásico
    print("  Clásico:  ")
    red_c, t_c = entrenar_clasica(topo, train_X, train_Y, EPOCAS, LR, SEMILLA)
    r2_train_c = evaluar_regresion(red_c, train_X, train_Y)
    r2_test_c = evaluar_regresion(red_c, test_X, test_Y)
    println("$(round(t_c, digits=1))ms — R² train=$(round(r2_train_c, digits=4)) test=$(round(r2_test_c, digits=4))")

    # Nube con reintentos — usa evaluar_regresion
    semilla_nube = SEMILLA
    tiempo_total = 0.0
    for intento in 1:3
        label = intento > 1 ? "  Nube(#$intento): " : "  Nube:     "
        print(label)
        config = ConfiguracionNube(
            tamano_nube = 100,
            topologia_inicial = topo,
            umbral_acierto = 0.05,
            neuronas_eliminar = 1,
            epocas_refinamiento = EPOCAS,
            tasa_aprendizaje = LR,
            semilla = semilla_nube
        )
        motor = MotorNube(config, train_X, train_Y, evaluar_regresion)
        informe = ejecutar(motor)
        tiempo_total += informe.tiempo_ejecucion_ms

        if informe.exitoso
            r2_test_n = evaluar_regresion(informe.mejor_red, test_X, test_Y)
            topo_f = informe.topologia_final
            params_f = contar_parametros(topo_f)
            red_pct = round((1.0 - params_f / params) * 100, digits=1)
            println("$(round(informe.tiempo_ejecucion_ms, digits=1))ms (s=$semilla_nube) — R² train=$(round(informe.precision, digits=4)) test=$(round(r2_test_n, digits=4)) → $topo_f (-$(red_pct)%)")
            push!(resultados, (topo=topo, params=params,
                c_r2_train=r2_train_c, c_r2_test=r2_test_c,
                n_r2_train=informe.precision, n_r2_test=r2_test_n,
                n_topo=topo_f, n_params=params_f, reduccion=red_pct, exitoso=true, intentos=intento))
            break
        else
            println("$(round(informe.tiempo_ejecucion_ms, digits=1))ms (s=$semilla_nube) — no viable")
            semilla_nube += 1000
            if intento == 3
                push!(resultados, (topo=topo, params=params,
                    c_r2_train=r2_train_c, c_r2_test=r2_test_c,
                    n_r2_train=0.0, n_r2_test=0.0,
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
    ct = "$(round(r.c_r2_test, digits=3))"
    if r.exitoso
        nt = "$(round(r.n_r2_test, digits=3))"
        s = r.intentos > 1 ? " (#$(r.intentos))" : ""
        println("  $(rpad(string(r.topo), 20)) │ Clásico R²: $(rpad(ct, 6)) │ Nube R²: $(rpad(nt, 6)) → $(r.n_topo) (-$(r.reduccion)%)$s")
    else
        println("  $(rpad(string(r.topo), 20)) │ Clásico R²: $(rpad(ct, 6)) │ Nube: FAIL (3 intentos)")
    end
end
println()
println("=" ^ 70)
