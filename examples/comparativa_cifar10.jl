# =============================================================================
# Comparativa: Método de la Nube Aleatoria vs Entrenamiento Clásico en CIFAR-10
# =============================================================================
#
# Ejecutar con:
#   julia --project=. examples/comparativa_cifar10.jl
#
# CIFAR-10: 32×32×3 = 3072 entradas, 10 clases, 50K train + 10K test.
# Imágenes a color de objetos reales. Problema difícil para redes feedforward.
# =============================================================================

using RandomCloud
using RandomCloud: RedNeuronal, feedforward, entrenar!, EntrenarBuffers, evaluar
using MLDatasets: CIFAR10
using Random

const SEMILLA = 42
const EPOCAS = 30
const LR = 0.05
const N_MUESTRAS = 50_000       # dataset completo de train

println("=" ^ 70)
println("  COMPARATIVA: Nube Aleatoria vs Clásico — CIFAR-10")
println("=" ^ 70)
println()

# --- Cargar CIFAR-10 ---
println("Cargando CIFAR-10...")
dataset_train = CIFAR10(Float64, :train)
dataset_test = CIFAR10(Float64, :test)

# Test set fijo (10K)
test_raw = dataset_test[:].features                    # 32×32×3×10000
test_X = reshape(test_raw, 3072, 10_000)               # aplanar a 3072×10000
test_labels = dataset_test[:].targets
test_Y = zeros(Float64, 10, 10_000)
for k in 1:10_000
    test_Y[test_labels[k] + 1, k] = 1.0
end

# Train subconjunto
rng_idx = MersenneTwister(SEMILLA)
all_raw = dataset_train[:].features
all_features = reshape(all_raw, 3072, 50_000)
all_labels = dataset_train[:].targets
indices = randperm(rng_idx, 50_000)[1:N_MUESTRAS]

train_X = all_features[:, indices]
train_labels = all_labels[indices]
train_Y = zeros(Float64, 10, N_MUESTRAS)
for k in 1:N_MUESTRAS
    train_Y[train_labels[k] + 1, k] = 1.0
end

println("  Train: $N_MUESTRAS muestras (de 50,000)")
println("  Test:  10,000 muestras")
println("  Entradas: 3,072 (32×32×3)")
println("  Clases: 10")
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
    [3072, 64, 10],
    [3072, 64, 32, 10],
]

println("  Épocas: $EPOCAS | LR: $LR | Nube: 50 redes, umbral=0.12, eliminar=2")
println("  Threads: $(Threads.nthreads())")
println()

resultados = []

for topo in topologias
    params = contar_parametros(topo)
    println("-" ^ 70)
    println("  Topología: $topo ($(params) parámetros)")
    println("-" ^ 70)

    # Clásico
    print("  Clásico:  entrenando... ")
    red_c, t_c = entrenar_clasica(topo, train_X, train_Y, EPOCAS, LR, SEMILLA)
    prec_train_c = evaluar(red_c, train_X, train_Y)
    prec_test_c = evaluar(red_c, test_X, test_Y)
    println("$(round(t_c/1000, digits=1))s")
    println("    Train: $(round(prec_train_c*100,digits=2))%  Test: $(round(prec_test_c*100,digits=2))%")

    # Nube — con reintentos
    semilla_nube = SEMILLA
    tiempo_total_nube = 0.0
    for intento in 1:3
        label = intento > 1 ? "  Nube(#$intento): " : "  Nube:     "
        print(label * "ejecutando... ")
        config = ConfiguracionNube(
            tamano_nube = 50,
            topologia_inicial = topo,
            umbral_acierto = 0.12,
            neuronas_eliminar = 2,
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
            println("$(round(informe.tiempo_ejecucion_ms/1000, digits=1))s (semilla=$semilla_nube)")
            println("    Train: $(round(informe.precision*100,digits=2))%  Test: $(round(prec_test_n*100,digits=2))%")
            println("    Topo:  $topo_f ($params_f params, -$(red_pct)%)")

            push!(resultados, (topo=topo, params=params,
                c_train=prec_train_c, c_test=prec_test_c, c_tiempo=t_c,
                n_train=informe.precision, n_test=prec_test_n, n_tiempo=tiempo_total_nube,
                n_topo=topo_f, n_params=params_f, reduccion=red_pct, exitoso=true, intentos=intento))
            break
        else
            println("$(round(informe.tiempo_ejecucion_ms/1000, digits=1))s (semilla=$semilla_nube) — no viable")
            semilla_nube += 1000
            if intento == 3
                push!(resultados, (topo=topo, params=params,
                    c_train=prec_train_c, c_test=prec_test_c, c_tiempo=t_c,
                    n_train=0.0, n_test=0.0, n_tiempo=tiempo_total_nube,
                    n_topo=Int[], n_params=0, reduccion=0.0, exitoso=false, intentos=3))
            end
        end
    end
    println()
end

# --- Resumen ---
println("=" ^ 70)
println("  RESUMEN")
println("=" ^ 70)
println()
for r in resultados
    ct = "$(round(r.c_test * 100, digits=1))%"
    if r.exitoso
        nt = "$(round(r.n_test * 100, digits=1))%"
        s = r.intentos > 1 ? " (intento #$(r.intentos))" : ""
        println("  $(rpad(string(r.topo), 22)) │ Clásico: $(rpad(ct, 7)) │ Nube: $(rpad(nt, 7)) → $(r.n_topo) (-$(r.reduccion)%)$s")
    else
        println("  $(rpad(string(r.topo), 22)) │ Clásico: $(rpad(ct, 7)) │ Nube: FAIL (3 intentos)")
    end
end
println()
println("=" ^ 70)
