# =============================================================================
# Comparativa: Método de la Nube Aleatoria vs Entrenamiento Clásico en MNIST
# =============================================================================
#
# Ejecutar con:
#   julia --project=. examples/comparativa_mnist.jl
#
# Compara ambos enfoques con la MISMA topología y MISMAS épocas de
# entrenamiento, variando la cantidad de datos: 1K, 5K, 10K, 30K, 60K.
# =============================================================================

using RandomCloud
using RandomCloud: RedNeuronal, feedforward, entrenar!, EntrenarBuffers, evaluar
using MLDatasets: MNIST
using Random

# --- Parámetros globales ---
const TOPOLOGIA = [784, 32, 16, 10]
const EPOCAS = 30                # épocas de entrenamiento/refinamiento
const LR = 0.1
const SEMILLA = 42
const TAMANOS = [1_000, 5_000, 10_000, 30_000, 60_000]

# Parámetros del método de la nube
const TAMANO_NUBE = 100
const UMBRAL = 0.15
const NEURONAS_ELIMINAR = 2


# --- Cargar MNIST completo ---
println("Cargando MNIST...")
dataset_train = MNIST(Float64, :train)
dataset_test = MNIST(Float64, :test)

# Test set fijo (10K imágenes) para evaluar generalización
test_features = reshape(dataset_test[:].features, 784, 10_000)
test_labels = dataset_test[:].targets
test_objetivos = zeros(Float64, 10, 10_000)
for k in 1:10_000
    test_objetivos[test_labels[k] + 1, k] = 1.0
end

# Train set completo (60K) — seleccionaremos subconjuntos
all_features = reshape(dataset_train[:].features, 784, 60_000)
all_labels = dataset_train[:].targets

# Índices aleatorios fijos para reproducibilidad
rng_indices = MersenneTwister(SEMILLA)
indices_shuffled = randperm(rng_indices, 60_000)

println("  Train: 60,000 imágenes")
println("  Test:  10,000 imágenes (fijo para todas las pruebas)")
println()

# --- Función: entrenar red clásica ---
function entrenar_clasica(topologia, entradas, objetivos, epocas, lr, semilla)
    rng = MersenneTwister(semilla)
    red = RedNeuronal(topologia, rng)
    bufs = EntrenarBuffers(topologia)
    n_muestras = size(entradas, 2)

    t_inicio = time_ns()
    for _ in 1:epocas
        @inbounds for k in 1:n_muestras
            entrenar!(red, @view(entradas[:, k]), @view(objetivos[:, k]), lr, bufs)
        end
    end
    t_fin = time_ns()
    tiempo_ms = (t_fin - t_inicio) / 1_000_000.0

    return red, tiempo_ms
end

# --- Función: contar parámetros ---
function contar_parametros(topologia::Vector{Int})
    total = 0
    for i in 1:(length(topologia) - 1)
        total += topologia[i + 1] * topologia[i] + topologia[i + 1]
    end
    return total
end


# --- Ejecutar comparativa ---
println("=" ^ 80)
println("  COMPARATIVA: Nube Aleatoria vs Entrenamiento Clásico — MNIST")
println("=" ^ 80)
println()
println("  Topología:  $TOPOLOGIA ($(contar_parametros(TOPOLOGIA)) parámetros)")
println("  Épocas:     $EPOCAS")
println("  LR:         $LR")
println("  Nube:       $TAMANO_NUBE redes, umbral=$UMBRAL, eliminar=$NEURONAS_ELIMINAR")
println()

# Tabla de resultados
resultados = []

for n in TAMANOS
    println("-" ^ 80)
    println("  N = $n muestras")
    println("-" ^ 80)

    # Preparar subconjunto
    idx = indices_shuffled[1:n]
    entradas = all_features[:, idx]
    labels = all_labels[idx]
    objetivos = zeros(Float64, 10, n)
    for k in 1:n
        objetivos[labels[k] + 1, k] = 1.0
    end

    # --- Método Clásico ---
    print("  Clásico:     entrenando... ")
    red_clasica, tiempo_clasica = entrenar_clasica(TOPOLOGIA, entradas, objetivos, EPOCAS, LR, SEMILLA)
    prec_train_clasica = evaluar(red_clasica, entradas, objetivos)
    prec_test_clasica = evaluar(red_clasica, test_features, test_objetivos)
    println("$(round(tiempo_clasica/1000, digits=1))s")
    println("    Train: $(round(prec_train_clasica * 100, digits=2))%")
    println("    Test:  $(round(prec_test_clasica * 100, digits=2))%")
    println("    Topo:  $TOPOLOGIA ($(contar_parametros(TOPOLOGIA)) params)")

    # --- Método Nube Aleatoria ---
    print("  Nube:        ejecutando... ")
    config = ConfiguracionNube(
        tamano_nube = TAMANO_NUBE,
        topologia_inicial = TOPOLOGIA,
        umbral_acierto = UMBRAL,
        neuronas_eliminar = NEURONAS_ELIMINAR,
        epocas_refinamiento = EPOCAS,
        tasa_aprendizaje = LR,
        semilla = SEMILLA
    )
    motor = MotorNube(config, entradas, objetivos)
    informe = ejecutar(motor)
    println("$(round(informe.tiempo_ejecucion_ms/1000, digits=1))s")

    if informe.exitoso
        prec_test_nube = evaluar(informe.mejor_red, test_features, test_objetivos)
        topo_final = informe.topologia_final
        params_final = contar_parametros(topo_final)
        reduccion = round((1.0 - params_final / contar_parametros(TOPOLOGIA)) * 100, digits=1)

        println("    Train: $(round(informe.precision * 100, digits=2))%")
        println("    Test:  $(round(prec_test_nube * 100, digits=2))%")
        println("    Topo:  $topo_final ($params_final params, -$(reduccion)%)")
        println("    Evals: $(informe.total_redes_evaluadas), Reducciones: $(informe.total_reducciones)")

        push!(resultados, (n=n,
            clasica_train=prec_train_clasica, clasica_test=prec_test_clasica,
            clasica_tiempo=tiempo_clasica, clasica_params=contar_parametros(TOPOLOGIA),
            nube_train=informe.precision, nube_test=prec_test_nube,
            nube_tiempo=informe.tiempo_ejecucion_ms, nube_topo=topo_final,
            nube_params=params_final, reduccion=reduccion, exitoso=true))
    else
        println("    FALLIDO — no se encontró red viable")
        push!(resultados, (n=n,
            clasica_train=prec_train_clasica, clasica_test=prec_test_clasica,
            clasica_tiempo=tiempo_clasica, clasica_params=contar_parametros(TOPOLOGIA),
            nube_train=0.0, nube_test=0.0,
            nube_tiempo=informe.tiempo_ejecucion_ms, nube_topo=Int[],
            nube_params=0, reduccion=0.0, exitoso=false))
    end
    println()
end


# --- Tabla resumen ---
println("=" ^ 80)
println("  RESUMEN")
println("=" ^ 80)
println()
println("  Topología inicial: $TOPOLOGIA | Épocas: $EPOCAS | LR: $LR")
println()

# Header
hdr = "  N      │ Clásico Train │ Clásico Test │ Nube Train │ Nube Test │ Topo Final        │ Reducción │ Tiempo C │ Tiempo N"
println(hdr)
println("  " * "─" ^ (length(hdr) - 2))

for r in resultados
    ct = "$(round(r.clasica_train * 100, digits=1))%"
    cte = "$(round(r.clasica_test * 100, digits=1))%"
    tc = "$(round(r.clasica_tiempo / 1000, digits=1))s"

    if r.exitoso
        nt = "$(round(r.nube_train * 100, digits=1))%"
        nte = "$(round(r.nube_test * 100, digits=1))%"
        topo = "$(r.nube_topo)"
        red = "-$(r.reduccion)%"
        tn = "$(round(r.nube_tiempo / 1000, digits=1))s"
    else
        nt = "FAIL"
        nte = "FAIL"
        topo = "—"
        red = "—"
        tn = "$(round(r.nube_tiempo / 1000, digits=1))s"
    end

    println("  $(lpad(r.n, 6)) │ $(rpad(ct, 13)) │ $(rpad(cte, 12)) │ $(rpad(nt, 10)) │ $(rpad(nte, 9)) │ $(rpad(topo, 17)) │ $(rpad(red, 9)) │ $(rpad(tc, 8)) │ $tn")
end

println()
println("=" ^ 80)
