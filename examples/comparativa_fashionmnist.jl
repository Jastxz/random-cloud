# =============================================================================
# Comparativa: Nube Aleatoria vs Clásico en Fashion-MNIST
# =============================================================================
#
# Ejecutar con:
#   julia --project=. -t auto examples/comparativa_fashionmnist.jl
#
# Fashion-MNIST: 28×28 = 784 entradas, 10 clases (prendas de ropa).
# Drop-in replacement de MNIST pero más difícil.
# =============================================================================

using RandomCloud
using RandomCloud: RedNeuronal, entrenar!, EntrenarBuffers, evaluar
using MLDatasets: FashionMNIST
using Random

const SEMILLA = 42
const EPOCAS = 30
const LR = 0.1

println("Cargando Fashion-MNIST...")
dataset_train = FashionMNIST(Float64, :train)
dataset_test = FashionMNIST(Float64, :test)

# Test set fijo (10K)
test_X = reshape(dataset_test[:].features, 784, 10_000)
test_labels = dataset_test[:].targets
test_Y = zeros(Float64, 10, 10_000)
for k in 1:10_000; test_Y[test_labels[k] + 1, k] = 1.0; end


# Train: subconjuntos crecientes
all_features = reshape(dataset_train[:].features, 784, 60_000)
all_labels = dataset_train[:].targets
rng_idx = MersenneTwister(SEMILLA)
indices_shuffled = randperm(rng_idx, 60_000)

println("  Train: 60,000 disponibles | Test: 10,000 fijo")
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
const TOPO = [784, 64, 32, 10]
const TAMANOS = [1_000, 5_000, 10_000, 60_000]
params = contar_parametros(TOPO)

println("=" ^ 75)
println("  COMPARATIVA: Nube Aleatoria vs Clásico — Fashion-MNIST")
println("=" ^ 75)
println()
println("  Topología: $TOPO ($params params) | Épocas: $EPOCAS | LR: $LR")
println("  Nube: 50 redes, umbral=0.12, eliminar=2 | Threads: $(Threads.nthreads())")
println()

resultados = []


for n in TAMANOS
    println("-" ^ 75)
    println("  N = $n muestras")
    println("-" ^ 75)

    idx = indices_shuffled[1:n]
    X = all_features[:, idx]
    labels = all_labels[idx]
    Y = zeros(Float64, 10, n)
    for k in 1:n; Y[labels[k] + 1, k] = 1.0; end

    # Clásico
    print("  Clásico:  entrenando... ")
    red_c, t_c = entrenar_clasica(TOPO, X, Y, EPOCAS, LR, SEMILLA)
    ct_train = evaluar(red_c, X, Y)
    ct_test = evaluar(red_c, test_X, test_Y)
    println("$(round(t_c/1000, digits=1))s — train=$(round(ct_train*100,digits=1))% test=$(round(ct_test*100,digits=1))%")

    # Nube con reintentos
    semilla_nube = SEMILLA
    tiempo_total = 0.0
    for intento in 1:3
        label = intento > 1 ? "  Nube(#$intento): " : "  Nube:     "
        print(label * "ejecutando... ")
        config = ConfiguracionNube(
            tamano_nube=50, topologia_inicial=TOPO, umbral_acierto=0.12,
            neuronas_eliminar=2, epocas_refinamiento=EPOCAS,
            tasa_aprendizaje=LR, semilla=semilla_nube
        )
        motor = MotorNube(config, X, Y)
        informe = ejecutar(motor)
        tiempo_total += informe.tiempo_ejecucion_ms

        if informe.exitoso
            nt_test = evaluar(informe.mejor_red, test_X, test_Y)
            topo_f = informe.topologia_final
            pf = contar_parametros(topo_f)
            red_pct = round((1.0 - pf / params) * 100, digits=1)
            println("$(round(informe.tiempo_ejecucion_ms/1000, digits=1))s (s=$semilla_nube)")
            println("    Train: $(round(informe.precision*100,digits=1))%  Test: $(round(nt_test*100,digits=1))%  → $topo_f (-$(red_pct)%)")
            push!(resultados, (n=n, c_test=ct_test, n_test=nt_test, n_topo=topo_f,
                n_params=pf, reduccion=red_pct, exitoso=true, intentos=intento,
                c_tiempo=t_c, n_tiempo=tiempo_total))
            break
        else
            println("$(round(informe.tiempo_ejecucion_ms/1000, digits=1))s (s=$semilla_nube) — no viable")
            semilla_nube += 1000
            if intento == 3
                push!(resultados, (n=n, c_test=ct_test, n_test=0.0, n_topo=Int[],
                    n_params=0, reduccion=0.0, exitoso=false, intentos=3,
                    c_tiempo=t_c, n_tiempo=tiempo_total))
            end
        end
    end
    println()
end

# --- Resumen ---
println("=" ^ 75)
println("  RESUMEN — Fashion-MNIST, topología $TOPO")
println("=" ^ 75)
println()
for r in resultados
    ct = "$(round(r.c_test * 100, digits=1))%"
    if r.exitoso
        nt = "$(round(r.n_test * 100, digits=1))%"
        s = r.intentos > 1 ? " (#$(r.intentos))" : ""
        println("  N=$(lpad(r.n, 6)) │ Clásico: $(rpad(ct, 7)) │ Nube: $(rpad(nt, 7)) → $(r.n_topo) (-$(r.reduccion)%)$s")
    else
        println("  N=$(lpad(r.n, 6)) │ Clásico: $(rpad(ct, 7)) │ Nube: FAIL (3 intentos)")
    end
end
println()
println("=" ^ 75)
