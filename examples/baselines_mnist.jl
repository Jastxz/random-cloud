# =============================================================================
# Baselines a escala: Nube vs Magnitude Pruning vs Random Pruning en MNIST
# =============================================================================
#
# Ejecutar con:
#   julia --project=. -t auto examples/baselines_mnist.jl
#
# MNIST: 784 entradas, 10 clases. Topología [784, 32, 16, 10].
# Prueba la escalabilidad del método con dimensionalidad alta.
# =============================================================================

using RandomCloud
using RandomCloud: RedNeuronal, entrenar!, EntrenarBuffers, reconstruir
using RandomCloud: evaluar, evaluar_f1, evaluar_auc
using MLDatasets: MNIST
using Random

function contar_parametros(topo::Vector{Int})
    sum(topo[i+1] * topo[i] + topo[i+1] for i in 1:length(topo)-1)
end

function entrenar_red!(red, X, Y, epocas, lr)
    bufs = EntrenarBuffers(red.topologia)
    n = size(X, 2)
    for _ in 1:epocas
        @inbounds for k in 1:n
            entrenar!(red, @view(X[:, k]), @view(Y[:, k]), lr, bufs)
        end
    end
    return red
end

function evaluar_todo(red, X, Y)
    acc = evaluar(red, X, Y)
    f1 = evaluar_f1(red, X, Y)
    auc = evaluar_auc(red, X, Y)
    return acc, f1, auc
end

# Magnitude pruning: eliminar neuronas con menor norma L2 de pesos entrantes
function magnitude_prune(red::RedNeuronal, topo_objetivo::Vector{Int})
    n_capas = length(red.topologia)
    nuevos_pesos = [copy(w) for w in red.pesos]
    nuevos_biases = [copy(b) for b in red.biases]
    topo_actual = copy(red.topologia)

    for capa in 2:(n_capas - 1)
        n_actual = topo_actual[capa]
        n_objetivo = topo_objetivo[capa]
        n_objetivo >= n_actual && continue

        idx_capa = capa - 1
        normas = [sum(nuevos_pesos[idx_capa][j, :].^2) for j in 1:n_actual]
        orden = sortperm(normas, rev=true)
        mantener = sort(orden[1:n_objetivo])

        nuevos_pesos[idx_capa] = nuevos_pesos[idx_capa][mantener, :]
        nuevos_biases[idx_capa] = nuevos_biases[idx_capa][mantener]
        if idx_capa < length(nuevos_pesos)
            nuevos_pesos[idx_capa + 1] = nuevos_pesos[idx_capa + 1][:, mantener]
        end
        topo_actual[capa] = n_objetivo
    end
    return RedNeuronal(topo_actual, nuevos_pesos, nuevos_biases)
end

# Random pruning: eliminar neuronas al azar
function random_prune(red::RedNeuronal, topo_objetivo::Vector{Int}, rng::AbstractRNG)
    n_capas = length(red.topologia)
    nuevos_pesos = [copy(w) for w in red.pesos]
    nuevos_biases = [copy(b) for b in red.biases]
    topo_actual = copy(red.topologia)

    for capa in 2:(n_capas - 1)
        n_actual = topo_actual[capa]
        n_objetivo = topo_objetivo[capa]
        n_objetivo >= n_actual && continue

        idx_capa = capa - 1
        mantener = sort(shuffle(rng, collect(1:n_actual))[1:n_objetivo])

        nuevos_pesos[idx_capa] = nuevos_pesos[idx_capa][mantener, :]
        nuevos_biases[idx_capa] = nuevos_biases[idx_capa][mantener]
        if idx_capa < length(nuevos_pesos)
            nuevos_pesos[idx_capa + 1] = nuevos_pesos[idx_capa + 1][:, mantener]
        end
        topo_actual[capa] = n_objetivo
    end
    return RedNeuronal(topo_actual, nuevos_pesos, nuevos_biases)
end

# ─── Cargar MNIST ───
println("Cargando MNIST...")
ds_train = MNIST(Float64, :train)
ds_test = MNIST(Float64, :test)

test_X = reshape(ds_test[:].features, 784, 10_000)
test_labels = ds_test[:].targets
test_Y = zeros(Float64, 10, 10_000)
for k in 1:10_000; test_Y[test_labels[k] + 1, k] = 1.0; end

all_X = reshape(ds_train[:].features, 784, 60_000)
all_labels = ds_train[:].targets

rng_idx = MersenneTwister(42)
idx_shuf = randperm(rng_idx, 60_000)

println("  Test fijo: 10,000 imágenes")
println()

const SEMILLA = 42
const TOPO = [784, 32, 16, 10]
const PARAMS_I = contar_parametros(TOPO)
const LR = 0.1
const EPOCAS = 30

println("=" ^ 105)
println("  BASELINES A ESCALA: Nube vs Magnitude Pruning vs Random Pruning — MNIST")
println("  Topología: $TOPO ($PARAMS_I params) | Épocas: $EPOCAS | LR: $LR")
println("=" ^ 105)
println()

for N in [1_000, 5_000, 10_000]
    println("─" ^ 105)
    println("  N = $N muestras")
    println("─" ^ 105)

    idx = idx_shuf[1:N]
    trX = all_X[:, idx]
    trY = zeros(Float64, 10, N)
    for k in 1:N; trY[all_labels[idx[k]] + 1, k] = 1.0; end

    # 1. NUBE — primero para obtener la topología objetivo
    print("  Nube:        ejecutando... ")
    t0 = time_ns()
    config = ConfiguracionNube(
        tamano_nube=100, topologia_inicial=TOPO,
        umbral_acierto=0.15, neuronas_eliminar=2,
        epocas_refinamiento=EPOCAS, tasa_aprendizaje=LR, semilla=SEMILLA)
    motor = MotorNube(config, trX, trY)
    informe = ejecutar(motor)
    t_nube = (time_ns() - t0) / 1e9

    if !informe.exitoso
        println("FAIL ($(round(t_nube, digits=1))s)")
        println()
        continue
    end

    topo_nube = informe.topologia_final
    params_nube = contar_parametros(topo_nube)
    red_pct = round((1.0 - params_nube / PARAMS_I) * 100, digits=1)
    acc_n, f1_n, auc_n = evaluar_todo(informe.mejor_red, test_X, test_Y)
    println("$(round(t_nube, digits=1))s → $topo_nube (-$(red_pct)%)")

    # 2. CLÁSICO — red completa
    print("  Clásico:     entrenando... ")
    t0 = time_ns()
    red_c = RedNeuronal(TOPO, MersenneTwister(SEMILLA))
    entrenar_red!(red_c, trX, trY, EPOCAS, LR)
    t_clasico = (time_ns() - t0) / 1e9
    acc_c, f1_c, auc_c = evaluar_todo(red_c, test_X, test_Y)
    println("$(round(t_clasico, digits=1))s")

    # 3. MAGNITUDE PRUNING — entrenar → podar → fine-tune
    print("  Magnitude:   train+prune+ft... ")
    t0 = time_ns()
    red_m = RedNeuronal(TOPO, MersenneTwister(SEMILLA))
    entrenar_red!(red_m, trX, trY, EPOCAS, LR)       # entrenar completa
    red_m = magnitude_prune(red_m, topo_nube)          # podar
    entrenar_red!(red_m, trX, trY, EPOCAS, LR)        # fine-tune
    t_mag = (time_ns() - t0) / 1e9
    acc_m, f1_m, auc_m = evaluar_todo(red_m, test_X, test_Y)
    println("$(round(t_mag, digits=1))s")

    # 4. RANDOM PRUNING — entrenar → podar al azar → fine-tune (×5)
    print("  Random(×5):  train+prune+ft... ")
    t0 = time_ns()
    acc_rs = Float64[]; f1_rs = Float64[]; auc_rs = Float64[]
    for s in 1:5
        red_r = RedNeuronal(TOPO, MersenneTwister(SEMILLA))
        entrenar_red!(red_r, trX, trY, EPOCAS, LR)
        red_r = random_prune(red_r, topo_nube, MersenneTwister(SEMILLA + s * 1000))
        entrenar_red!(red_r, trX, trY, EPOCAS, LR)
        a, f, u = evaluar_todo(red_r, test_X, test_Y)
        push!(acc_rs, a); push!(f1_rs, f); push!(auc_rs, u)
    end
    t_rnd = (time_ns() - t0) / 1e9
    acc_r = sum(acc_rs)/5; f1_r = sum(f1_rs)/5; auc_r = sum(auc_rs)/5
    println("$(round(t_rnd, digits=1))s")

    # Tabla
    println()
    fmt(v) = rpad(string(round(v, digits=3)), 5)
    fmtp(v) = rpad(string(round(v*100, digits=1)) * "%", 7)

    println("  Método             │ Acc     │ F1    │ AUC   │ Topología            │ Params │ Red.    │ Tiempo")
    println("  ───────────────────┼─────────┼───────┼───────┼──────────────────────┼────────┼─────────┼───────")
    println("  Clásico (full)     │ $(fmtp(acc_c)) │ $(fmt(f1_c)) │ $(fmt(auc_c)) │ $(rpad(string(TOPO), 20)) │ $(rpad(PARAMS_I, 6)) │ —       │ $(round(t_clasico, digits=1))s")
    println("  Magnitude Pruning  │ $(fmtp(acc_m)) │ $(fmt(f1_m)) │ $(fmt(auc_m)) │ $(rpad(string(topo_nube), 20)) │ $(rpad(params_nube, 6)) │ -$(red_pct)%  │ $(round(t_mag, digits=1))s")
    println("  Random Pruning (×5)│ $(fmtp(acc_r)) │ $(fmt(f1_r)) │ $(fmt(auc_r)) │ $(rpad(string(topo_nube), 20)) │ $(rpad(params_nube, 6)) │ -$(red_pct)%  │ $(round(t_rnd/5, digits=1))s/run")
    println("  Nube Aleatoria     │ $(fmtp(acc_n)) │ $(fmt(f1_n)) │ $(fmt(auc_n)) │ $(rpad(string(topo_nube), 20)) │ $(rpad(params_nube, 6)) │ -$(red_pct)%  │ $(round(t_nube, digits=1))s")
    println()

    println("  Δ vs Nube:  Mag: $(round((acc_m - acc_n)*100, digits=1))pp acc, $(round(f1_m - f1_n, digits=3)) F1, $(round(auc_m - auc_n, digits=3)) AUC")
    println("              Rnd: $(round((acc_r - acc_n)*100, digits=1))pp acc, $(round(f1_r - f1_n, digits=3)) F1, $(round(auc_r - auc_n, digits=3)) AUC")
    println("              Cls: $(round((acc_c - acc_n)*100, digits=1))pp acc, $(round(f1_c - f1_n, digits=3)) F1, $(round(auc_c - auc_n, digits=3)) AUC")
    println()
end

println("=" ^ 105)
println("  FIN")
println("=" ^ 105)
