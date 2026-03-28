# =============================================================================
# Comparativa: Método de la Nube Aleatoria vs Clásico en Two Moons
# =============================================================================
#
# Ejecutar con:
#   julia --project=. -t auto examples/comparativa_twomoons.jl
#
# Two Moons: 2 features, 2 clases, frontera de decisión no lineal.
# Dataset sintético generado proceduralmente.
# =============================================================================

using RandomCloud
using RandomCloud: RedNeuronal, feedforward, entrenar!, EntrenarBuffers, evaluar
using Random

const SEMILLA = 42
const EPOCAS = 200
const LR = 0.5


# --- Generador de Two Moons ---
function generar_two_moons(n::Int, ruido::Float64, rng::AbstractRNG)
    n_por_clase = n ÷ 2
    # Luna superior
    theta1 = range(0, π, length=n_por_clase)
    x1 = cos.(theta1) .+ ruido .* randn(rng, n_por_clase)
    y1 = sin.(theta1) .+ ruido .* randn(rng, n_por_clase)
    # Luna inferior (desplazada)
    theta2 = range(0, π, length=n_por_clase)
    x2 = 1.0 .- cos.(theta2) .+ ruido .* randn(rng, n_por_clase)
    y2 = -sin.(theta2) .+ 0.5 .+ ruido .* randn(rng, n_por_clase)

    features = hcat(vcat(x1, x2), vcat(y1, y2))'  # 2×n
    labels = vcat(zeros(Int, n_por_clase), ones(Int, n_por_clase))

    # Shuffle
    perm = randperm(rng, n)
    return Float64.(features[:, perm]), labels[perm]
end

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


# --- Generar datos ---
rng_data = MersenneTwister(SEMILLA)
train_features, train_labels = generar_two_moons(1000, 0.1, rng_data)
test_features, test_labels = generar_two_moons(500, 0.1, rng_data)

# One-hot: 2 clases → vectores [1,0] y [0,1]
function one_hot(labels, n_clases)
    n = length(labels)
    Y = zeros(Float64, n_clases, n)
    for k in 1:n
        Y[labels[k] + 1, k] = 1.0
    end
    return Y
end

train_Y = one_hot(train_labels, 2)
test_Y = one_hot(test_labels, 2)

println("=" ^ 70)
println("  COMPARATIVA: Nube Aleatoria vs Clásico — Two Moons")
println("=" ^ 70)
println()
println("  Train: $(size(train_features, 2)) muestras | Test: $(size(test_features, 2)) muestras")
println("  Features: 2 | Clases: 2 | Ruido: 0.1")
println("  Épocas: $EPOCAS | LR: $LR | Threads: $(Threads.nthreads())")
println("  Nube: 50 redes, umbral=0.6, eliminar=1")
println()

# --- Comparativa ---
topologias = [
    [2, 4, 2],
    [2, 8, 2],
    [2, 8, 4, 2],
    [2, 16, 8, 2],
    [2, 16, 8, 4, 2],
]

resultados = []

for topo in topologias
    params = contar_parametros(topo)
    println("-" ^ 70)
    println("  Topología: $topo ($params parámetros)")
    println("-" ^ 70)

    # Clásico
    print("  Clásico:  ")
    red_c, t_c = entrenar_clasica(topo, train_features, train_Y, EPOCAS, LR, SEMILLA)
    prec_train_c = evaluar(red_c, train_features, train_Y)
    prec_test_c = evaluar(red_c, test_features, test_Y)
    println("$(round(t_c, digits=1))ms — train=$(round(prec_train_c*100,digits=1))% test=$(round(prec_test_c*100,digits=1))%")


    # Nube — con reintentos
    semilla_nube = SEMILLA
    tiempo_total_nube = 0.0
    for intento in 1:3
        label = intento > 1 ? "  Nube(#$intento): " : "  Nube:     "
        print(label)
        config = ConfiguracionNube(
            tamano_nube = 50,
            topologia_inicial = topo,
            umbral_acierto = 0.6,
            neuronas_eliminar = 1,
            epocas_refinamiento = EPOCAS,
            tasa_aprendizaje = LR,
            semilla = semilla_nube
        )
        motor = MotorNube(config, train_features, train_Y)
        informe = ejecutar(motor)
        tiempo_total_nube += informe.tiempo_ejecucion_ms

        if informe.exitoso
            prec_test_n = evaluar(informe.mejor_red, test_features, test_Y)
            topo_f = informe.topologia_final
            params_f = contar_parametros(topo_f)
            red_pct = round((1.0 - params_f / params) * 100, digits=1)
            println("$(round(informe.tiempo_ejecucion_ms, digits=1))ms (semilla=$semilla_nube) — train=$(round(informe.precision*100,digits=1))% test=$(round(prec_test_n*100,digits=1))% → $topo_f (-$(red_pct)%)")

            push!(resultados, (topo=topo, params=params,
                c_train=prec_train_c, c_test=prec_test_c, c_tiempo=t_c,
                n_train=informe.precision, n_test=prec_test_n, n_tiempo=tiempo_total_nube,
                n_topo=topo_f, n_params=params_f, reduccion=red_pct, exitoso=true, intentos=intento))
            break
        else
            println("$(round(informe.tiempo_ejecucion_ms, digits=1))ms (semilla=$semilla_nube) — no viable")
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
