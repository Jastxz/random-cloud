# =============================================================================
# Comparativa: Nube Aleatoria vs Clásico en Optical Digits
# =============================================================================
#
# Ejecutar con:
#   julia --project=. -t auto examples/comparativa_digits.jl
#
# Optical Digits (UCI): imágenes 8×8 = 64 features, 10 clases (dígitos 0-9).
# 3823 train + 1797 test. Puente entre datasets tabulares y MNIST.
# =============================================================================

using RandomCloud
using RandomCloud: RedNeuronal, entrenar!, EntrenarBuffers, evaluar
using Random
using Downloads: download

const SEMILLA = 42
const EPOCAS = 100
const LR = 0.1

println("Descargando Optical Digits...")
url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra"
url_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes"
cache_train = joinpath(@__DIR__, "..", ".cache_digits_train.csv")
cache_test = joinpath(@__DIR__, "..", ".cache_digits_test.csv")

for (u, c, lbl) in [(url_train, cache_train, "Train"), (url_test, cache_test, "Test")]
    if !isfile(c)
        download(u, c)
        println("  $lbl descargado")
    else
        println("  $lbl: cache local")
    end
end

function parsear_digits(filepath)
    lines = filter(l -> !isempty(strip(l)), readlines(filepath))
    n = length(lines)
    features = zeros(Float64, 64, n)
    labels = zeros(Int, n)
    for (k, line) in enumerate(lines)
        parts = split(line, ',')
        for j in 1:64
            features[j, k] = parse(Float64, parts[j])
        end
        labels[k] = parse(Int, parts[65])
    end
    return features, labels
end


train_features, train_labels = parsear_digits(cache_train)
test_features, test_labels = parsear_digits(cache_test)

# Normalizar a [0,1] (valores originales 0-16)
train_features ./= 16.0
test_features ./= 16.0

# One-hot
function one_hot(labels, n_clases, n)
    Y = zeros(Float64, n_clases, n)
    for k in 1:n; Y[labels[k] + 1, k] = 1.0; end
    return Y
end

train_Y = one_hot(train_labels, 10, size(train_features, 2))
test_Y = one_hot(test_labels, 10, size(test_features, 2))

println("  Train: $(size(train_features, 2)) | Test: $(size(test_features, 2))")
println("  Features: 64 (8×8 píxeles) | Clases: 10")
println()

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


topologias = [
    [64, 16, 10],
    [64, 32, 10],
    [64, 32, 16, 10],
    [64, 64, 32, 10],
]

println("=" ^ 75)
println("  COMPARATIVA: Nube Aleatoria vs Clásico — Optical Digits")
println("=" ^ 75)
println()
println("  Épocas: $EPOCAS | LR: $LR | Threads: $(Threads.nthreads())")
println("  Nube: 50 redes, umbral=0.15, eliminar=2")
println()

resultados = []

for topo in topologias
    params = contar_parametros(topo)
    println("-" ^ 75)
    println("  Topología: $topo ($params params)")
    println("-" ^ 75)

    print("  Clásico:  entrenando... ")
    red_c, t_c = entrenar_clasica(topo, train_features, train_Y, EPOCAS, LR, SEMILLA)
    ct_train = evaluar(red_c, train_features, train_Y)
    ct_test = evaluar(red_c, test_features, test_Y)
    println("$(round(t_c/1000, digits=1))s — train=$(round(ct_train*100,digits=1))% test=$(round(ct_test*100,digits=1))%")

    semilla_nube = SEMILLA
    tiempo_total = 0.0
    for intento in 1:3
        label = intento > 1 ? "  Nube(#$intento): " : "  Nube:     "
        print(label * "ejecutando... ")
        config = ConfiguracionNube(
            tamano_nube=50, topologia_inicial=topo, umbral_acierto=0.15,
            neuronas_eliminar=2, epocas_refinamiento=EPOCAS,
            tasa_aprendizaje=LR, semilla=semilla_nube
        )
        motor = MotorNube(config, train_features, train_Y)
        informe = ejecutar(motor)
        tiempo_total += informe.tiempo_ejecucion_ms

        if informe.exitoso
            nt_test = evaluar(informe.mejor_red, test_features, test_Y)
            topo_f = informe.topologia_final
            pf = contar_parametros(topo_f)
            red_pct = round((1.0 - pf / params) * 100, digits=1)
            println("$(round(informe.tiempo_ejecucion_ms/1000, digits=1))s (s=$semilla_nube)")
            println("    Train: $(round(informe.precision*100,digits=1))%  Test: $(round(nt_test*100,digits=1))%  → $topo_f (-$(red_pct)%)")
            push!(resultados, (topo=topo, params=params, c_test=ct_test, n_test=nt_test,
                n_topo=topo_f, n_params=pf, reduccion=red_pct, exitoso=true, intentos=intento))
            break
        else
            println("$(round(informe.tiempo_ejecucion_ms/1000, digits=1))s (s=$semilla_nube) — no viable")
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
println("  RESUMEN — Optical Digits")
println("=" ^ 75)
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
println("=" ^ 75)
