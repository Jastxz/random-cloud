# =============================================================================
# Comparativa: Nube Aleatoria vs Clásico en Adult Income
# =============================================================================
#
# Ejecutar con:
#   julia --project=. -t auto examples/comparativa_adult.jl
#
# Adult Income (Census): 48,842 muestras, 14 features, 2 clases (>50K / ≤50K).
# Problema real de predicción de ingresos. Datos del UCI ML Repository.
# =============================================================================

using RandomCloud
using RandomCloud: RedNeuronal, entrenar!, EntrenarBuffers, evaluar
using Random
using Downloads: download

const SEMILLA = 42
const EPOCAS = 30
const LR = 0.1


# --- Descargar y parsear Adult Income ---
println("Descargando Adult Income...")
url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
url_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
cache_train = joinpath(@__DIR__, "..", ".cache_adult_train.csv")
cache_test = joinpath(@__DIR__, "..", ".cache_adult_test.csv")

if !isfile(cache_train)
    download(url_train, cache_train)
    println("  Train descargado")
else
    println("  Train: cache local")
end
if !isfile(cache_test)
    download(url_test, cache_test)
    println("  Test descargado")
else
    println("  Test: cache local")
end

# Columnas categóricas y sus valores posibles (para one-hot encoding)
# 0:age, 1:workclass, 2:fnlwgt, 3:education, 4:education-num, 5:marital-status,
# 6:occupation, 7:relationship, 8:race, 9:sex, 10:capital-gain, 11:capital-loss,
# 12:hours-per-week, 13:native-country, 14:income
const CAT_COLS = [2, 4, 6, 7, 8, 9, 10, 14]  # 1-indexed categorical columns
const NUM_COLS = [1, 3, 5, 11, 12, 13]         # 1-indexed numerical columns

function parsear_adult(filepath::String, skip_first::Bool=false)
    lines = readlines(filepath)
    if skip_first
        lines = lines[2:end]  # adult.test tiene una línea de header
    end
    # Filtrar líneas vacías y con '?'
    lines = filter(l -> !isempty(strip(l)) && !occursin("?", l), lines)

    n = length(lines)
    # Recoger todos los valores categóricos primero
    cat_values = Dict{Int, Set{String}}()
    for c in CAT_COLS
        cat_values[c] = Set{String}()
    end

    parsed = Vector{Vector{String}}(undef, n)
    labels = zeros(Int, n)

    for (i, line) in enumerate(lines)
        parts = [strip(p) for p in split(line, ',')]
        if length(parts) < 15
            continue
        end
        parsed[i] = parts[1:14]
        # Label: ">50K" o ">50K." (test file tiene punto)
        labels[i] = occursin(">50K", parts[15]) ? 1 : 0
        for c in CAT_COLS
            push!(cat_values[c], parts[c])
        end
    end

    return parsed, labels, cat_values
end


# Parsear ambos conjuntos para obtener vocabulario completo
parsed_train, labels_train, cat_vals_train = parsear_adult(cache_train, false)
parsed_test, labels_test, cat_vals_test = parsear_adult(cache_test, true)

# Unir vocabularios categóricos
cat_vals = Dict{Int, Vector{String}}()
for c in CAT_COLS
    cat_vals[c] = sort(collect(union(cat_vals_train[c], cat_vals_test[c])))
end

function codificar(parsed, labels, cat_vals)
    n = length(parsed)
    # Calcular dimensión total: numéricas + one-hot de categóricas
    n_num = length(NUM_COLS)
    n_cat = sum(length(cat_vals[c]) for c in CAT_COLS)
    n_features = n_num + n_cat

    features = zeros(Float64, n_features, n)
    objetivos = zeros(Float64, 2, n)

    for i in 1:n
        parts = parsed[i]
        col = 1
        # Numéricas
        for c in NUM_COLS
            features[col, i] = parse(Float64, parts[c])
            col += 1
        end
        # Categóricas (one-hot)
        for c in CAT_COLS
            vals = cat_vals[c]
            idx = findfirst(==(parts[c]), vals)
            if idx !== nothing
                features[col + idx - 1, i] = 1.0
            end
            col += length(vals)
        end
        # Label
        objetivos[labels[i] + 1, i] = 1.0
    end

    # Normalizar columnas numéricas a [0,1]
    for j in 1:n_num
        mn, mx = extrema(@view features[j, :])
        if mx > mn
            features[j, :] .= (features[j, :] .- mn) ./ (mx - mn)
        end
    end

    return features, objetivos
end

train_X, train_Y = codificar(parsed_train, labels_train, cat_vals)
test_X, test_Y = codificar(parsed_test, labels_test, cat_vals)

n_features = size(train_X, 1)
n_train = size(train_X, 2)
n_test = size(test_X, 2)
pct_pos = round(sum(labels_train) / n_train * 100, digits=1)

println("  Features: $n_features (6 numéricas + one-hot categóricas)")
println("  Train: $n_train | Test: $n_test")
println("  Distribución: $(pct_pos)% >50K, $(round(100-pct_pos, digits=1))% ≤50K")
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
    [n_features, 16, 2],
    [n_features, 32, 2],
    [n_features, 32, 16, 2],
    [n_features, 64, 32, 2],
]

println("=" ^ 75)
println("  COMPARATIVA: Nube Aleatoria vs Clásico — Adult Income")
println("=" ^ 75)
println()
println("  Épocas: $EPOCAS | LR: $LR | Threads: $(Threads.nthreads())")
println("  Nube: 50 redes, umbral=0.6, eliminar=2")
println()

resultados = []

for topo in topologias
    params = contar_parametros(topo)
    println("-" ^ 75)
    println("  Topología: [$(topo[1]), $(join(topo[2:end], ", "))] ($params params)")
    println("-" ^ 75)

    # Clásico
    print("  Clásico:  entrenando... ")
    red_c, t_c = entrenar_clasica(topo, train_X, train_Y, EPOCAS, LR, SEMILLA)
    ct_train = evaluar(red_c, train_X, train_Y)
    ct_test = evaluar(red_c, test_X, test_Y)
    println("$(round(t_c/1000, digits=1))s — train=$(round(ct_train*100,digits=1))% test=$(round(ct_test*100,digits=1))%")

    # Nube con reintentos
    semilla_nube = SEMILLA
    tiempo_total = 0.0
    for intento in 1:3
        label = intento > 1 ? "  Nube(#$intento): " : "  Nube:     "
        print(label * "ejecutando... ")
        config = ConfiguracionNube(
            tamano_nube=50, topologia_inicial=topo, umbral_acierto=0.6,
            neuronas_eliminar=2, epocas_refinamiento=EPOCAS,
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
            println("$(round(informe.tiempo_ejecucion_ms/1000, digits=1))s (s=$semilla_nube)")
            println("    Train: $(round(informe.precision*100,digits=1))%  Test: $(round(nt_test*100,digits=1))%  → [$(topo_f[1]),$(join(topo_f[2:end],","))] (-$(red_pct)%)")
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
println("  RESUMEN — Adult Income")
println("=" ^ 75)
println()
for r in resultados
    ct = "$(round(r.c_test * 100, digits=1))%"
    topo_short = "[$(r.topo[1]),$(join(r.topo[2:end],","))]"
    if r.exitoso
        nt = "$(round(r.n_test * 100, digits=1))%"
        s = r.intentos > 1 ? " (#$(r.intentos))" : ""
        nf = "[$(r.n_topo[1]),$(join(r.n_topo[2:end],","))]"
        println("  $(rpad(topo_short, 20)) │ Clásico: $(rpad(ct, 7)) │ Nube: $(rpad(nt, 7)) → $nf (-$(r.reduccion)%)$s")
    else
        println("  $(rpad(topo_short, 20)) │ Clásico: $(rpad(ct, 7)) │ Nube: FAIL (3 intentos)")
    end
end
println()
println("=" ^ 75)
