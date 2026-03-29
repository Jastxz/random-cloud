# =============================================================================
# Comparativa: Nube Aleatoria vs Magnitude Pruning vs Random Pruning
# =============================================================================
#
# Ejecutar con:
#   julia --project=. -t auto examples/comparativa_baselines.jl
#
# Compara 4 métodos con la misma topología objetivo:
#   1. Clásico: red completa entrenada (baseline superior)
#   2. Magnitude Pruning: entrenar → podar neuronas por norma L2 → fine-tune
#   3. Random Pruning: entrenar → podar neuronas al azar → fine-tune
#   4. Nube Aleatoria: explorar sin entrenar → podar → refinar
#
# Todos usan las mismas épocas totales para comparación justa.
# =============================================================================

using RandomCloud
using RandomCloud: RedNeuronal, entrenar!, EntrenarBuffers, reconstruir
using RandomCloud: evaluar, evaluar_f1, evaluar_auc
using Random
using Downloads: download

# ─── Utilidades ───

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

# ─── Magnitude Pruning (estructural, por neurona) ───
# Elimina neuronas de capas ocultas con menor norma L2 de pesos entrantes.
# Produce una red con la topología objetivo, preservando los pesos de las
# neuronas más importantes.

function magnitude_prune(red::RedNeuronal, topo_objetivo::Vector{Int})
    n_capas = length(red.topologia)
    nuevos_pesos = [copy(w) for w in red.pesos]
    nuevos_biases = [copy(b) for b in red.biases]
    topo_actual = copy(red.topologia)

    # Para cada capa oculta, seleccionar las neuronas con mayor norma L2
    for capa in 2:(n_capas - 1)
        n_actual = topo_actual[capa]
        n_objetivo = topo_objetivo[capa]
        if n_objetivo >= n_actual
            continue
        end

        # Norma L2 de pesos entrantes de cada neurona (fila de la matriz de pesos)
        idx_capa = capa - 1  # índice en el array de pesos
        normas = [sum(nuevos_pesos[idx_capa][j, :].^2) for j in 1:n_actual]

        # Seleccionar las n_objetivo neuronas con mayor norma (mantener las más importantes)
        orden = sortperm(normas, rev=true)
        mantener = sort(orden[1:n_objetivo])

        # Recortar pesos entrantes (filas) y biases
        nuevos_pesos[idx_capa] = nuevos_pesos[idx_capa][mantener, :]
        nuevos_biases[idx_capa] = nuevos_biases[idx_capa][mantener]

        # Recortar pesos salientes (columnas de la siguiente capa)
        if idx_capa < length(nuevos_pesos)
            nuevos_pesos[idx_capa + 1] = nuevos_pesos[idx_capa + 1][:, mantener]
        end

        topo_actual[capa] = n_objetivo
    end

    return RedNeuronal(topo_actual, nuevos_pesos, nuevos_biases)
end

# ─── Random Pruning (estructural, por neurona) ───
# Elimina neuronas al azar de capas ocultas.

function random_prune(red::RedNeuronal, topo_objetivo::Vector{Int}, rng::AbstractRNG)
    n_capas = length(red.topologia)
    nuevos_pesos = [copy(w) for w in red.pesos]
    nuevos_biases = [copy(b) for b in red.biases]
    topo_actual = copy(red.topologia)

    for capa in 2:(n_capas - 1)
        n_actual = topo_actual[capa]
        n_objetivo = topo_objetivo[capa]
        if n_objetivo >= n_actual
            continue
        end

        idx_capa = capa - 1
        # Seleccionar n_objetivo neuronas al azar
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

# ─── Split estratificado ───

function split_estratificado(labels, clases, semilla; ratio=0.8)
    rng = MersenneTwister(semilla)
    train_idx = Int[]; test_idx = Int[]
    for c in clases
        idx = findall(==(c), labels)
        perm = shuffle(rng, idx)
        n_train = round(Int, ratio * length(perm))
        append!(train_idx, perm[1:n_train])
        append!(test_idx, perm[n_train+1:end])
    end
    shuffle!(rng, train_idx); shuffle!(rng, test_idx)
    return train_idx, test_idx
end

# ─── Comparativa para un dataset ───

function comparar_metodos(nombre, trX, trY, teX, teY, topo_inicial,
                          epocas_total, epocas_refine, lr, umbral, neuronas_elim;
                          semilla=42)
    params_i = contar_parametros(topo_inicial)

    println("─" ^ 100)
    println("  $nombre — $topo_inicial ($params_i params)")
    println("  Épocas total: $epocas_total | Épocas refinamiento: $epocas_refine | LR: $lr")
    println("─" ^ 100)
    println()

    # 1. CLÁSICO: entrenar red completa
    red_clasica = RedNeuronal(topo_inicial, MersenneTwister(semilla))
    entrenar_red!(red_clasica, trX, trY, epocas_total, lr)
    acc_c, f1_c, auc_c = evaluar_todo(red_clasica, teX, teY)

    # 2. NUBE: explorar + refinar
    config = ConfiguracionNube(
        tamano_nube=50, topologia_inicial=topo_inicial,
        umbral_acierto=umbral, neuronas_eliminar=neuronas_elim,
        epocas_refinamiento=epocas_refine, tasa_aprendizaje=lr, semilla=semilla)
    motor = MotorNube(config, trX, trY)
    informe = ejecutar(motor)

    if !informe.exitoso
        println("  Nube: FAIL — no se encontró red viable. Saltando comparativa.")
        println()
        return
    end

    topo_nube = informe.topologia_final
    params_nube = contar_parametros(topo_nube)
    red_pct = round((1.0 - params_nube / params_i) * 100, digits=1)
    acc_n, f1_n, auc_n = evaluar_todo(informe.mejor_red, teX, teY)

    # 3. MAGNITUDE PRUNING: entrenar completa → podar → fine-tune
    red_mag = RedNeuronal(topo_inicial, MersenneTwister(semilla))
    entrenar_red!(red_mag, trX, trY, epocas_total - epocas_refine, lr)  # entrenar
    red_mag = magnitude_prune(red_mag, topo_nube)                        # podar
    entrenar_red!(red_mag, trX, trY, epocas_refine, lr)                  # fine-tune
    acc_m, f1_m, auc_m = evaluar_todo(red_mag, teX, teY)

    # 4. RANDOM PRUNING: entrenar completa → podar al azar → fine-tune (promedio de 5 runs)
    acc_rs = Float64[]; f1_rs = Float64[]; auc_rs = Float64[]
    for s in 1:5
        red_rp = RedNeuronal(topo_inicial, MersenneTwister(semilla))
        entrenar_red!(red_rp, trX, trY, epocas_total - epocas_refine, lr)
        red_rp = random_prune(red_rp, topo_nube, MersenneTwister(semilla + s * 1000))
        entrenar_red!(red_rp, trX, trY, epocas_refine, lr)
        a, f, u = evaluar_todo(red_rp, teX, teY)
        push!(acc_rs, a); push!(f1_rs, f); push!(auc_rs, u)
    end
    acc_r = sum(acc_rs) / 5; f1_r = sum(f1_rs) / 5; auc_r = sum(auc_rs) / 5

    # Imprimir resultados
    fmt(v) = rpad(string(round(v, digits=3)), 5)
    fmtp(v) = rpad(string(round(v*100, digits=1)) * "%", 7)

    println("  Método             │ Acc     │ F1    │ AUC   │ Topología            │ Params │ Reducción")
    println("  ───────────────────┼─────────┼───────┼───────┼──────────────────────┼────────┼──────────")
    println("  Clásico (full)     │ $(fmtp(acc_c)) │ $(fmt(f1_c)) │ $(fmt(auc_c)) │ $(rpad(string(topo_inicial), 20)) │ $(rpad(params_i, 6)) │ —")
    println("  Magnitude Pruning  │ $(fmtp(acc_m)) │ $(fmt(f1_m)) │ $(fmt(auc_m)) │ $(rpad(string(topo_nube), 20)) │ $(rpad(params_nube, 6)) │ -$(red_pct)%")
    println("  Random Pruning (×5)│ $(fmtp(acc_r)) │ $(fmt(f1_r)) │ $(fmt(auc_r)) │ $(rpad(string(topo_nube), 20)) │ $(rpad(params_nube, 6)) │ -$(red_pct)%")
    println("  Nube Aleatoria     │ $(fmtp(acc_n)) │ $(fmt(f1_n)) │ $(fmt(auc_n)) │ $(rpad(string(topo_nube), 20)) │ $(rpad(params_nube, 6)) │ -$(red_pct)%")
    println()

    # Deltas vs nube
    println("  Δ vs Nube:  Mag: $(round((acc_m - acc_n)*100, digits=1))pp acc, $(round(f1_m - f1_n, digits=3)) F1, $(round(auc_m - auc_n, digits=3)) AUC")
    println("              Rnd: $(round((acc_r - acc_n)*100, digits=1))pp acc, $(round(f1_r - f1_n, digits=3)) F1, $(round(auc_r - auc_n, digits=3)) AUC")
    println()
end

const SEMILLA = 42

println("=" ^ 100)
println("  COMPARATIVA: Nube Aleatoria vs Magnitude Pruning vs Random Pruning")
println("  Todos los métodos podados usan la misma topología objetivo (la que encontró la nube)")
println("=" ^ 100)
println()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. BREAST CANCER
# ═══════════════════════════════════════════════════════════════════════════════
bc_path = joinpath(@__DIR__, "..", ".cache_breastcancer.csv")
if isfile(bc_path)
    lines = readlines(bc_path)
    n = length(lines)
    X = zeros(Float64, 30, n); lab = zeros(Int, n)
    for (k, line) in enumerate(lines)
        parts = split(line, ',')
        lab[k] = parts[2] == "M" ? 1 : 0
        for j in 1:30; X[j, k] = parse(Float64, parts[j + 2]); end
    end
    for f in 1:30
        mn, mx = extrema(@view X[f, :])
        if mx > mn; X[f, :] .= (X[f, :] .- mn) ./ (mx - mn); end
    end
    Y = zeros(Float64, 2, n)
    for k in 1:n; Y[lab[k] + 1, k] = 1.0; end
    tr, te = split_estratificado(lab, [0, 1], SEMILLA)

    comparar_metodos("Breast Cancer", X[:, tr], Y[:, tr], X[:, te], Y[:, te],
        [30, 8, 2], 100, 100, 0.1, 0.7, 1)
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SONAR
# ═══════════════════════════════════════════════════════════════════════════════
sonar_path = joinpath(@__DIR__, "..", ".cache_sonar.csv")
if isfile(sonar_path)
    lines = filter(l -> !isempty(strip(l)), readlines(sonar_path))
    n = length(lines)
    X = zeros(Float64, 60, n); lab = zeros(Int, n)
    for (k, line) in enumerate(lines)
        parts = split(line, ',')
        for j in 1:60; X[j, k] = parse(Float64, parts[j]); end
        lab[k] = parts[61] == "M" ? 1 : 0
    end
    for f in 1:60
        mn, mx = extrema(@view X[f, :])
        if mx > mn; X[f, :] .= (X[f, :] .- mn) ./ (mx - mn); end
    end
    Y = zeros(Float64, 2, n)
    for k in 1:n; Y[lab[k] + 1, k] = 1.0; end
    tr, te = split_estratificado(lab, [0, 1], SEMILLA)

    comparar_metodos("Sonar", X[:, tr], Y[:, tr], X[:, te], Y[:, te],
        [60, 8, 2], 200, 200, 0.1, 0.55, 1)
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3. IONOSPHERE
# ═══════════════════════════════════════════════════════════════════════════════
iono_path = joinpath(@__DIR__, "..", ".cache_ionosphere.csv")
if isfile(iono_path)
    lines = filter(l -> !isempty(strip(l)), readlines(iono_path))
    n = length(lines)
    X = zeros(Float64, 34, n); lab = zeros(Int, n)
    for (k, line) in enumerate(lines)
        parts = split(line, ',')
        for j in 1:34; X[j, k] = parse(Float64, parts[j]); end
        lab[k] = parts[35] == "g" ? 1 : 0
    end
    for f in 1:34
        mn, mx = extrema(@view X[f, :])
        if mx > mn; X[f, :] .= (X[f, :] .- mn) ./ (mx - mn); end
    end
    Y = zeros(Float64, 2, n)
    for k in 1:n; Y[lab[k] + 1, k] = 1.0; end
    tr, te = split_estratificado(lab, [0, 1], SEMILLA)

    comparar_metodos("Ionosphere", X[:, tr], Y[:, tr], X[:, te], Y[:, te],
        [34, 16, 2], 200, 200, 0.1, 0.6, 1)
end

# ═══════════════════════════════════════════════════════════════════════════════
# 4. ADULT INCOME
# ═══════════════════════════════════════════════════════════════════════════════
const CAT_COLS = [2, 4, 6, 7, 8, 9, 10, 14]
const NUM_COLS = [1, 3, 5, 11, 12, 13]

function parsear_adult(filepath::String, skip_first::Bool=false)
    lines = readlines(filepath)
    if skip_first; lines = lines[2:end]; end
    lines = filter(l -> !isempty(strip(l)) && !occursin("?", l), lines)
    n = length(lines)
    cat_values = Dict{Int, Set{String}}(c => Set{String}() for c in CAT_COLS)
    parsed = Vector{Vector{String}}(undef, n)
    labels = zeros(Int, n)
    for (i, line) in enumerate(lines)
        parts = [strip(p) for p in split(line, ',')]
        length(parts) < 15 && continue
        parsed[i] = parts[1:14]; labels[i] = occursin(">50K", parts[15]) ? 1 : 0
        for c in CAT_COLS; push!(cat_values[c], parts[c]); end
    end
    return parsed, labels, cat_values
end

function codificar_adult(parsed, labels, cat_vals)
    n = length(parsed)
    n_num = length(NUM_COLS)
    n_cat = sum(length(cat_vals[c]) for c in CAT_COLS)
    nf = n_num + n_cat
    features = zeros(Float64, nf, n)
    objetivos = zeros(Float64, 2, n)
    for i in 1:n
        parts = parsed[i]; col = 1
        for c in NUM_COLS; features[col, i] = parse(Float64, parts[c]); col += 1; end
        for c in CAT_COLS
            vals = cat_vals[c]; idx = findfirst(==(parts[c]), vals)
            if idx !== nothing; features[col + idx - 1, i] = 1.0; end
            col += length(vals)
        end
        objetivos[labels[i] + 1, i] = 1.0
    end
    for j in 1:n_num
        mn, mx = extrema(@view features[j, :])
        if mx > mn; features[j, :] .= (features[j, :] .- mn) ./ (mx - mn); end
    end
    return features, objetivos
end

atr = joinpath(@__DIR__, "..", ".cache_adult_train.csv")
ate = joinpath(@__DIR__, "..", ".cache_adult_test.csv")
if isfile(atr) && isfile(ate)
    p_tr, l_tr, cv_tr = parsear_adult(atr, false)
    p_te, l_te, cv_te = parsear_adult(ate, true)
    cv = Dict{Int, Vector{String}}()
    for c in CAT_COLS; cv[c] = sort(collect(union(cv_tr[c], cv_te[c]))); end
    trX, trY = codificar_adult(p_tr, l_tr, cv)
    teX, teY = codificar_adult(p_te, l_te, cv)
    nf = size(trX, 1)

    comparar_metodos("Adult Income", trX, trY, teX, teY,
        [nf, 16, 2], 30, 30, 0.1, 0.6, 2)
end

# ═══════════════════════════════════════════════════════════════════════════════
# 5. IRIS (3 clases)
# ═══════════════════════════════════════════════════════════════════════════════
using MLDatasets: Iris
import DataFrames

ds = Iris(as_df=false)
X = Float64.(ds.features); lab_str = vec(ds.targets)
for f in 1:size(X, 1)
    mn, mx = extrema(@view X[f, :])
    if mx > mn; X[f, :] .= (X[f, :] .- mn) ./ (mx - mn); end
end
clases = sort(unique(lab_str))
ci = Dict(c => i for (i, c) in enumerate(clases))
n = size(X, 2)
Y = zeros(Float64, 3, n)
lab_int = [ci[lab_str[k]] for k in 1:n]
for k in 1:n; Y[lab_int[k], k] = 1.0; end

rng_i = MersenneTwister(SEMILLA)
tri = Int[]; tei = Int[]
for c in clases
    idx = findall(==(c), lab_str); perm = shuffle(rng_i, idx)
    nt = round(Int, 0.8 * length(perm))
    append!(tri, perm[1:nt]); append!(tei, perm[nt+1:end])
end
shuffle!(rng_i, tri); shuffle!(rng_i, tei)

comparar_metodos("Iris", X[:, tri], Y[:, tri], X[:, tei], Y[:, tei],
    [4, 16, 8, 3], 100, 100, 0.1, 0.4, 1)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. WINE (3 clases)
# ═══════════════════════════════════════════════════════════════════════════════
using MLDatasets: Wine

ds = Wine(as_df=false)
X = Float64.(ds.features); lab_str = vec(ds.targets)
for f in 1:size(X, 1)
    mn, mx = extrema(@view X[f, :])
    if mx > mn; X[f, :] .= (X[f, :] .- mn) ./ (mx - mn); end
end
clases = sort(unique(lab_str))
ci = Dict(c => i for (i, c) in enumerate(clases))
n = size(X, 2)
Y = zeros(Float64, 3, n)
lab_int = [ci[lab_str[k]] for k in 1:n]
for k in 1:n; Y[lab_int[k], k] = 1.0; end

rng_w = MersenneTwister(SEMILLA)
tri = Int[]; tei = Int[]
for c in clases
    idx = findall(==(c), lab_str); perm = shuffle(rng_w, idx)
    nt = round(Int, 0.8 * length(perm))
    append!(tri, perm[1:nt]); append!(tei, perm[nt+1:end])
end
shuffle!(rng_w, tri); shuffle!(rng_w, tei)

comparar_metodos("Wine", X[:, tri], Y[:, tri], X[:, tei], Y[:, tei],
    [13, 16, 3], 100, 100, 0.1, 0.4, 1)

# ═══════════════════════════════════════════════════════════════════════════════
# 7. OPTICAL DIGITS (10 clases)
# ═══════════════════════════════════════════════════════════════════════════════
function parsear_digits(filepath)
    lines = filter(l -> !isempty(strip(l)), readlines(filepath))
    n = length(lines)
    features = zeros(Float64, 64, n); labels = zeros(Int, n)
    for (k, line) in enumerate(lines)
        parts = split(line, ',')
        for j in 1:64; features[j, k] = parse(Float64, parts[j]); end
        labels[k] = parse(Int, parts[65])
    end
    return features, labels
end

dtr = joinpath(@__DIR__, "..", ".cache_digits_train.csv")
dte = joinpath(@__DIR__, "..", ".cache_digits_test.csv")
if isfile(dtr) && isfile(dte)
    trX, trL = parsear_digits(dtr); teX, teL = parsear_digits(dte)
    trX ./= 16.0; teX ./= 16.0
    trY = zeros(Float64, 10, size(trX, 2)); teY = zeros(Float64, 10, size(teX, 2))
    for k in 1:size(trX, 2); trY[trL[k]+1, k] = 1.0; end
    for k in 1:size(teX, 2); teY[teL[k]+1, k] = 1.0; end

    comparar_metodos("Optical Digits", trX, trY, teX, teY,
        [64, 32, 10], 100, 100, 0.1, 0.15, 2)
end

println("=" ^ 100)
println("  FIN")
println("=" ^ 100)
