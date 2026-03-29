# =============================================================================
# Evaluación F1 y AUC: Nube Aleatoria vs Clásico en todos los datasets
# =============================================================================
#
# Ejecutar con:
#   julia --project=. -t auto examples/metricas_f1_auc.jl
#
# Evalúa Accuracy, F1-score (macro) y AUC-ROC en cada dataset,
# comparando la red clásica vs la red encontrada por la nube.
# =============================================================================

using RandomCloud
using RandomCloud: RedNeuronal, entrenar!, EntrenarBuffers
using RandomCloud: evaluar, evaluar_f1, evaluar_auc, evaluar_regresion
using Random
using Downloads: download

function contar_parametros(topo::Vector{Int})
    sum(topo[i+1] * topo[i] + topo[i+1] for i in 1:length(topo)-1)
end

function entrenar_clasica(topo, X, Y, epocas, lr, semilla)
    rng = MersenneTwister(semilla)
    red = RedNeuronal(topo, rng)
    bufs = EntrenarBuffers(topo)
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

function imprimir_metricas(label, acc, f1, auc; extra="")
    println("  $(rpad(label, 12)) │ Acc: $(rpad(string(round(acc*100,digits=1))*"%", 7)) │ F1: $(rpad(string(round(f1, digits=3)), 5)) │ AUC: $(rpad(string(round(auc, digits=3)), 5))$extra")
end

# Estratificado split helper
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

const SEMILLA = 42

println("=" ^ 85)
println("  MÉTRICAS: Accuracy / F1-Score / AUC-ROC — Nube vs Clásico")
println("=" ^ 85)
println()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. BREAST CANCER — 2 clases (benigno/maligno), desbalanceado (63%/37%)
# ═══════════════════════════════════════════════════════════════════════════════
bc_path = joinpath(@__DIR__, "..", ".cache_breastcancer.csv")
if isfile(bc_path)
    println("─" ^ 85)
    println("  Breast Cancer — [30, 8, 2], 100 épocas, LR=0.1")
    println("─" ^ 85)

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
    trX, trY = X[:, tr], Y[:, tr]
    teX, teY = X[:, te], Y[:, te]

    pct_m = round(sum(lab[te] .== 1) / length(te) * 100, digits=1)
    println("  Test: $(length(te)) muestras ($(pct_m)% maligno)")
    println()

    # Clásico
    red_c = entrenar_clasica([30, 8, 2], trX, trY, 100, 0.1, SEMILLA)
    acc_c, f1_c, auc_c = evaluar_todo(red_c, teX, teY)
    imprimir_metricas("Clásico", acc_c, f1_c, auc_c)

    # Nube
    config = ConfiguracionNube(tamano_nube=50, topologia_inicial=[30, 8, 2],
        umbral_acierto=0.7, neuronas_eliminar=1, epocas_refinamiento=100,
        tasa_aprendizaje=0.1, semilla=SEMILLA)
    motor = MotorNube(config, trX, trY)
    informe = ejecutar(motor)
    if informe.exitoso
        acc_n, f1_n, auc_n = evaluar_todo(informe.mejor_red, teX, teY)
        pf = contar_parametros(informe.topologia_final)
        red_pct = round((1.0 - pf / 266) * 100, digits=1)
        imprimir_metricas("Nube", acc_n, f1_n, auc_n; extra=" → $(informe.topologia_final) (-$(red_pct)%)")
    else
        println("  Nube:        FAIL")
    end
    println()
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SONAR — 2 clases (minas/rocas), casi balanceado (53%/47%)
# ═══════════════════════════════════════════════════════════════════════════════
sonar_path = joinpath(@__DIR__, "..", ".cache_sonar.csv")
if isfile(sonar_path)
    println("─" ^ 85)
    println("  Sonar — [60, 8, 2], 200 épocas, LR=0.1")
    println("─" ^ 85)

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
    trX, trY = X[:, tr], Y[:, tr]
    teX, teY = X[:, te], Y[:, te]

    pct_m = round(sum(lab[te] .== 1) / length(te) * 100, digits=1)
    println("  Test: $(length(te)) muestras ($(pct_m)% minas)")
    println()

    red_c = entrenar_clasica([60, 8, 2], trX, trY, 200, 0.1, SEMILLA)
    acc_c, f1_c, auc_c = evaluar_todo(red_c, teX, teY)
    imprimir_metricas("Clásico", acc_c, f1_c, auc_c)

    config = ConfiguracionNube(tamano_nube=50, topologia_inicial=[60, 8, 2],
        umbral_acierto=0.55, neuronas_eliminar=1, epocas_refinamiento=200,
        tasa_aprendizaje=0.1, semilla=SEMILLA)
    motor = MotorNube(config, trX, trY)
    informe = ejecutar(motor)
    if informe.exitoso
        acc_n, f1_n, auc_n = evaluar_todo(informe.mejor_red, teX, teY)
        pf = contar_parametros(informe.topologia_final)
        red_pct = round((1.0 - pf / 506) * 100, digits=1)
        imprimir_metricas("Nube", acc_n, f1_n, auc_n; extra=" → $(informe.topologia_final) (-$(red_pct)%)")
    else
        println("  Nube:        FAIL")
    end
    println()
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3. IONOSPHERE — 2 clases (good/bad), desbalanceado (64%/36%)
# ═══════════════════════════════════════════════════════════════════════════════
iono_path = joinpath(@__DIR__, "..", ".cache_ionosphere.csv")
if isfile(iono_path)
    println("─" ^ 85)
    println("  Ionosphere — [34, 16, 2], 200 épocas, LR=0.1")
    println("─" ^ 85)

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
    trX, trY = X[:, tr], Y[:, tr]
    teX, teY = X[:, te], Y[:, te]

    pct_g = round(sum(lab[te] .== 1) / length(te) * 100, digits=1)
    println("  Test: $(length(te)) muestras ($(pct_g)% good)")
    println()

    red_c = entrenar_clasica([34, 16, 2], trX, trY, 200, 0.1, SEMILLA)
    acc_c, f1_c, auc_c = evaluar_todo(red_c, teX, teY)
    imprimir_metricas("Clásico", acc_c, f1_c, auc_c)

    config = ConfiguracionNube(tamano_nube=50, topologia_inicial=[34, 16, 2],
        umbral_acierto=0.6, neuronas_eliminar=1, epocas_refinamiento=200,
        tasa_aprendizaje=0.1, semilla=SEMILLA)
    motor = MotorNube(config, trX, trY)
    informe = ejecutar(motor)
    if informe.exitoso
        acc_n, f1_n, auc_n = evaluar_todo(informe.mejor_red, teX, teY)
        pf = contar_parametros(informe.topologia_final)
        red_pct = round((1.0 - pf / 594) * 100, digits=1)
        imprimir_metricas("Nube", acc_n, f1_n, auc_n; extra=" → $(informe.topologia_final) (-$(red_pct)%)")
    else
        println("  Nube:        FAIL")
    end
    println()
end

# ═══════════════════════════════════════════════════════════════════════════════
# 4. ADULT INCOME — 2 clases (>50K/≤50K), desbalanceado (75%/25%)
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
    println("─" ^ 85)
    println("  Adult Income — [104, 16, 2], 30 épocas, LR=0.1")
    println("─" ^ 85)

    p_tr, l_tr, cv_tr = parsear_adult(atr, false)
    p_te, l_te, cv_te = parsear_adult(ate, true)
    cv = Dict{Int, Vector{String}}()
    for c in CAT_COLS; cv[c] = sort(collect(union(cv_tr[c], cv_te[c]))); end
    trX, trY = codificar_adult(p_tr, l_tr, cv)
    teX, teY = codificar_adult(p_te, l_te, cv)
    nf = size(trX, 1)

    pct_pos = round(sum(l_te) / length(l_te) * 100, digits=1)
    println("  Train: $(size(trX,2)) | Test: $(size(teX,2)) ($(pct_pos)% >50K)")
    println()

    red_c = entrenar_clasica([nf, 16, 2], trX, trY, 30, 0.1, SEMILLA)
    acc_c, f1_c, auc_c = evaluar_todo(red_c, teX, teY)
    imprimir_metricas("Clásico", acc_c, f1_c, auc_c)

    config = ConfiguracionNube(tamano_nube=50, topologia_inicial=[nf, 16, 2],
        umbral_acierto=0.6, neuronas_eliminar=2, epocas_refinamiento=30,
        tasa_aprendizaje=0.1, semilla=SEMILLA)
    motor = MotorNube(config, trX, trY)
    informe = ejecutar(motor)
    if informe.exitoso
        acc_n, f1_n, auc_n = evaluar_todo(informe.mejor_red, teX, teY)
        pi = contar_parametros([nf, 16, 2])
        pf = contar_parametros(informe.topologia_final)
        red_pct = round((1.0 - pf / pi) * 100, digits=1)
        imprimir_metricas("Nube", acc_n, f1_n, auc_n; extra=" → $(informe.topologia_final) (-$(red_pct)%)")
    else
        println("  Nube:        FAIL")
    end
    println()
end

# ═══════════════════════════════════════════════════════════════════════════════
# 5. IRIS — 3 clases (balanceado, 33% cada una)
# ═══════════════════════════════════════════════════════════════════════════════
using MLDatasets: Iris
import DataFrames

println("─" ^ 85)
println("  Iris — [4, 16, 8, 3], 100 épocas, LR=0.1")
println("─" ^ 85)

ds = Iris(as_df=false)
X = Float64.(ds.features); lab = vec(ds.targets)
for f in 1:size(X, 1)
    mn, mx = extrema(@view X[f, :])
    if mx > mn; X[f, :] .= (X[f, :] .- mn) ./ (mx - mn); end
end
clases = sort(unique(lab))
ci = Dict(c => i for (i, c) in enumerate(clases))
n = size(X, 2)
Y = zeros(Float64, 3, n)
for k in 1:n; Y[ci[lab[k]], k] = 1.0; end

rng_i = MersenneTwister(SEMILLA)
tri = Int[]; tei = Int[]
for c in clases
    idx = findall(==(c), lab); perm = shuffle(rng_i, idx)
    nt = round(Int, 0.8 * length(perm))
    append!(tri, perm[1:nt]); append!(tei, perm[nt+1:end])
end
shuffle!(rng_i, tri); shuffle!(rng_i, tei)
trX, trY = X[:, tri], Y[:, tri]
teX, teY = X[:, tei], Y[:, tei]

println("  Test: $(length(tei)) muestras (3 clases balanceadas)")
println()

red_c = entrenar_clasica([4, 16, 8, 3], trX, trY, 100, 0.1, SEMILLA)
acc_c, f1_c, auc_c = evaluar_todo(red_c, teX, teY)
imprimir_metricas("Clásico", acc_c, f1_c, auc_c)

config = ConfiguracionNube(tamano_nube=50, topologia_inicial=[4, 16, 8, 3],
    umbral_acierto=0.4, neuronas_eliminar=1, epocas_refinamiento=100,
    tasa_aprendizaje=0.1, semilla=SEMILLA)
motor = MotorNube(config, trX, trY)
informe = ejecutar(motor)
if informe.exitoso
    acc_n, f1_n, auc_n = evaluar_todo(informe.mejor_red, teX, teY)
    pf = contar_parametros(informe.topologia_final)
    red_pct = round((1.0 - pf / 243) * 100, digits=1)
    imprimir_metricas("Nube", acc_n, f1_n, auc_n; extra=" → $(informe.topologia_final) (-$(red_pct)%)")
else
    println("  Nube:        FAIL")
end
println()

# ═══════════════════════════════════════════════════════════════════════════════
# 6. WINE — 3 clases (desbalanceado: 33%/40%/27%)
# ═══════════════════════════════════════════════════════════════════════════════
using MLDatasets: Wine

println("─" ^ 85)
println("  Wine — [13, 16, 3], 100 épocas, LR=0.1")
println("─" ^ 85)

ds = Wine(as_df=false)
X = Float64.(ds.features); lab = vec(ds.targets)
for f in 1:size(X, 1)
    mn, mx = extrema(@view X[f, :])
    if mx > mn; X[f, :] .= (X[f, :] .- mn) ./ (mx - mn); end
end
clases = sort(unique(lab))
ci = Dict(c => i for (i, c) in enumerate(clases))
n = size(X, 2)
Y = zeros(Float64, 3, n)
for k in 1:n; Y[ci[lab[k]], k] = 1.0; end

rng_w = MersenneTwister(SEMILLA)
tri = Int[]; tei = Int[]
for c in clases
    idx = findall(==(c), lab); perm = shuffle(rng_w, idx)
    nt = round(Int, 0.8 * length(perm))
    append!(tri, perm[1:nt]); append!(tei, perm[nt+1:end])
end
shuffle!(rng_w, tri); shuffle!(rng_w, tei)
trX, trY = X[:, tri], Y[:, tri]
teX, teY = X[:, tei], Y[:, tei]

println("  Test: $(length(tei)) muestras (3 clases)")
println()

red_c = entrenar_clasica([13, 16, 3], trX, trY, 100, 0.1, SEMILLA)
acc_c, f1_c, auc_c = evaluar_todo(red_c, teX, teY)
imprimir_metricas("Clásico", acc_c, f1_c, auc_c)

config = ConfiguracionNube(tamano_nube=50, topologia_inicial=[13, 16, 3],
    umbral_acierto=0.4, neuronas_eliminar=1, epocas_refinamiento=100,
    tasa_aprendizaje=0.1, semilla=SEMILLA)
motor = MotorNube(config, trX, trY)
informe = ejecutar(motor)
if informe.exitoso
    acc_n, f1_n, auc_n = evaluar_todo(informe.mejor_red, teX, teY)
    pf = contar_parametros(informe.topologia_final)
    red_pct = round((1.0 - pf / 275) * 100, digits=1)
    imprimir_metricas("Nube", acc_n, f1_n, auc_n; extra=" → $(informe.topologia_final) (-$(red_pct)%)")
else
    println("  Nube:        FAIL")
end
println()

# ═══════════════════════════════════════════════════════════════════════════════
# 7. OPTICAL DIGITS — 10 clases (balanceado)
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
    println("─" ^ 85)
    println("  Optical Digits — [64, 32, 10], 100 épocas, LR=0.1")
    println("─" ^ 85)

    trX, trL = parsear_digits(dtr)
    teX, teL = parsear_digits(dte)
    trX ./= 16.0; teX ./= 16.0
    trY = zeros(Float64, 10, size(trX, 2))
    teY = zeros(Float64, 10, size(teX, 2))
    for k in 1:size(trX, 2); trY[trL[k]+1, k] = 1.0; end
    for k in 1:size(teX, 2); teY[teL[k]+1, k] = 1.0; end

    println("  Train: $(size(trX,2)) | Test: $(size(teX,2)) (10 clases)")
    println()

    red_c = entrenar_clasica([64, 32, 10], trX, trY, 100, 0.1, SEMILLA)
    acc_c, f1_c, auc_c = evaluar_todo(red_c, teX, teY)
    imprimir_metricas("Clásico", acc_c, f1_c, auc_c)

    config = ConfiguracionNube(tamano_nube=50, topologia_inicial=[64, 32, 10],
        umbral_acierto=0.15, neuronas_eliminar=2, epocas_refinamiento=100,
        tasa_aprendizaje=0.1, semilla=SEMILLA)
    motor = MotorNube(config, trX, trY)
    informe = ejecutar(motor)
    if informe.exitoso
        acc_n, f1_n, auc_n = evaluar_todo(informe.mejor_red, teX, teY)
        pf = contar_parametros(informe.topologia_final)
        red_pct = round((1.0 - pf / 2410) * 100, digits=1)
        imprimir_metricas("Nube", acc_n, f1_n, auc_n; extra=" → $(informe.topologia_final) (-$(red_pct)%)")
    else
        println("  Nube:        FAIL")
    end
    println()
end

println("=" ^ 85)
println("  FIN")
println("=" ^ 85)
