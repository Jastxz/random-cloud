# =============================================================================
# Comparativa ajustada: ReLU con LR reducido vs Sigmoid baseline
# =============================================================================
#
# ReLU necesita LR más bajo para evitar "dying ReLU".
# Probamos LR=0.01 para ReLU vs LR=0.1 para Sigmoid.
# =============================================================================

using RandomCloud
using RandomCloud: RedNeuronal, feedforward, entrenar!, EntrenarBuffers
using RandomCloud: evaluar, evaluar_regresion, activaciones_por_capa
using Random

function contar_parametros(topologia::Vector{Int})
    sum(topologia[i+1] * topologia[i] + topologia[i+1] for i in 1:length(topologia)-1)
end

function entrenar_clasica(topologia, entradas, objetivos, epocas, lr, semilla;
                          activacion=:sigmoid)
    rng = MersenneTwister(semilla)
    red = RedNeuronal(topologia, rng)
    bufs = EntrenarBuffers(topologia)
    n = size(entradas, 2)
    acts = activaciones_por_capa(length(topologia) - 1, activacion)
    use_acts = activacion !== :sigmoid
    t0 = time_ns()
    if use_acts
        for _ in 1:epocas
            @inbounds for k in 1:n
                entrenar!(red, @view(entradas[:, k]), @view(objetivos[:, k]), lr, bufs, acts)
            end
        end
    else
        for _ in 1:epocas
            @inbounds for k in 1:n
                entrenar!(red, @view(entradas[:, k]), @view(objetivos[:, k]), lr, bufs)
            end
        end
    end
    t1 = time_ns()
    return red, (t1 - t0) / 1_000_000.0, acts
end

println("=" ^ 95)
println("  COMPARATIVA AJUSTADA: Sigmoid(LR=0.1) vs ReLU(LR=0.01) × SGD vs Mini-batch")
println("=" ^ 95)
println()

# Configuraciones: (activacion, batch_size, lr, label)
configs = [
    (act=:sigmoid, bs=0,  lr=0.1,  label="Sigmoid(LR=0.1) + SGD"),
    (act=:sigmoid, bs=32, lr=0.1,  label="Sigmoid(LR=0.1) + MB(32)"),
    (act=:relu,    bs=0,  lr=0.01, label="ReLU(LR=0.01) + SGD"),
    (act=:relu,    bs=32, lr=0.01, label="ReLU(LR=0.01) + MB(32)"),
    (act=:relu,    bs=64, lr=0.01, label="ReLU(LR=0.01) + MB(64)"),
]

# ─── XOR ───
println("─" ^ 95)
println("  XOR — [2, 8, 4, 2], 2000 épocas, nube=50")
println("─" ^ 95)

xor_X = [0.0 0.0 1.0 1.0; 0.0 1.0 0.0 1.0]
xor_Y = [1.0 0.0 0.0 1.0; 0.0 1.0 1.0 0.0]

for c in configs
    config = ConfiguracionNube(
        tamano_nube = 50, topologia_inicial = [2, 8, 4, 2],
        umbral_acierto = 0.5, neuronas_eliminar = 1,
        epocas_refinamiento = 2000, tasa_aprendizaje = c.lr,
        semilla = 42, activacion = c.act, batch_size = c.bs
    )
    motor = MotorNube(config, xor_X, xor_Y)
    informe = ejecutar(motor)
    if informe.exitoso
        pf = contar_parametros(informe.topologia_final)
        red = round((1.0 - pf / 70) * 100, digits=1)
        println("  $(rpad(c.label, 30)) │ $(round(informe.precision*100,digits=1))% │ $(informe.topologia_final) (-$(red)%) │ $(round(informe.tiempo_ejecucion_ms, digits=0))ms")
    else
        println("  $(rpad(c.label, 30)) │ FAIL ($(round(informe.precision*100,digits=1))%) │ $(round(informe.tiempo_ejecucion_ms, digits=0))ms")
    end
end
println()

# ─── Iris ───
using MLDatasets: Iris
import DataFrames

dataset_iris = Iris(as_df=false)
iris_X = Float64.(dataset_iris.features)
iris_labels = vec(dataset_iris.targets)
for fila in 1:size(iris_X, 1)
    mn, mx = extrema(@view iris_X[fila, :])
    if mx > mn; iris_X[fila, :] .= (iris_X[fila, :] .- mn) ./ (mx - mn); end
end
clases_iris = sort(unique(iris_labels))
clase_idx_iris = Dict(c => i for (i, c) in enumerate(clases_iris))
n_iris = size(iris_X, 2)
iris_Y = zeros(Float64, 3, n_iris)
for k in 1:n_iris; iris_Y[clase_idx_iris[iris_labels[k]], k] = 1.0; end

rng_iris = MersenneTwister(42)
train_idx = Int[]; test_idx = Int[]
for c in clases_iris
    idx = findall(==(c), iris_labels); perm = shuffle(rng_iris, idx)
    n_train = round(Int, 0.8 * length(perm))
    append!(train_idx, perm[1:n_train]); append!(test_idx, perm[n_train+1:end])
end
shuffle!(rng_iris, train_idx); shuffle!(rng_iris, test_idx)
iris_trX = iris_X[:, train_idx]; iris_trY = iris_Y[:, train_idx]
iris_teX = iris_X[:, test_idx]; iris_teY = iris_Y[:, test_idx]

println("─" ^ 95)
println("  Iris — [4, 16, 8, 3], 100 épocas, nube=50")
println("─" ^ 95)

for c in configs
    red_c, _, acts_c = entrenar_clasica([4,16,8,3], iris_trX, iris_trY, 100, c.lr, 42; activacion=c.act)
    ptc = c.act !== :sigmoid ? evaluar(red_c, iris_teX, iris_teY; acts=acts_c) : evaluar(red_c, iris_teX, iris_teY)

    config = ConfiguracionNube(tamano_nube=50, topologia_inicial=[4,16,8,3],
        umbral_acierto=0.4, neuronas_eliminar=1, epocas_refinamiento=100,
        tasa_aprendizaje=c.lr, semilla=42, activacion=c.act, batch_size=c.bs)
    motor = MotorNube(config, iris_trX, iris_trY)
    inf = ejecutar(motor)
    if inf.exitoso
        acts_n = activaciones_por_capa(length(inf.mejor_red.pesos), c.act)
        ptn = c.act !== :sigmoid ? evaluar(inf.mejor_red, iris_teX, iris_teY; acts=acts_n) : evaluar(inf.mejor_red, iris_teX, iris_teY)
        pf = contar_parametros(inf.topologia_final)
        red = round((1.0 - pf / 243) * 100, digits=1)
        println("  $(rpad(c.label, 30)) │ C: $(round(ptc*100,digits=1))% │ N: $(round(ptn*100,digits=1))% → $(inf.topologia_final) (-$(red)%) │ $(round(inf.tiempo_ejecucion_ms, digits=0))ms")
    else
        println("  $(rpad(c.label, 30)) │ C: $(round(ptc*100,digits=1))% │ N: FAIL")
    end
end
println()

# ─── Wine ───
using MLDatasets: Wine
dataset_wine = Wine(as_df=false)
wine_X = Float64.(dataset_wine.features)
wine_labels = vec(dataset_wine.targets)
for fila in 1:size(wine_X, 1)
    mn, mx = extrema(@view wine_X[fila, :])
    if mx > mn; wine_X[fila, :] .= (wine_X[fila, :] .- mn) ./ (mx - mn); end
end
clases_wine = sort(unique(wine_labels))
clase_idx_wine = Dict(c => i for (i, c) in enumerate(clases_wine))
n_wine = size(wine_X, 2)
wine_Y = zeros(Float64, 3, n_wine)
for k in 1:n_wine; wine_Y[clase_idx_wine[wine_labels[k]], k] = 1.0; end

rng_wine = MersenneTwister(42)
wtr = Int[]; wte = Int[]
for c in clases_wine
    idx = findall(==(c), wine_labels); perm = shuffle(rng_wine, idx)
    n_train = round(Int, 0.8 * length(perm))
    append!(wtr, perm[1:n_train]); append!(wte, perm[n_train+1:end])
end
shuffle!(rng_wine, wtr); shuffle!(rng_wine, wte)
wine_trX = wine_X[:, wtr]; wine_trY = wine_Y[:, wtr]
wine_teX = wine_X[:, wte]; wine_teY = wine_Y[:, wte]

println("─" ^ 95)
println("  Wine — [13, 16, 3], 100 épocas, nube=50")
println("─" ^ 95)

for c in configs
    red_c, _, acts_c = entrenar_clasica([13,16,3], wine_trX, wine_trY, 100, c.lr, 42; activacion=c.act)
    ptc = c.act !== :sigmoid ? evaluar(red_c, wine_teX, wine_teY; acts=acts_c) : evaluar(red_c, wine_teX, wine_teY)

    config = ConfiguracionNube(tamano_nube=50, topologia_inicial=[13,16,3],
        umbral_acierto=0.4, neuronas_eliminar=1, epocas_refinamiento=100,
        tasa_aprendizaje=c.lr, semilla=42, activacion=c.act, batch_size=c.bs)
    motor = MotorNube(config, wine_trX, wine_trY)
    inf = ejecutar(motor)
    if inf.exitoso
        acts_n = activaciones_por_capa(length(inf.mejor_red.pesos), c.act)
        ptn = c.act !== :sigmoid ? evaluar(inf.mejor_red, wine_teX, wine_teY; acts=acts_n) : evaluar(inf.mejor_red, wine_teX, wine_teY)
        pf = contar_parametros(inf.topologia_final)
        red = round((1.0 - pf / 275) * 100, digits=1)
        println("  $(rpad(c.label, 30)) │ C: $(round(ptc*100,digits=1))% │ N: $(round(ptn*100,digits=1))% → $(inf.topologia_final) (-$(red)%) │ $(round(inf.tiempo_ejecucion_ms, digits=0))ms")
    else
        println("  $(rpad(c.label, 30)) │ C: $(round(ptc*100,digits=1))% │ N: FAIL")
    end
end
println()

# ─── Breast Cancer ───
bc_path = joinpath(@__DIR__, "..", ".cache_breastcancer.csv")
if isfile(bc_path)
    bc_lines = readlines(bc_path)
    n_bc = length(bc_lines)
    bc_X = zeros(Float64, 30, n_bc); bc_lab = zeros(Int, n_bc)
    for (k, line) in enumerate(bc_lines)
        parts = split(line, ',')
        bc_lab[k] = parts[2] == "M" ? 1 : 0
        for j in 1:30; bc_X[j, k] = parse(Float64, parts[j + 2]); end
    end
    for fila in 1:30
        mn, mx = extrema(@view bc_X[fila, :])
        if mx > mn; bc_X[fila, :] .= (bc_X[fila, :] .- mn) ./ (mx - mn); end
    end
    bc_Y = zeros(Float64, 2, n_bc)
    for k in 1:n_bc; bc_Y[bc_lab[k] + 1, k] = 1.0; end

    rng_bc = MersenneTwister(42)
    btr = Int[]; bte = Int[]
    for c in [0, 1]
        idx = findall(==(c), bc_lab); perm = shuffle(rng_bc, idx)
        n_train = round(Int, 0.8 * length(perm))
        append!(btr, perm[1:n_train]); append!(bte, perm[n_train+1:end])
    end
    shuffle!(rng_bc, btr); shuffle!(rng_bc, bte)
    bc_trX = bc_X[:, btr]; bc_trY = bc_Y[:, btr]
    bc_teX = bc_X[:, bte]; bc_teY = bc_Y[:, bte]

    println("─" ^ 95)
    println("  Breast Cancer — [30, 8, 2], 100 épocas, nube=50")
    println("─" ^ 95)

    for c in configs
        red_c, _, acts_c = entrenar_clasica([30,8,2], bc_trX, bc_trY, 100, c.lr, 42; activacion=c.act)
        ptc = c.act !== :sigmoid ? evaluar(red_c, bc_teX, bc_teY; acts=acts_c) : evaluar(red_c, bc_teX, bc_teY)

        config = ConfiguracionNube(tamano_nube=50, topologia_inicial=[30,8,2],
            umbral_acierto=0.7, neuronas_eliminar=1, epocas_refinamiento=100,
            tasa_aprendizaje=c.lr, semilla=42, activacion=c.act, batch_size=c.bs)
        motor = MotorNube(config, bc_trX, bc_trY)
        inf = ejecutar(motor)
        if inf.exitoso
            acts_n = activaciones_por_capa(length(inf.mejor_red.pesos), c.act)
            ptn = c.act !== :sigmoid ? evaluar(inf.mejor_red, bc_teX, bc_teY; acts=acts_n) : evaluar(inf.mejor_red, bc_teX, bc_teY)
            pf = contar_parametros(inf.topologia_final)
            red = round((1.0 - pf / 266) * 100, digits=1)
            println("  $(rpad(c.label, 30)) │ C: $(round(ptc*100,digits=1))% │ N: $(round(ptn*100,digits=1))% → $(inf.topologia_final) (-$(red)%) │ $(round(inf.tiempo_ejecucion_ms, digits=0))ms")
        else
            println("  $(rpad(c.label, 30)) │ C: $(round(ptc*100,digits=1))% │ N: FAIL")
        end
    end
    println()
end

# ─── Boston Housing (Regresión) ───
using MLDatasets: BostonHousing
dataset_boston = BostonHousing(as_df=false)
bos_X = Float64.(dataset_boston.features)
bos_targets = Float64.(dataset_boston.targets)
for fila in 1:size(bos_X, 1)
    mn, mx = extrema(@view bos_X[fila, :])
    if mx > mn; bos_X[fila, :] .= (bos_X[fila, :] .- mn) ./ (mx - mn); end
end
t_min, t_max = extrema(bos_targets)
bos_Y = reshape((bos_targets .- t_min) ./ (t_max - t_min), 1, size(bos_X, 2))

rng_bos = MersenneTwister(42)
perm_bos = randperm(rng_bos, size(bos_X, 2))
n_tr_bos = round(Int, 0.8 * size(bos_X, 2))
bos_trX = bos_X[:, perm_bos[1:n_tr_bos]]; bos_trY = bos_Y[:, perm_bos[1:n_tr_bos]]
bos_teX = bos_X[:, perm_bos[n_tr_bos+1:end]]; bos_teY = bos_Y[:, perm_bos[n_tr_bos+1:end]]

println("─" ^ 95)
println("  Boston Housing — [13, 64, 32, 1], 500 épocas, nube=100 (R²)")
println("─" ^ 95)

# Para regresión con ReLU, usamos LR=0.001 (más conservador)
configs_reg = [
    (act=:sigmoid, bs=0,  lr=0.01,  label="Sigmoid(LR=0.01) + SGD"),
    (act=:sigmoid, bs=32, lr=0.01,  label="Sigmoid(LR=0.01) + MB(32)"),
    (act=:relu,    bs=0,  lr=0.001, label="ReLU(LR=0.001) + SGD"),
    (act=:relu,    bs=32, lr=0.001, label="ReLU(LR=0.001) + MB(32)"),
    (act=:relu,    bs=64, lr=0.001, label="ReLU(LR=0.001) + MB(64)"),
]

for c in configs_reg
    red_c, _, acts_c = entrenar_clasica([13,64,32,1], bos_trX, bos_trY, 500, c.lr, 42; activacion=c.act)
    r2c = c.act !== :sigmoid ? evaluar_regresion(red_c, bos_teX, bos_teY; acts=acts_c) : evaluar_regresion(red_c, bos_teX, bos_teY)

    config = ConfiguracionNube(tamano_nube=100, topologia_inicial=[13,64,32,1],
        umbral_acierto=0.05, neuronas_eliminar=1, epocas_refinamiento=500,
        tasa_aprendizaje=c.lr, semilla=42, activacion=c.act, batch_size=c.bs)
    motor = MotorNube(config, bos_trX, bos_trY, evaluar_regresion)
    inf = ejecutar(motor)
    if inf.exitoso
        acts_n = activaciones_por_capa(length(inf.mejor_red.pesos), c.act)
        r2n = c.act !== :sigmoid ? evaluar_regresion(inf.mejor_red, bos_teX, bos_teY; acts=acts_n) : evaluar_regresion(inf.mejor_red, bos_teX, bos_teY)
        pf = contar_parametros(inf.topologia_final)
        red = round((1.0 - pf / 3009) * 100, digits=1)
        println("  $(rpad(c.label, 30)) │ C R²: $(round(r2c, digits=3)) │ N R²: $(round(r2n, digits=3)) → $(inf.topologia_final) (-$(red)%) │ $(round(inf.tiempo_ejecucion_ms/1000, digits=1))s")
    else
        println("  $(rpad(c.label, 30)) │ C R²: $(round(r2c, digits=3)) │ N: FAIL ($(round(inf.precision, digits=3)))")
    end
end
println()

# ─── Adult Income ───
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
    n_features = n_num + n_cat
    features = zeros(Float64, n_features, n)
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

atr_path = joinpath(@__DIR__, "..", ".cache_adult_train.csv")
ate_path = joinpath(@__DIR__, "..", ".cache_adult_test.csv")

if isfile(atr_path) && isfile(ate_path)
    p_tr, l_tr, cv_tr = parsear_adult(atr_path, false)
    p_te, l_te, cv_te = parsear_adult(ate_path, true)
    cv = Dict{Int, Vector{String}}()
    for c in CAT_COLS; cv[c] = sort(collect(union(cv_tr[c], cv_te[c]))); end
    a_trX, a_trY = codificar_adult(p_tr, l_tr, cv)
    a_teX, a_teY = codificar_adult(p_te, l_te, cv)
    nf = size(a_trX, 1)

    println("─" ^ 95)
    println("  Adult Income — [$nf, 16, 2], 30 épocas, nube=50")
    println("─" ^ 95)

    for c in configs
        red_c, _, acts_c = entrenar_clasica([nf,16,2], a_trX, a_trY, 30, c.lr, 42; activacion=c.act)
        ptc = c.act !== :sigmoid ? evaluar(red_c, a_teX, a_teY; acts=acts_c) : evaluar(red_c, a_teX, a_teY)

        config = ConfiguracionNube(tamano_nube=50, topologia_inicial=[nf,16,2],
            umbral_acierto=0.6, neuronas_eliminar=2, epocas_refinamiento=30,
            tasa_aprendizaje=c.lr, semilla=42, activacion=c.act, batch_size=c.bs)
        motor = MotorNube(config, a_trX, a_trY)
        inf = ejecutar(motor)
        if inf.exitoso
            acts_n = activaciones_por_capa(length(inf.mejor_red.pesos), c.act)
            ptn = c.act !== :sigmoid ? evaluar(inf.mejor_red, a_teX, a_teY; acts=acts_n) : evaluar(inf.mejor_red, a_teX, a_teY)
            pf = contar_parametros(inf.topologia_final)
            pi = contar_parametros([nf,16,2])
            red = round((1.0 - pf / pi) * 100, digits=1)
            println("  $(rpad(c.label, 30)) │ C: $(round(ptc*100,digits=1))% │ N: $(round(ptn*100,digits=1))% → $(inf.topologia_final) (-$(red)%) │ $(round(inf.tiempo_ejecucion_ms/1000, digits=1))s")
        else
            println("  $(rpad(c.label, 30)) │ C: $(round(ptc*100,digits=1))% │ N: FAIL")
        end
    end
    println()
end

# ─── Optical Digits ───
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

dtr_path = joinpath(@__DIR__, "..", ".cache_digits_train.csv")
dte_path = joinpath(@__DIR__, "..", ".cache_digits_test.csv")

if isfile(dtr_path) && isfile(dte_path)
    d_trX, d_trL = parsear_digits(dtr_path)
    d_teX, d_teL = parsear_digits(dte_path)
    d_trX ./= 16.0; d_teX ./= 16.0
    d_trY = zeros(Float64, 10, size(d_trX, 2))
    d_teY = zeros(Float64, 10, size(d_teX, 2))
    for k in 1:size(d_trX, 2); d_trY[d_trL[k]+1, k] = 1.0; end
    for k in 1:size(d_teX, 2); d_teY[d_teL[k]+1, k] = 1.0; end

    println("─" ^ 95)
    println("  Optical Digits — [64, 32, 10], 100 épocas, nube=50")
    println("─" ^ 95)

    for c in configs
        red_c, _, acts_c = entrenar_clasica([64,32,10], d_trX, d_trY, 100, c.lr, 42; activacion=c.act)
        ptc = c.act !== :sigmoid ? evaluar(red_c, d_teX, d_teY; acts=acts_c) : evaluar(red_c, d_teX, d_teY)

        config = ConfiguracionNube(tamano_nube=50, topologia_inicial=[64,32,10],
            umbral_acierto=0.15, neuronas_eliminar=2, epocas_refinamiento=100,
            tasa_aprendizaje=c.lr, semilla=42, activacion=c.act, batch_size=c.bs)
        motor = MotorNube(config, d_trX, d_trY)
        inf = ejecutar(motor)
        if inf.exitoso
            acts_n = activaciones_por_capa(length(inf.mejor_red.pesos), c.act)
            ptn = c.act !== :sigmoid ? evaluar(inf.mejor_red, d_teX, d_teY; acts=acts_n) : evaluar(inf.mejor_red, d_teX, d_teY)
            pf = contar_parametros(inf.topologia_final)
            red = round((1.0 - pf / 2410) * 100, digits=1)
            println("  $(rpad(c.label, 30)) │ C: $(round(ptc*100,digits=1))% │ N: $(round(ptn*100,digits=1))% → $(inf.topologia_final) (-$(red)%) │ $(round(inf.tiempo_ejecucion_ms, digits=0))ms")
        else
            println("  $(rpad(c.label, 30)) │ C: $(round(ptc*100,digits=1))% │ N: FAIL")
        end
    end
    println()
end

println("=" ^ 95)
println("  FIN")
println("=" ^ 95)
