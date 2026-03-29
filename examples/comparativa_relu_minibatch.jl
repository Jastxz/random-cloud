# =============================================================================
# Comparativa: Sigmoid vs ReLU × Sample-by-sample vs Mini-batch
# =============================================================================
#
# Ejecutar con:
#   julia --project=. -t auto examples/comparativa_relu_minibatch.jl
#
# Compara las 4 combinaciones en múltiples datasets para medir el impacto
# de ReLU y mini-batches sobre precisión, compresión y velocidad.
# =============================================================================

using RandomCloud
using RandomCloud: RedNeuronal, feedforward, entrenar!, EntrenarBuffers
using RandomCloud: evaluar, evaluar_regresion, activaciones_por_capa
using Random

# ─── Utilidades ───

function contar_parametros(topologia::Vector{Int})
    sum(topologia[i+1] * topologia[i] + topologia[i+1] for i in 1:length(topologia)-1)
end

function entrenar_clasica(topologia, entradas, objetivos, epocas, lr, semilla;
                          activacion=:sigmoid, batch_size=0)
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

# ─── Dataset XOR ───

println("=" ^ 90)
println("  COMPARATIVA: Sigmoid vs ReLU × SGD vs Mini-batch")
println("=" ^ 90)
println()

xor_X = [0.0 0.0 1.0 1.0; 0.0 1.0 0.0 1.0]
xor_Y = [1.0 0.0 0.0 1.0; 0.0 1.0 1.0 0.0]

configs = [
    (act=:sigmoid, bs=0,  label="Sigmoid + SGD"),
    (act=:sigmoid, bs=32, label="Sigmoid + Mini-batch(32)"),
    (act=:relu,    bs=0,  label="ReLU + SGD"),
    (act=:relu,    bs=32, label="ReLU + Mini-batch(32)"),
]

println("─" ^ 90)
println("  XOR — [2, 8, 4, 2], 2000 épocas, LR=0.5, nube=50")
println("─" ^ 90)

for c in configs
    config = ConfiguracionNube(
        tamano_nube = 50,
        topologia_inicial = [2, 8, 4, 2],
        umbral_acierto = 0.5,
        neuronas_eliminar = 1,
        epocas_refinamiento = 2000,
        tasa_aprendizaje = 0.5,
        semilla = 42,
        activacion = c.act,
        batch_size = c.bs
    )
    motor = MotorNube(config, xor_X, xor_Y)
    informe = ejecutar(motor)
    if informe.exitoso
        params_f = contar_parametros(informe.topologia_final)
        params_i = contar_parametros([2, 8, 4, 2])
        red_pct = round((1.0 - params_f / params_i) * 100, digits=1)
        println("  $(rpad(c.label, 28)) │ $(round(informe.precision*100,digits=1))% │ $(informe.topologia_final) ($params_f params, -$(red_pct)%) │ $(round(informe.tiempo_ejecucion_ms, digits=1))ms")
    else
        println("  $(rpad(c.label, 28)) │ FAIL │ $(round(informe.precision*100,digits=1))% │ $(round(informe.tiempo_ejecucion_ms, digits=1))ms")
    end
end
println()

# ─── Iris ───
println("─" ^ 90)
println("  Cargando Iris...")
println("─" ^ 90)

using MLDatasets: Iris
import DataFrames

dataset_iris = Iris(as_df=false)
iris_X = Float64.(dataset_iris.features)
iris_labels = vec(dataset_iris.targets)

for fila in 1:size(iris_X, 1)
    mn, mx = extrema(@view iris_X[fila, :])
    if mx > mn
        iris_X[fila, :] .= (iris_X[fila, :] .- mn) ./ (mx - mn)
    end
end

clases_iris = sort(unique(iris_labels))
clase_idx_iris = Dict(c => i for (i, c) in enumerate(clases_iris))
n_iris = size(iris_X, 2)
iris_Y = zeros(Float64, 3, n_iris)
for k in 1:n_iris
    iris_Y[clase_idx_iris[iris_labels[k]], k] = 1.0
end

rng_iris = MersenneTwister(42)
train_idx_iris = Int[]
test_idx_iris = Int[]
for c in clases_iris
    idx = findall(==(c), iris_labels)
    perm = shuffle(rng_iris, idx)
    n_train = round(Int, 0.8 * length(perm))
    append!(train_idx_iris, perm[1:n_train])
    append!(test_idx_iris, perm[n_train+1:end])
end
shuffle!(rng_iris, train_idx_iris)
shuffle!(rng_iris, test_idx_iris)

iris_train_X = iris_X[:, train_idx_iris]
iris_train_Y = iris_Y[:, train_idx_iris]
iris_test_X = iris_X[:, test_idx_iris]
iris_test_Y = iris_Y[:, test_idx_iris]

topo_iris = [4, 16, 8, 3]
params_iris = contar_parametros(topo_iris)

println()
println("  Iris — $topo_iris ($params_iris params), 100 épocas, LR=0.1, nube=50")
println()

for c in configs
    # Clásico
    red_c, t_c, acts_c = entrenar_clasica(topo_iris, iris_train_X, iris_train_Y, 100, 0.1, 42;
                                           activacion=c.act)
    if c.act !== :sigmoid
        prec_test_c = evaluar(red_c, iris_test_X, iris_test_Y; acts=acts_c)
    else
        prec_test_c = evaluar(red_c, iris_test_X, iris_test_Y)
    end

    # Nube
    config = ConfiguracionNube(
        tamano_nube = 50,
        topologia_inicial = topo_iris,
        umbral_acierto = 0.4,
        neuronas_eliminar = 1,
        epocas_refinamiento = 100,
        tasa_aprendizaje = 0.1,
        semilla = 42,
        activacion = c.act,
        batch_size = c.bs
    )
    motor = MotorNube(config, iris_train_X, iris_train_Y)
    informe = ejecutar(motor)

    if informe.exitoso
        acts_n = activaciones_por_capa(length(informe.mejor_red.pesos), c.act)
        if c.act !== :sigmoid
            prec_test_n = evaluar(informe.mejor_red, iris_test_X, iris_test_Y; acts=acts_n)
        else
            prec_test_n = evaluar(informe.mejor_red, iris_test_X, iris_test_Y)
        end
        params_f = contar_parametros(informe.topologia_final)
        red_pct = round((1.0 - params_f / params_iris) * 100, digits=1)
        println("  $(rpad(c.label, 28)) │ Clásico: $(round(prec_test_c*100,digits=1))% │ Nube: $(round(prec_test_n*100,digits=1))% → $(informe.topologia_final) (-$(red_pct)%) │ $(round(informe.tiempo_ejecucion_ms, digits=1))ms")
    else
        println("  $(rpad(c.label, 28)) │ Clásico: $(round(prec_test_c*100,digits=1))% │ Nube: FAIL")
    end
end
println()

# ─── Wine ───
println("─" ^ 90)
println("  Cargando Wine...")
println("─" ^ 90)

using MLDatasets: Wine

dataset_wine = Wine(as_df=false)
wine_X = Float64.(dataset_wine.features)
wine_labels = vec(dataset_wine.targets)

for fila in 1:size(wine_X, 1)
    mn, mx = extrema(@view wine_X[fila, :])
    if mx > mn
        wine_X[fila, :] .= (wine_X[fila, :] .- mn) ./ (mx - mn)
    end
end

clases_wine = sort(unique(wine_labels))
clase_idx_wine = Dict(c => i for (i, c) in enumerate(clases_wine))
n_wine = size(wine_X, 2)
wine_Y = zeros(Float64, 3, n_wine)
for k in 1:n_wine
    wine_Y[clase_idx_wine[wine_labels[k]], k] = 1.0
end

rng_wine = MersenneTwister(42)
train_idx_wine = Int[]
test_idx_wine = Int[]
for c in clases_wine
    idx = findall(==(c), wine_labels)
    perm = shuffle(rng_wine, idx)
    n_train = round(Int, 0.8 * length(perm))
    append!(train_idx_wine, perm[1:n_train])
    append!(test_idx_wine, perm[n_train+1:end])
end
shuffle!(rng_wine, train_idx_wine)
shuffle!(rng_wine, test_idx_wine)

wine_train_X = wine_X[:, train_idx_wine]
wine_train_Y = wine_Y[:, train_idx_wine]
wine_test_X = wine_X[:, test_idx_wine]
wine_test_Y = wine_Y[:, test_idx_wine]

topo_wine = [13, 16, 3]
params_wine = contar_parametros(topo_wine)

println()
println("  Wine — $topo_wine ($params_wine params), 100 épocas, LR=0.1, nube=50")
println()

for c in configs
    red_c, t_c, acts_c = entrenar_clasica(topo_wine, wine_train_X, wine_train_Y, 100, 0.1, 42;
                                           activacion=c.act)
    if c.act !== :sigmoid
        prec_test_c = evaluar(red_c, wine_test_X, wine_test_Y; acts=acts_c)
    else
        prec_test_c = evaluar(red_c, wine_test_X, wine_test_Y)
    end

    config = ConfiguracionNube(
        tamano_nube = 50,
        topologia_inicial = topo_wine,
        umbral_acierto = 0.4,
        neuronas_eliminar = 1,
        epocas_refinamiento = 100,
        tasa_aprendizaje = 0.1,
        semilla = 42,
        activacion = c.act,
        batch_size = c.bs
    )
    motor = MotorNube(config, wine_train_X, wine_train_Y)
    informe = ejecutar(motor)

    if informe.exitoso
        acts_n = activaciones_por_capa(length(informe.mejor_red.pesos), c.act)
        if c.act !== :sigmoid
            prec_test_n = evaluar(informe.mejor_red, wine_test_X, wine_test_Y; acts=acts_n)
        else
            prec_test_n = evaluar(informe.mejor_red, wine_test_X, wine_test_Y)
        end
        params_f = contar_parametros(informe.topologia_final)
        red_pct = round((1.0 - params_f / params_wine) * 100, digits=1)
        println("  $(rpad(c.label, 28)) │ Clásico: $(round(prec_test_c*100,digits=1))% │ Nube: $(round(prec_test_n*100,digits=1))% → $(informe.topologia_final) (-$(red_pct)%) │ $(round(informe.tiempo_ejecucion_ms, digits=1))ms")
    else
        println("  $(rpad(c.label, 28)) │ Clásico: $(round(prec_test_c*100,digits=1))% │ Nube: FAIL")
    end
end
println()

# ─── Breast Cancer ───
println("─" ^ 90)
println("  Cargando Breast Cancer...")
println("─" ^ 90)

bc_path = joinpath(@__DIR__, "..", ".cache_breastcancer.csv")
if !isfile(bc_path)
    println("  SKIP: no se encontró .cache_breastcancer.csv")
else
    bc_lines = readlines(bc_path)
    n_bc = length(bc_lines)
    bc_features = zeros(Float64, 30, n_bc)
    bc_labels_raw = zeros(Int, n_bc)
    for (k, line) in enumerate(bc_lines)
        parts = split(line, ',')
        bc_labels_raw[k] = parts[2] == "M" ? 1 : 0
        for j in 1:30
            bc_features[j, k] = parse(Float64, parts[j + 2])
        end
    end

    for fila in 1:30
        mn, mx = extrema(@view bc_features[fila, :])
        if mx > mn
            bc_features[fila, :] .= (bc_features[fila, :] .- mn) ./ (mx - mn)
        end
    end

    bc_Y = zeros(Float64, 2, n_bc)
    for k in 1:n_bc
        bc_Y[bc_labels_raw[k] + 1, k] = 1.0
    end

    rng_bc = MersenneTwister(42)
    bc_train_idx = Int[]
    bc_test_idx = Int[]
    for c in [0, 1]
        idx = findall(==(c), bc_labels_raw)
        perm = shuffle(rng_bc, idx)
        n_train = round(Int, 0.8 * length(perm))
        append!(bc_train_idx, perm[1:n_train])
        append!(bc_test_idx, perm[n_train+1:end])
    end
    shuffle!(rng_bc, bc_train_idx)
    shuffle!(rng_bc, bc_test_idx)

    bc_train_X = bc_features[:, bc_train_idx]
    bc_train_Y = bc_Y[:, bc_train_idx]
    bc_test_X = bc_features[:, bc_test_idx]
    bc_test_Y = bc_Y[:, bc_test_idx]

    topo_bc = [30, 8, 2]
    params_bc = contar_parametros(topo_bc)

    println()
    println("  Breast Cancer — $topo_bc ($params_bc params), 100 épocas, LR=0.1, nube=50")
    println()

    for c in configs
        red_c, t_c, acts_c = entrenar_clasica(topo_bc, bc_train_X, bc_train_Y, 100, 0.1, 42;
                                               activacion=c.act)
        if c.act !== :sigmoid
            prec_test_c = evaluar(red_c, bc_test_X, bc_test_Y; acts=acts_c)
        else
            prec_test_c = evaluar(red_c, bc_test_X, bc_test_Y)
        end

        config = ConfiguracionNube(
            tamano_nube = 50,
            topologia_inicial = topo_bc,
            umbral_acierto = 0.7,
            neuronas_eliminar = 1,
            epocas_refinamiento = 100,
            tasa_aprendizaje = 0.1,
            semilla = 42,
            activacion = c.act,
            batch_size = c.bs
        )
        motor = MotorNube(config, bc_train_X, bc_train_Y)
        informe = ejecutar(motor)

        if informe.exitoso
            acts_n = activaciones_por_capa(length(informe.mejor_red.pesos), c.act)
            if c.act !== :sigmoid
                prec_test_n = evaluar(informe.mejor_red, bc_test_X, bc_test_Y; acts=acts_n)
            else
                prec_test_n = evaluar(informe.mejor_red, bc_test_X, bc_test_Y)
            end
            params_f = contar_parametros(informe.topologia_final)
            red_pct = round((1.0 - params_f / params_bc) * 100, digits=1)
            println("  $(rpad(c.label, 28)) │ Clásico: $(round(prec_test_c*100,digits=1))% │ Nube: $(round(prec_test_n*100,digits=1))% → $(informe.topologia_final) (-$(red_pct)%) │ $(round(informe.tiempo_ejecucion_ms, digits=1))ms")
        else
            println("  $(rpad(c.label, 28)) │ Clásico: $(round(prec_test_c*100,digits=1))% │ Nube: FAIL")
        end
    end
    println()
end

# ─── Boston Housing (Regresión) ───
println("─" ^ 90)
println("  Cargando Boston Housing...")
println("─" ^ 90)

using MLDatasets: BostonHousing

dataset_boston = BostonHousing(as_df=false)
boston_X = Float64.(dataset_boston.features)
boston_targets = Float64.(dataset_boston.targets)

for fila in 1:size(boston_X, 1)
    mn, mx = extrema(@view boston_X[fila, :])
    if mx > mn
        boston_X[fila, :] .= (boston_X[fila, :] .- mn) ./ (mx - mn)
    end
end

t_min, t_max = extrema(boston_targets)
boston_targets_norm = (boston_targets .- t_min) ./ (t_max - t_min)
n_boston = size(boston_X, 2)
boston_Y = reshape(boston_targets_norm, 1, n_boston)

rng_boston = MersenneTwister(42)
perm_boston = randperm(rng_boston, n_boston)
n_train_boston = round(Int, 0.8 * n_boston)
boston_train_X = boston_X[:, perm_boston[1:n_train_boston]]
boston_train_Y = boston_Y[:, perm_boston[1:n_train_boston]]
boston_test_X = boston_X[:, perm_boston[n_train_boston+1:end]]
boston_test_Y = boston_Y[:, perm_boston[n_train_boston+1:end]]

topo_boston = [13, 64, 32, 1]
params_boston = contar_parametros(topo_boston)

println()
println("  Boston Housing — $topo_boston ($params_boston params), 500 épocas, LR=0.01, nube=100")
println("  Métrica: R²")
println()

configs_reg = [
    (act=:sigmoid, bs=0,  label="Sigmoid + SGD"),
    (act=:sigmoid, bs=32, label="Sigmoid + Mini-batch(32)"),
    (act=:relu,    bs=0,  label="ReLU + SGD"),
    (act=:relu,    bs=32, label="ReLU + Mini-batch(32)"),
]

for c in configs_reg
    red_c, t_c, acts_c = entrenar_clasica(topo_boston, boston_train_X, boston_train_Y, 500, 0.01, 42;
                                           activacion=c.act)
    if c.act !== :sigmoid
        r2_test_c = evaluar_regresion(red_c, boston_test_X, boston_test_Y; acts=acts_c)
    else
        r2_test_c = evaluar_regresion(red_c, boston_test_X, boston_test_Y)
    end

    config = ConfiguracionNube(
        tamano_nube = 100,
        topologia_inicial = topo_boston,
        umbral_acierto = 0.05,
        neuronas_eliminar = 1,
        epocas_refinamiento = 500,
        tasa_aprendizaje = 0.01,
        semilla = 42,
        activacion = c.act,
        batch_size = c.bs
    )
    motor = MotorNube(config, boston_train_X, boston_train_Y, evaluar_regresion)
    informe = ejecutar(motor)

    if informe.exitoso
        acts_n = activaciones_por_capa(length(informe.mejor_red.pesos), c.act)
        if c.act !== :sigmoid
            r2_test_n = evaluar_regresion(informe.mejor_red, boston_test_X, boston_test_Y; acts=acts_n)
        else
            r2_test_n = evaluar_regresion(informe.mejor_red, boston_test_X, boston_test_Y)
        end
        params_f = contar_parametros(informe.topologia_final)
        red_pct = round((1.0 - params_f / params_boston) * 100, digits=1)
        println("  $(rpad(c.label, 28)) │ Clásico R²: $(round(r2_test_c, digits=3)) │ Nube R²: $(round(r2_test_n, digits=3)) → $(informe.topologia_final) (-$(red_pct)%) │ $(round(informe.tiempo_ejecucion_ms, digits=1))ms")
    else
        println("  $(rpad(c.label, 28)) │ Clásico R²: $(round(r2_test_c, digits=3)) │ Nube: FAIL ($(round(informe.precision, digits=3)))")
    end
end
println()

# ─── Adult Income ───
println("─" ^ 90)
println("  Cargando Adult Income...")
println("─" ^ 90)

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
        parsed[i] = parts[1:14]
        labels[i] = occursin(">50K", parts[15]) ? 1 : 0
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
        parts = parsed[i]
        col = 1
        for c in NUM_COLS
            features[col, i] = parse(Float64, parts[c]); col += 1
        end
        for c in CAT_COLS
            vals = cat_vals[c]
            idx = findfirst(==(parts[c]), vals)
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

adult_train_path = joinpath(@__DIR__, "..", ".cache_adult_train.csv")
adult_test_path = joinpath(@__DIR__, "..", ".cache_adult_test.csv")

if !isfile(adult_train_path) || !isfile(adult_test_path)
    println("  SKIP: no se encontraron .cache_adult_train.csv / .cache_adult_test.csv")
else
    parsed_train, labels_train, cat_vals_train = parsear_adult(adult_train_path, false)
    parsed_test, labels_test, cat_vals_test = parsear_adult(adult_test_path, true)
    cat_vals = Dict{Int, Vector{String}}()
    for c in CAT_COLS
        cat_vals[c] = sort(collect(union(cat_vals_train[c], cat_vals_test[c])))
    end

    adult_train_X, adult_train_Y = codificar_adult(parsed_train, labels_train, cat_vals)
    adult_test_X, adult_test_Y = codificar_adult(parsed_test, labels_test, cat_vals)
    n_feat_adult = size(adult_train_X, 1)

    topo_adult = [n_feat_adult, 16, 2]
    params_adult = contar_parametros(topo_adult)

    println()
    println("  Adult Income — $topo_adult ($params_adult params), 30 épocas, LR=0.1, nube=50")
    println("  Train: $(size(adult_train_X, 2)) | Test: $(size(adult_test_X, 2))")
    println()

    for c in configs
        red_c, t_c, acts_c = entrenar_clasica(topo_adult, adult_train_X, adult_train_Y, 30, 0.1, 42;
                                               activacion=c.act)
        if c.act !== :sigmoid
            prec_test_c = evaluar(red_c, adult_test_X, adult_test_Y; acts=acts_c)
        else
            prec_test_c = evaluar(red_c, adult_test_X, adult_test_Y)
        end

        config = ConfiguracionNube(
            tamano_nube = 50,
            topologia_inicial = topo_adult,
            umbral_acierto = 0.6,
            neuronas_eliminar = 2,
            epocas_refinamiento = 30,
            tasa_aprendizaje = 0.1,
            semilla = 42,
            activacion = c.act,
            batch_size = c.bs
        )
        motor = MotorNube(config, adult_train_X, adult_train_Y)
        informe = ejecutar(motor)

        if informe.exitoso
            acts_n = activaciones_por_capa(length(informe.mejor_red.pesos), c.act)
            if c.act !== :sigmoid
                prec_test_n = evaluar(informe.mejor_red, adult_test_X, adult_test_Y; acts=acts_n)
            else
                prec_test_n = evaluar(informe.mejor_red, adult_test_X, adult_test_Y)
            end
            params_f = contar_parametros(informe.topologia_final)
            red_pct = round((1.0 - params_f / params_adult) * 100, digits=1)
            println("  $(rpad(c.label, 28)) │ Clásico: $(round(prec_test_c*100,digits=1))% │ Nube: $(round(prec_test_n*100,digits=1))% → $(informe.topologia_final) (-$(red_pct)%) │ $(round(informe.tiempo_ejecucion_ms/1000, digits=1))s")
        else
            println("  $(rpad(c.label, 28)) │ Clásico: $(round(prec_test_c*100,digits=1))% │ Nube: FAIL")
        end
    end
    println()
end

# ─── Optical Digits ───
println("─" ^ 90)
println("  Cargando Optical Digits...")
println("─" ^ 90)

digits_train_path = joinpath(@__DIR__, "..", ".cache_digits_train.csv")
digits_test_path = joinpath(@__DIR__, "..", ".cache_digits_test.csv")

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

if !isfile(digits_train_path) || !isfile(digits_test_path)
    println("  SKIP: no se encontraron .cache_digits_train.csv / .cache_digits_test.csv")
else
    digits_train_features, digits_train_labels = parsear_digits(digits_train_path)
    digits_test_features, digits_test_labels = parsear_digits(digits_test_path)
    digits_train_features ./= 16.0
    digits_test_features ./= 16.0

    n_digits_train = size(digits_train_features, 2)
    n_digits_test = size(digits_test_features, 2)
    digits_train_Y = zeros(Float64, 10, n_digits_train)
    digits_test_Y = zeros(Float64, 10, n_digits_test)
    for k in 1:n_digits_train; digits_train_Y[digits_train_labels[k] + 1, k] = 1.0; end
    for k in 1:n_digits_test; digits_test_Y[digits_test_labels[k] + 1, k] = 1.0; end

    topo_digits = [64, 32, 10]
    params_digits = contar_parametros(topo_digits)

    println()
    println("  Optical Digits — $topo_digits ($params_digits params), 100 épocas, LR=0.1, nube=50")
    println("  Train: $n_digits_train | Test: $n_digits_test")
    println()

    for c in configs
        red_c, t_c, acts_c = entrenar_clasica(topo_digits, digits_train_features, digits_train_Y, 100, 0.1, 42;
                                               activacion=c.act)
        if c.act !== :sigmoid
            prec_test_c = evaluar(red_c, digits_test_features, digits_test_Y; acts=acts_c)
        else
            prec_test_c = evaluar(red_c, digits_test_features, digits_test_Y)
        end

        config = ConfiguracionNube(
            tamano_nube = 50,
            topologia_inicial = topo_digits,
            umbral_acierto = 0.15,
            neuronas_eliminar = 2,
            epocas_refinamiento = 100,
            tasa_aprendizaje = 0.1,
            semilla = 42,
            activacion = c.act,
            batch_size = c.bs
        )
        motor = MotorNube(config, digits_train_features, digits_train_Y)
        informe = ejecutar(motor)

        if informe.exitoso
            acts_n = activaciones_por_capa(length(informe.mejor_red.pesos), c.act)
            if c.act !== :sigmoid
                prec_test_n = evaluar(informe.mejor_red, digits_test_features, digits_test_Y; acts=acts_n)
            else
                prec_test_n = evaluar(informe.mejor_red, digits_test_features, digits_test_Y)
            end
            params_f = contar_parametros(informe.topologia_final)
            red_pct = round((1.0 - params_f / params_digits) * 100, digits=1)
            println("  $(rpad(c.label, 28)) │ Clásico: $(round(prec_test_c*100,digits=1))% │ Nube: $(round(prec_test_n*100,digits=1))% → $(informe.topologia_final) (-$(red_pct)%) │ $(round(informe.tiempo_ejecucion_ms, digits=1))ms")
        else
            println("  $(rpad(c.label, 28)) │ Clásico: $(round(prec_test_c*100,digits=1))% │ Nube: FAIL")
        end
    end
    println()
end

println("=" ^ 90)
println("  FIN")
println("=" ^ 90)
