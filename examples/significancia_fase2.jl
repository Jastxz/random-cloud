# =============================================================================
# Significance tests: Phase 1 vs Phase 1+2 (10 seeds × Wilcoxon)
# =============================================================================
#
# julia --project=. examples/significancia_fase2.jl

using RandomCloud
using RandomCloud: RedNeuronal, evaluar
using MLDatasets: Iris
import DataFrames
using Random
using DelimitedFiles
using SpecialFunctions: erfc

const N_SEEDS = 10
const BASE_SEED = 42

# ─── Utilidades (mismas que fase2_test_set.jl) ───

function normalizar_minmax!(X::Matrix{Float64})
    for fila in axes(X, 1)
        mn, mx = extrema(@view X[fila, :])
        if mx > mn
            X[fila, :] .= (X[fila, :] .- mn) ./ (mx - mn)
        end
    end
end

function onehot(labels, clases)
    n = length(labels)
    nc = length(clases)
    clase_idx = Dict(c => i for (i, c) in enumerate(clases))
    Y = zeros(Float64, nc, n)
    for k in 1:n
        Y[clase_idx[labels[k]], k] = 1.0
    end
    return Y
end

function split_estratificado(labels, ratio_train=0.8; seed=42)
    rng = MersenneTwister(seed)
    clases = sort(unique(labels))
    train_idx = Int[]
    test_idx = Int[]
    for c in clases
        idx_clase = findall(==(c), labels)
        perm = shuffle(rng, idx_clase)
        n_train = round(Int, ratio_train * length(perm))
        append!(train_idx, perm[1:n_train])
        append!(test_idx, perm[n_train+1:end])
    end
    shuffle!(rng, train_idx)
    shuffle!(rng, test_idx)
    return train_idx, test_idx
end

function contar_parametros(topo::Vector{Int})
    sum(topo[i+1] * topo[i] + topo[i+1] for i in 1:length(topo)-1)
end

function _descargar_cache(nombre, url)
    cache = ".cache_$(nombre).csv"
    if !isfile(cache)
        run(`curl -sL -o $cache $url`)
    end
    return cache
end

# Wilcoxon signed-rank test (two-sided, normal approximation)
function wilcoxon_test(x::Vector{Float64}, y::Vector{Float64})
    diffs = x .- y
    diffs = filter(!=(0.0), diffs)
    n = length(diffs)
    n == 0 && return 1.0  # no differences

    abs_diffs = abs.(diffs)
    ranks = sortperm(sortperm(abs_diffs))  # ranking
    # Proper ranking with ties
    sorted_abs = sort(abs_diffs)
    rank_map = zeros(n)
    i = 1
    while i <= n
        j = i
        while j <= n && sorted_abs[j] == sorted_abs[i]
            j += 1
        end
        avg_rank = (i + j - 1) / 2.0
        for k in i:j-1
            rank_map[k] = avg_rank
        end
        i = j
    end
    # Assign ranks to original positions
    order = sortperm(abs_diffs)
    final_ranks = zeros(n)
    for (pos, orig_idx) in enumerate(order)
        final_ranks[orig_idx] = rank_map[pos]
    end

    W_plus = sum(final_ranks[i] for i in 1:n if diffs[i] > 0; init=0.0)
    W_minus = sum(final_ranks[i] for i in 1:n if diffs[i] < 0; init=0.0)
    W = min(W_plus, W_minus)

    # Normal approximation
    mean_W = n * (n + 1) / 4.0
    std_W = sqrt(n * (n + 1) * (2n + 1) / 24.0)
    z = (W - mean_W) / std_W
    p = erfc(abs(z) / sqrt(2.0))  # two-sided
    return p
end

# ─── Datasets ───

function cargar_two_moons(; n=400, ruido=0.15)
    rng = MersenneTwister(BASE_SEED)
    n_half = n ÷ 2
    theta1 = range(0, π, length=n_half)
    x1 = cos.(theta1) .+ ruido .* randn(rng, n_half)
    y1 = sin.(theta1) .+ ruido .* randn(rng, n_half)
    theta2 = range(0, π, length=n - n_half)
    x2 = 1.0 .- cos.(theta2) .+ ruido .* randn(rng, n - n_half)
    y2 = 1.0 .- sin.(theta2) .- 0.5 .+ ruido .* randn(rng, n - n_half)
    X = Float64.(hcat(vcat(x1, x2), vcat(y1, y2))')
    labels = vcat(fill(0, n_half), fill(1, n - n_half))
    clases = [0, 1]
    Y = onehot(labels, clases)
    return X, Y, labels, "Two Moons"
end

function cargar_iris()
    dataset = Iris(as_df=false)
    X = Float64.(dataset.features)
    labels = vec(dataset.targets)
    normalizar_minmax!(X)
    clases = sort(unique(labels))
    Y = onehot(labels, clases)
    return X, Y, labels, "Iris"
end

function cargar_wine()
    cache = _descargar_cache("wine", "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data")
    data = readdlm(cache, ',', Float64)
    labels = Int.(data[:, 1])
    X = Float64.(data[:, 2:end]')
    normalizar_minmax!(X)
    clases = sort(unique(labels))
    Y = onehot(labels, clases)
    return X, Y, labels, "Wine"
end

function cargar_breastcancer()
    cache = _descargar_cache("breastcancer",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data")
    lines = readlines(cache)
    n = length(lines)
    X = zeros(Float64, 30, n)
    labels = Vector{String}(undef, n)
    for (i, line) in enumerate(lines)
        parts = split(line, ',')
        labels[i] = String(parts[2])
        for j in 1:30
            X[j, i] = parse(Float64, parts[j+2])
        end
    end
    normalizar_minmax!(X)
    clases = sort(unique(labels))
    Y = onehot(labels, clases)
    return X, Y, labels, "Breast Cancer"
end

function cargar_ionosphere()
    cache = _descargar_cache("ionosphere",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data")
    lines = readlines(cache)
    n = length(lines)
    X = zeros(Float64, 34, n)
    labels = Vector{String}(undef, n)
    for (i, line) in enumerate(lines)
        parts = split(line, ',')
        labels[i] = String(parts[end])
        for j in 1:34
            X[j, i] = parse(Float64, parts[j])
        end
    end
    normalizar_minmax!(X)
    clases = sort(unique(labels))
    Y = onehot(labels, clases)
    return X, Y, labels, "Ionosphere"
end

function cargar_sonar()
    cache = _descargar_cache("sonar",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")
    lines = readlines(cache)
    n = length(lines)
    X = zeros(Float64, 60, n)
    labels = Vector{String}(undef, n)
    for (i, line) in enumerate(lines)
        parts = split(line, ',')
        labels[i] = String(parts[end])
        for j in 1:60
            X[j, i] = parse(Float64, parts[j])
        end
    end
    normalizar_minmax!(X)
    clases = sort(unique(labels))
    Y = onehot(labels, clases)
    return X, Y, labels, "Sonar"
end

# ─── Configs ───

datasets_configs = [
    (cargar_two_moons,    [2, 32, 16, 2],      50, 0.6, 2, 500, 0.1),
    (cargar_iris,         [4, 16, 8, 3],        50, 0.4, 1, 200, 0.1),
    (cargar_wine,         [13, 24, 12, 3],      50, 0.5, 1, 300, 0.1),
    (cargar_breastcancer, [30, 32, 16, 2],      30, 0.7, 2, 200, 0.1),
    (cargar_ionosphere,   [34, 24, 12, 2],      30, 0.5, 1, 300, 0.1),
    (cargar_sonar,        [60, 32, 16, 2],      30, 0.5, 1, 300, 0.1),
]

# ─── Ejecución ───

# Helper stats
_mean(x) = sum(x) / length(x)
_std(x) = (m = _mean(x); sqrt(sum((xi - m)^2 for xi in x) / (length(x) - 1)))

println("=" ^ 100)
println("  SIGNIFICANCE TEST: Phase 1 vs Phase 1+2 ($N_SEEDS seeds, Wilcoxon signed-rank)")
println("=" ^ 100)
println()

all_results = []

for (cargar_fn, topo, nube_size, umbral, elim, epocas, lr) in datasets_configs
    X, Y, labels, nombre = cargar_fn()
    train_idx, test_idx = split_estratificado(labels)
    train_X, train_Y = X[:, train_idx], Y[:, train_idx]
    test_X, test_Y = X[:, test_idx], Y[:, test_idx]

    accs_f1 = Float64[]
    accs_f2 = Float64[]
    params_f1_list = Int[]
    params_f2_list = Int[]

    print("  $nombre: ")
    for s in 1:N_SEEDS
        seed = BASE_SEED + (s - 1) * 100

        # Phase 1
        config_f1 = ConfiguracionNube(
            tamano_nube=nube_size, topologia_inicial=topo,
            umbral_acierto=umbral, neuronas_eliminar=elim,
            epocas_refinamiento=epocas, tasa_aprendizaje=lr,
            semilla=seed, explorar_estructura=false
        )
        motor_f1 = MotorNube(config_f1, train_X, train_Y)
        inf_f1 = ejecutar(motor_f1)

        acc_f1 = 0.0
        p_f1 = 0
        if inf_f1.mejor_red !== nothing
            acts = RandomCloud.activaciones_por_capa(length(inf_f1.mejor_red.pesos), config_f1.activacion)
            acc_f1 = evaluar(inf_f1.mejor_red, test_X, test_Y; acts=acts)
            p_f1 = contar_parametros(inf_f1.topologia_final)
        end

        # Phase 1+2
        config_f2 = ConfiguracionNube(
            tamano_nube=nube_size, topologia_inicial=topo,
            umbral_acierto=umbral, neuronas_eliminar=elim,
            epocas_refinamiento=epocas, tasa_aprendizaje=lr,
            semilla=seed, explorar_estructura=true,
            max_profundidad_split=2, ancho_minimo_split=4,
            n_candidatos_estructura=5
        )
        motor_f2 = MotorNube(config_f2, train_X, train_Y)
        inf_f2 = ejecutar(motor_f2)

        acc_f2 = 0.0
        p_f2 = 0
        if inf_f2.mejor_red !== nothing
            acts = RandomCloud.activaciones_por_capa(length(inf_f2.mejor_red.pesos), config_f2.activacion)
            acc_f2 = evaluar(inf_f2.mejor_red, test_X, test_Y; acts=acts)
            p_f2 = contar_parametros(inf_f2.topologia_final)
        end

        push!(accs_f1, acc_f1)
        push!(accs_f2, acc_f2)
        push!(params_f1_list, p_f1)
        push!(params_f2_list, p_f2)
        print(".")
    end

    # Stats
    mean_f1 = round(_mean(accs_f1) * 100, digits=1)
    std_f1 = round(_std(accs_f1) * 100, digits=1)
    mean_f2 = round(_mean(accs_f2) * 100, digits=1)
    std_f2 = round(_std(accs_f2) * 100, digits=1)
    mean_params_f1 = round(_mean(params_f1_list), digits=0)
    mean_params_f2 = round(_mean(params_f2_list), digits=0)
    p_value = wilcoxon_test(accs_f2, accs_f1)
    sig = p_value < 0.05 ? "*" : ""

    println()
    println("    Phase 1:   $(mean_f1) ± $(std_f1)% | params≈$(Int(mean_params_f1))")
    println("    Phase 1+2: $(mean_f2) ± $(std_f2)% | params≈$(Int(mean_params_f2))")
    println("    Wilcoxon p=$(round(p_value, digits=4)) $sig")
    println()

    push!(all_results, (nombre=nombre, mean_f1=mean_f1, std_f1=std_f1,
                        mean_f2=mean_f2, std_f2=std_f2,
                        p_value=round(p_value, digits=4),
                        mean_params_f1=Int(mean_params_f1),
                        mean_params_f2=Int(mean_params_f2)))
end

# ─── Tabla final ───

println("=" ^ 100)
println("  TABLA PARA EL PAPER")
println("=" ^ 100)
println()
println("  Dataset          | Phase 1 (test)    | Phase 1+2 (test)  | p-value | Params F1 | Params F1+2")
println("  -----------------+-------------------+-------------------+---------+-----------+------------")
for r in all_results
    sig = r.p_value < 0.05 ? " *" : "  "
    println("  $(rpad(r.nombre, 17))| $(rpad("$(r.mean_f1)±$(r.std_f1)%", 18))| $(rpad("$(r.mean_f2)±$(r.std_f2)%", 18))| $(rpad("$(r.p_value)$sig", 8))| $(rpad(r.mean_params_f1, 10))| $(r.mean_params_f2)")
end
println()
println("  * = p < 0.05 (estadísticamente significativo)")
