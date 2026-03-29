# =============================================================================
# Análisis de coste computacional: Nube vs Magnitude vs Random vs Clásico
# =============================================================================
#
# Ejecutar con:
#   julia --project=. -t auto examples/coste_computacional.jl
#
# Mide tiempo total y desglose por fase para cada método.
# Cada método se ejecuta 3 veces y se reporta la mediana.
#
# Fases medidas:
#   Clásico:   train
#   Magnitude: train + prune + fine-tune
#   Random:    train + prune + fine-tune
#   Nube:      exploración + refinamiento
# =============================================================================

using RandomCloud
using RandomCloud: RedNeuronal, entrenar!, EntrenarBuffers, reconstruir
using RandomCloud: evaluar
using Random
using Statistics: median

function contar_parametros(topo::Vector{Int})
    sum(topo[i+1] * topo[i] + topo[i+1] for i in 1:length(topo)-1)
end

function split_estratificado(labels, clases, semilla; ratio=0.8)
    rng = MersenneTwister(semilla)
    tr = Int[]; te = Int[]
    for c in clases
        idx = findall(==(c), labels); perm = shuffle(rng, idx)
        nt = round(Int, ratio * length(perm))
        append!(tr, perm[1:nt]); append!(te, perm[nt+1:end])
    end
    shuffle!(rng, tr); shuffle!(rng, te)
    return tr, te
end

function entrenar_red!(red, X, Y, epocas, lr)
    bufs = EntrenarBuffers(red.topologia)
    n = size(X, 2)
    for _ in 1:epocas
        @inbounds for k in 1:n
            entrenar!(red, @view(X[:, k]), @view(Y[:, k]), lr, bufs)
        end
    end
end

function magnitude_prune(red::RedNeuronal, topo_obj::Vector{Int})
    nc = length(red.topologia)
    np = [copy(w) for w in red.pesos]; nb = [copy(b) for b in red.biases]
    ta = copy(red.topologia)
    for capa in 2:(nc-1)
        na = ta[capa]; no = topo_obj[capa]; no >= na && continue
        ic = capa - 1
        normas = [sum(np[ic][j,:].^2) for j in 1:na]
        mant = sort(sortperm(normas, rev=true)[1:no])
        np[ic] = np[ic][mant, :]; nb[ic] = nb[ic][mant]
        if ic < length(np); np[ic+1] = np[ic+1][:, mant]; end
        ta[capa] = no
    end
    return RedNeuronal(ta, np, nb)
end

function random_prune(red::RedNeuronal, topo_obj::Vector{Int}, rng)
    nc = length(red.topologia)
    np = [copy(w) for w in red.pesos]; nb = [copy(b) for b in red.biases]
    ta = copy(red.topologia)
    for capa in 2:(nc-1)
        na = ta[capa]; no = topo_obj[capa]; no >= na && continue
        ic = capa - 1
        mant = sort(shuffle(rng, collect(1:na))[1:no])
        np[ic] = np[ic][mant, :]; nb[ic] = nb[ic][mant]
        if ic < length(np); np[ic+1] = np[ic+1][:, mant]; end
        ta[capa] = no
    end
    return RedNeuronal(ta, np, nb)
end

# Contar FLOPs de un feedforward (multiplicaciones + sumas)
function flops_feedforward(topo::Vector{Int})
    total = 0
    for i in 1:(length(topo)-1)
        total += 2 * topo[i] * topo[i+1]  # mul + add por elemento
        total += topo[i+1]                  # bias add
        total += topo[i+1] * 4             # sigmoid ≈ 4 ops (exp, add, div, sub)
    end
    return total
end

# Contar FLOPs de un paso de backprop (forward + backward + update)
function flops_backprop(topo::Vector{Int})
    fwd = flops_feedforward(topo)
    # Backward ≈ 2× forward (gradientes + outer products)
    # Update ≈ total_params × 2 (mul lr + sub)
    params = contar_parametros(topo)
    return fwd + 2 * fwd + 2 * params
end

function medir_tiempos(nombre, trX, trY, teX, teY, topo, epocas, lr,
                       umbral, neuronas_elim; semilla=42, runs=3)
    params_i = contar_parametros(topo)
    n_muestras = size(trX, 2)

    # Primero: obtener topología objetivo con la nube
    config0 = ConfiguracionNube(tamano_nube=50, topologia_inicial=topo,
        umbral_acierto=umbral, neuronas_eliminar=neuronas_elim,
        epocas_refinamiento=epocas, tasa_aprendizaje=lr, semilla=semilla)
    motor0 = MotorNube(config0, trX, trY)
    inf0 = ejecutar(motor0)
    if !inf0.exitoso
        println("  $nombre: Nube FAIL, saltando."); return
    end
    topo_obj = inf0.topologia_final
    params_obj = contar_parametros(topo_obj)
    red_pct = round((1.0 - params_obj / params_i) * 100, digits=1)

    println("─" ^ 100)
    println("  $nombre — $topo → $topo_obj (-$(red_pct)%)")
    println("  Train: $n_muestras muestras | Épocas: $epocas | LR: $lr | Runs: $runs")
    println("─" ^ 100)

    # Medir tiempos (runs repeticiones, tomar mediana)
    t_clasico = Float64[]
    t_mag_train = Float64[]; t_mag_prune = Float64[]; t_mag_ft = Float64[]
    t_rnd_train = Float64[]; t_rnd_prune = Float64[]; t_rnd_ft = Float64[]
    t_nube_explor = Float64[]; t_nube_refine = Float64[]

    for r in 1:runs
        # Clásico
        t0 = time_ns()
        rc = RedNeuronal(topo, MersenneTwister(semilla))
        entrenar_red!(rc, trX, trY, epocas, lr)
        push!(t_clasico, (time_ns() - t0) / 1e6)

        # Magnitude: train → prune → fine-tune
        t0 = time_ns()
        rm = RedNeuronal(topo, MersenneTwister(semilla))
        entrenar_red!(rm, trX, trY, epocas, lr)
        t1 = time_ns()
        push!(t_mag_train, (t1 - t0) / 1e6)

        t0 = time_ns()
        rm = magnitude_prune(rm, topo_obj)
        t1 = time_ns()
        push!(t_mag_prune, (t1 - t0) / 1e6)

        t0 = time_ns()
        entrenar_red!(rm, trX, trY, epocas, lr)
        t1 = time_ns()
        push!(t_mag_ft, (t1 - t0) / 1e6)

        # Random: train → prune → fine-tune
        t0 = time_ns()
        rr = RedNeuronal(topo, MersenneTwister(semilla))
        entrenar_red!(rr, trX, trY, epocas, lr)
        t1 = time_ns()
        push!(t_rnd_train, (t1 - t0) / 1e6)

        t0 = time_ns()
        rr = random_prune(rr, topo_obj, MersenneTwister(semilla + 99999))
        t1 = time_ns()
        push!(t_rnd_prune, (t1 - t0) / 1e6)

        t0 = time_ns()
        entrenar_red!(rr, trX, trY, epocas, lr)
        t1 = time_ns()
        push!(t_rnd_ft, (t1 - t0) / 1e6)

        # Nube: exploración + refinamiento (medido internamente)
        config = ConfiguracionNube(tamano_nube=50, topologia_inicial=topo,
            umbral_acierto=umbral, neuronas_eliminar=neuronas_elim,
            epocas_refinamiento=epocas, tasa_aprendizaje=lr, semilla=semilla)
        motor = MotorNube(config, trX, trY)

        t0 = time_ns()
        inf = ejecutar(motor)
        t_total_nube = (time_ns() - t0) / 1e6

        # Estimar desglose: refinamiento ≈ tiempo de entrenar red reducida
        t0r = time_ns()
        red_tmp = RedNeuronal(topo_obj, MersenneTwister(semilla))
        entrenar_red!(red_tmp, trX, trY, epocas, lr)
        t_ref_est = (time_ns() - t0r) / 1e6

        push!(t_nube_refine, t_ref_est)
        push!(t_nube_explor, max(0.0, t_total_nube - t_ref_est))
    end

    # Medianas
    mc = median(t_clasico)
    mm_tr = median(t_mag_train); mm_pr = median(t_mag_prune); mm_ft = median(t_mag_ft)
    mr_tr = median(t_rnd_train); mr_pr = median(t_rnd_prune); mr_ft = median(t_rnd_ft)
    mn_ex = median(t_nube_explor); mn_rf = median(t_nube_refine)

    mm_total = mm_tr + mm_pr + mm_ft
    mr_total = mr_tr + mr_pr + mr_ft
    mn_total = mn_ex + mn_rf

    # FLOPs teóricos
    flops_fwd_full = flops_feedforward(topo)
    flops_bp_full = flops_backprop(topo)
    flops_bp_red = flops_backprop(topo_obj)

    flops_clasico = epocas * n_muestras * flops_bp_full
    flops_mag = epocas * n_muestras * flops_bp_full + epocas * n_muestras * flops_bp_red  # train + ft
    flops_rnd = flops_mag  # mismo coste
    # Nube: exploración (50 redes × ~8 reducciones × n_muestras × fwd) + refinamiento
    n_evals_est = 50 * 8  # estimación conservadora
    flops_nube = n_evals_est * n_muestras * flops_fwd_full + epocas * n_muestras * flops_bp_red

    fmt_ms(v) = v < 1000 ? "$(round(v, digits=0))ms" : "$(round(v/1000, digits=1))s"
    fmt_flops(v) = v < 1e6 ? "$(round(v/1e3, digits=0))K" :
                   v < 1e9 ? "$(round(v/1e6, digits=1))M" :
                             "$(round(v/1e9, digits=2))G"

    println()
    println("  Método             │ Total      │ Desglose                              │ FLOPs teóricos")
    println("  ───────────────────┼────────────┼───────────────────────────────────────┼────────────────")
    println("  Clásico            │ $(rpad(fmt_ms(mc), 10)) │ train=$(fmt_ms(mc))                            │ $(fmt_flops(flops_clasico))")
    println("  Magnitude Pruning  │ $(rpad(fmt_ms(mm_total), 10)) │ train=$(fmt_ms(mm_tr)) + prune=$(fmt_ms(mm_pr)) + ft=$(fmt_ms(mm_ft)) │ $(fmt_flops(flops_mag))")
    println("  Random Pruning     │ $(rpad(fmt_ms(mr_total), 10)) │ train=$(fmt_ms(mr_tr)) + prune=$(fmt_ms(mr_pr)) + ft=$(fmt_ms(mr_ft)) │ $(fmt_flops(flops_rnd))")
    println("  Nube Aleatoria     │ $(rpad(fmt_ms(mn_total), 10)) │ explor=$(fmt_ms(mn_ex)) + refine=$(fmt_ms(mn_rf))          │ $(fmt_flops(flops_nube))")
    println()
    println("  Ratios vs Clásico: Mag=$(round(mm_total/mc, digits=2))× | Rnd=$(round(mr_total/mc, digits=2))× | Nube=$(round(mn_total/mc, digits=2))×")
    println("  Ratios FLOPs:      Mag=$(round(flops_mag/flops_clasico, digits=2))× | Rnd=$(round(flops_rnd/flops_clasico, digits=2))× | Nube=$(round(flops_nube/flops_clasico, digits=2))×")
    println()
end

println("=" ^ 100)
println("  ANÁLISIS DE COSTE COMPUTACIONAL (mediana de 3 runs)")
println("  Threads: $(Threads.nthreads())")
println("=" ^ 100)
println()

# ─── Cargar datasets ───

# Sonar
sonar_path = joinpath(@__DIR__, "..", ".cache_sonar.csv")
lines = filter(l->!isempty(strip(l)), readlines(sonar_path))
n = length(lines)
sonar_X = zeros(Float64, 60, n); sonar_lab = zeros(Int, n)
for (k, line) in enumerate(lines)
    parts = split(line, ',')
    for j in 1:60; sonar_X[j,k]=parse(Float64,parts[j]); end
    sonar_lab[k] = parts[61]=="M" ? 1 : 0
end
for f in 1:60; mn,mx=extrema(@view sonar_X[f,:]); mx>mn && (sonar_X[f,:].=(sonar_X[f,:].-mn)./(mx-mn)); end
sonar_Y = zeros(Float64, 2, n); for k in 1:n; sonar_Y[sonar_lab[k]+1,k]=1.0; end
str, ste = split_estratificado(sonar_lab, [0,1], 42)

medir_tiempos("Sonar (60d, 167 train)", sonar_X[:,str], sonar_Y[:,str], sonar_X[:,ste], sonar_Y[:,ste],
    [60,8,2], 200, 0.1, 0.55, 1)

# Ionosphere
iono_path = joinpath(@__DIR__, "..", ".cache_ionosphere.csv")
lines = filter(l->!isempty(strip(l)), readlines(iono_path))
n = length(lines)
iono_X = zeros(Float64, 34, n); iono_lab = zeros(Int, n)
for (k, line) in enumerate(lines)
    parts = split(line, ',')
    for j in 1:34; iono_X[j,k]=parse(Float64,parts[j]); end
    iono_lab[k] = parts[35]=="g" ? 1 : 0
end
for f in 1:34; mn,mx=extrema(@view iono_X[f,:]); mx>mn && (iono_X[f,:].=(iono_X[f,:].-mn)./(mx-mn)); end
iono_Y = zeros(Float64, 2, n); for k in 1:n; iono_Y[iono_lab[k]+1,k]=1.0; end
itr, ite = split_estratificado(iono_lab, [0,1], 42)

medir_tiempos("Ionosphere (34d, 281 train)", iono_X[:,itr], iono_Y[:,itr], iono_X[:,ite], iono_Y[:,ite],
    [34,16,2], 200, 0.1, 0.6, 1)

# Breast Cancer
bc_path = joinpath(@__DIR__, "..", ".cache_breastcancer.csv")
lines = readlines(bc_path); n = length(lines)
bc_X = zeros(Float64, 30, n); bc_lab = zeros(Int, n)
for (k, line) in enumerate(lines)
    parts = split(line, ',')
    bc_lab[k] = parts[2]=="M" ? 1 : 0
    for j in 1:30; bc_X[j,k]=parse(Float64,parts[j+2]); end
end
for f in 1:30; mn,mx=extrema(@view bc_X[f,:]); mx>mn && (bc_X[f,:].=(bc_X[f,:].-mn)./(mx-mn)); end
bc_Y = zeros(Float64, 2, n); for k in 1:n; bc_Y[bc_lab[k]+1,k]=1.0; end
btr, bte = split_estratificado(bc_lab, [0,1], 42)

medir_tiempos("Breast Cancer (30d, 456 train)", bc_X[:,btr], bc_Y[:,btr], bc_X[:,bte], bc_Y[:,bte],
    [30,8,2], 100, 0.1, 0.7, 1)

# Optical Digits
function parsear_digits(fp)
    lines=filter(l->!isempty(strip(l)),readlines(fp)); n=length(lines)
    F=zeros(Float64,64,n); L=zeros(Int,n)
    for (k,line) in enumerate(lines); parts=split(line,',')
        for j in 1:64; F[j,k]=parse(Float64,parts[j]); end; L[k]=parse(Int,parts[65]); end
    return F,L
end
dtr=joinpath(@__DIR__,"../.cache_digits_train.csv"); dte=joinpath(@__DIR__,"../.cache_digits_test.csv")
if isfile(dtr) && isfile(dte)
    dtX,dtL=parsear_digits(dtr); deX,deL=parsear_digits(dte)
    dtX./=16.0; deX./=16.0
    dtY=zeros(Float64,10,size(dtX,2)); deY=zeros(Float64,10,size(deX,2))
    for k in 1:size(dtX,2); dtY[dtL[k]+1,k]=1.0; end
    for k in 1:size(deX,2); deY[deL[k]+1,k]=1.0; end

    medir_tiempos("Opt. Digits (64d, 3823 train)", dtX, dtY, deX, deY,
        [64,32,10], 100, 0.1, 0.15, 2)
end

# Adult Income
const CAT_COLS = [2,4,6,7,8,9,10,14]; const NUM_COLS = [1,3,5,11,12,13]
function parsear_adult(fp, skip=false)
    lines=readlines(fp); skip && (lines=lines[2:end])
    lines=filter(l->!isempty(strip(l))&&!occursin("?",l), lines)
    n=length(lines); cv=Dict{Int,Set{String}}(c=>Set{String}() for c in CAT_COLS)
    parsed=Vector{Vector{String}}(undef,n); labels=zeros(Int,n)
    for (i,line) in enumerate(lines)
        parts=[strip(p) for p in split(line,',')]; length(parts)<15 && continue
        parsed[i]=parts[1:14]; labels[i]=occursin(">50K",parts[15]) ? 1 : 0
        for c in CAT_COLS; push!(cv[c],parts[c]); end
    end; return parsed,labels,cv
end
function codificar_adult(parsed,labels,cv)
    n=length(parsed); nn=length(NUM_COLS); nc=sum(length(cv[c]) for c in CAT_COLS); nf=nn+nc
    F=zeros(Float64,nf,n); O=zeros(Float64,2,n)
    for i in 1:n
        parts=parsed[i]; col=1
        for c in NUM_COLS; F[col,i]=parse(Float64,parts[c]); col+=1; end
        for c in CAT_COLS; vals=cv[c]; idx=findfirst(==(parts[c]),vals)
            idx!==nothing && (F[col+idx-1,i]=1.0); col+=length(vals); end
        O[labels[i]+1,i]=1.0
    end
    for j in 1:nn; mn,mx=extrema(@view F[j,:]); mx>mn && (F[j,:].=(F[j,:].-mn)./(mx-mn)); end
    return F,O
end

atr=joinpath(@__DIR__,"../.cache_adult_train.csv"); ate=joinpath(@__DIR__,"../.cache_adult_test.csv")
if isfile(atr) && isfile(ate)
    p1,l1,c1=parsear_adult(atr,false); p2,l2,c2=parsear_adult(ate,true)
    cv=Dict{Int,Vector{String}}(); for c in CAT_COLS; cv[c]=sort(collect(union(c1[c],c2[c]))); end
    atrX,atrY=codificar_adult(p1,l1,cv); ateX,ateY=codificar_adult(p2,l2,cv)
    nf=size(atrX,1)

    medir_tiempos("Adult Income (104d, 30162 train)", atrX, atrY, ateX, ateY,
        [nf,16,2], 30, 0.1, 0.6, 2)
end

println("=" ^ 100)
println("  FIN")
println("=" ^ 100)
