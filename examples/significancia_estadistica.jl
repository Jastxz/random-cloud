# =============================================================================
# Significancia estadística: 10 semillas × 4 métodos × 7 datasets
# =============================================================================
#
# Ejecutar con:
#   julia --project=. -t auto examples/significancia_estadistica.jl
#
# Cada método se ejecuta con 10 semillas distintas. Se reporta media ± std
# de Accuracy y F1 en test. Se aplica Wilcoxon signed-rank test (aproximado)
# para comparar Nube vs cada baseline.
# =============================================================================

using RandomCloud
using RandomCloud: RedNeuronal, entrenar!, EntrenarBuffers, reconstruir
using RandomCloud: evaluar, evaluar_f1, evaluar_auc
using Random
using Statistics: mean, std
using SpecialFunctions: erfc
using Downloads: download

const SEMILLAS = [42, 123, 256, 314, 501, 667, 789, 888, 1024, 1337]
const N_SEEDS = length(SEMILLAS)

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

function magnitude_prune(red::RedNeuronal, topo_obj::Vector{Int})
    n_capas = length(red.topologia)
    np = [copy(w) for w in red.pesos]; nb = [copy(b) for b in red.biases]
    ta = copy(red.topologia)
    for capa in 2:(n_capas-1)
        na = ta[capa]; no = topo_obj[capa]
        no >= na && continue
        ic = capa - 1
        normas = [sum(np[ic][j,:].^2) for j in 1:na]
        mant = sort(sortperm(normas, rev=true)[1:no])
        np[ic] = np[ic][mant, :]; nb[ic] = nb[ic][mant]
        if ic < length(np); np[ic+1] = np[ic+1][:, mant]; end
        ta[capa] = no
    end
    return RedNeuronal(ta, np, nb)
end

function random_prune(red::RedNeuronal, topo_obj::Vector{Int}, rng::AbstractRNG)
    n_capas = length(red.topologia)
    np = [copy(w) for w in red.pesos]; nb = [copy(b) for b in red.biases]
    ta = copy(red.topologia)
    for capa in 2:(n_capas-1)
        na = ta[capa]; no = topo_obj[capa]
        no >= na && continue
        ic = capa - 1
        mant = sort(shuffle(rng, collect(1:na))[1:no])
        np[ic] = np[ic][mant, :]; nb[ic] = nb[ic][mant]
        if ic < length(np); np[ic+1] = np[ic+1][:, mant]; end
        ta[capa] = no
    end
    return RedNeuronal(ta, np, nb)
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

# Wilcoxon signed-rank test aproximado (normal approx para n≥10)
function wilcoxon_p(x::Vector{Float64}, y::Vector{Float64})
    d = x .- y
    d = d[d .!= 0.0]
    n = length(d)
    n == 0 && return 1.0
    ranks = sortperm(sortperm(abs.(d)))  # rank de |d|
    W_plus = sum(ranks[i] for i in 1:n if d[i] > 0; init=0.0)
    mu = n * (n + 1) / 4.0
    sigma = sqrt(n * (n + 1) * (2n + 1) / 24.0)
    sigma == 0.0 && return 1.0
    z = (W_plus - mu) / sigma
    # Aproximación normal bilateral: p ≈ 2 * Φ(-|z|)
    # Φ(-|z|) ≈ erfc(|z|/√2)/2
    p = 2.0 * 0.5 * erfc(abs(z) / sqrt(2.0))
    return p
end

function fmt_pm(vals)
    m = mean(vals); s = std(vals)
    "$(round(m*100, digits=1))±$(round(s*100, digits=1))%"
end

function fmt_pm_f1(vals)
    m = mean(vals); s = std(vals)
    "$(round(m, digits=3))±$(round(s, digits=3))"
end

# ─── Función principal: correr 4 métodos × 10 semillas ───

function correr_significancia(nombre, X, Y, labels, clases, topo, epocas, lr,
                              umbral, neuronas_elim; split_semilla=42)
    params_i = contar_parametros(topo)

    # Split fijo (misma partición para todos los métodos y semillas)
    tr, te = split_estratificado(labels, clases, split_semilla)
    trX, trY = X[:, tr], Y[:, tr]
    teX, teY = X[:, te], Y[:, te]

    # Primero: obtener topología objetivo con la nube (semilla=42)
    config0 = ConfiguracionNube(tamano_nube=50, topologia_inicial=topo,
        umbral_acierto=umbral, neuronas_eliminar=neuronas_elim,
        epocas_refinamiento=epocas, tasa_aprendizaje=lr, semilla=42)
    motor0 = MotorNube(config0, trX, trY)
    inf0 = ejecutar(motor0)
    if !inf0.exitoso
        println("  $nombre: Nube FAIL con semilla 42, saltando.")
        return
    end
    topo_obj = inf0.topologia_final
    params_obj = contar_parametros(topo_obj)
    red_pct = round((1.0 - params_obj / params_i) * 100, digits=1)

    println("─" ^ 95)
    println("  $nombre — $topo → $topo_obj (-$(red_pct)%)")
    println("  Train: $(length(tr)) | Test: $(length(te)) | Épocas: $epocas | LR: $lr | Seeds: $N_SEEDS")
    println("─" ^ 95)

    acc_clasico = Float64[]; f1_clasico = Float64[]
    acc_mag = Float64[]; f1_mag = Float64[]
    acc_rnd = Float64[]; f1_rnd = Float64[]
    acc_nube = Float64[]; f1_nube = Float64[]

    for (i, s) in enumerate(SEMILLAS)
        print("  Seed $s ($(i)/$(N_SEEDS))... ")

        # Clásico
        rc = RedNeuronal(topo, MersenneTwister(s))
        entrenar_red!(rc, trX, trY, epocas, lr)
        ac = evaluar(rc, teX, teY); fc = evaluar_f1(rc, teX, teY)
        push!(acc_clasico, ac); push!(f1_clasico, fc)

        # Magnitude: train → prune → fine-tune
        rm = RedNeuronal(topo, MersenneTwister(s))
        entrenar_red!(rm, trX, trY, epocas, lr)
        rm = magnitude_prune(rm, topo_obj)
        entrenar_red!(rm, trX, trY, epocas, lr)
        am = evaluar(rm, teX, teY); fm = evaluar_f1(rm, teX, teY)
        push!(acc_mag, am); push!(f1_mag, fm)

        # Random: train → prune → fine-tune
        rr = RedNeuronal(topo, MersenneTwister(s))
        entrenar_red!(rr, trX, trY, epocas, lr)
        rr = random_prune(rr, topo_obj, MersenneTwister(s + 99999))
        entrenar_red!(rr, trX, trY, epocas, lr)
        ar = evaluar(rr, teX, teY); fr = evaluar_f1(rr, teX, teY)
        push!(acc_rnd, ar); push!(f1_rnd, fr)

        # Nube
        config = ConfiguracionNube(tamano_nube=50, topologia_inicial=topo,
            umbral_acierto=umbral, neuronas_eliminar=neuronas_elim,
            epocas_refinamiento=epocas, tasa_aprendizaje=lr, semilla=s)
        motor = MotorNube(config, trX, trY)
        inf = ejecutar(motor)
        if inf.exitoso
            an = evaluar(inf.mejor_red, teX, teY); fn = evaluar_f1(inf.mejor_red, teX, teY)
        else
            an = 0.0; fn = 0.0
        end
        push!(acc_nube, an); push!(f1_nube, fn)

        println("C=$(round(ac*100,digits=1))% M=$(round(am*100,digits=1))% R=$(round(ar*100,digits=1))% N=$(round(an*100,digits=1))%")
    end

    # Filtrar runs donde la nube falló
    valid = acc_nube .> 0.0
    n_valid = sum(valid)

    println()
    println("  Resultados ($n_valid/$N_SEEDS runs válidos de la nube):")
    println()
    println("  Método             │ Acc (mean±std)     │ F1 (mean±std)")
    println("  ───────────────────┼────────────────────┼──────────────────")
    println("  Clásico            │ $(rpad(fmt_pm(acc_clasico), 18)) │ $(fmt_pm_f1(f1_clasico))")
    println("  Magnitude Pruning  │ $(rpad(fmt_pm(acc_mag), 18)) │ $(fmt_pm_f1(f1_mag))")
    println("  Random Pruning     │ $(rpad(fmt_pm(acc_rnd), 18)) │ $(fmt_pm_f1(f1_rnd))")
    println("  Nube Aleatoria     │ $(rpad(fmt_pm(acc_nube[valid]), 18)) │ $(fmt_pm_f1(f1_nube[valid]))")

    # Wilcoxon tests (solo sobre runs válidos)
    if n_valid >= 5
        acc_n_v = acc_nube[valid]; acc_m_v = acc_mag[valid]; acc_r_v = acc_rnd[valid]
        p_nm_acc = wilcoxon_p(acc_n_v, acc_m_v)
        p_nr_acc = wilcoxon_p(acc_n_v, acc_r_v)
        p_nm_f1 = wilcoxon_p(f1_nube[valid], f1_mag[valid])
        p_nr_f1 = wilcoxon_p(f1_nube[valid], f1_rnd[valid])

        println()
        println("  Wilcoxon signed-rank (Nube vs baseline):")
        sig(p) = p < 0.05 ? "✓ (p<0.05)" : "✗ (p=$(round(p, digits=3)))"
        println("    Nube vs Magnitude:  Acc p=$(round(p_nm_acc, digits=4)) $(sig(p_nm_acc))  │  F1 p=$(round(p_nm_f1, digits=4)) $(sig(p_nm_f1))")
        println("    Nube vs Random:     Acc p=$(round(p_nr_acc, digits=4)) $(sig(p_nr_acc))  │  F1 p=$(round(p_nr_f1, digits=4)) $(sig(p_nr_f1))")
    end
    println()
end

# ═══════════════════════════════════════════════════════════════════════════════

println("=" ^ 95)
println("  SIGNIFICANCIA ESTADÍSTICA: 10 semillas × 4 métodos")
println("=" ^ 95)
println()

# 1. Breast Cancer
bc_path = joinpath(@__DIR__, "..", ".cache_breastcancer.csv")
if isfile(bc_path)
    lines = readlines(bc_path); n = length(lines)
    X = zeros(Float64, 30, n); lab = zeros(Int, n)
    for (k, line) in enumerate(lines)
        parts = split(line, ',')
        lab[k] = parts[2] == "M" ? 1 : 0
        for j in 1:30; X[j, k] = parse(Float64, parts[j+2]); end
    end
    for f in 1:30; mn,mx=extrema(@view X[f,:]); mx>mn && (X[f,:].=(X[f,:].-mn)./(mx-mn)); end
    Y = zeros(Float64, 2, n); for k in 1:n; Y[lab[k]+1,k]=1.0; end
    correr_significancia("Breast Cancer", X, Y, lab, [0,1], [30,8,2], 100, 0.1, 0.7, 1)
end

# 2. Sonar
sonar_path = joinpath(@__DIR__, "..", ".cache_sonar.csv")
if isfile(sonar_path)
    lines = filter(l->!isempty(strip(l)), readlines(sonar_path)); n = length(lines)
    X = zeros(Float64, 60, n); lab = zeros(Int, n)
    for (k, line) in enumerate(lines)
        parts = split(line, ',')
        for j in 1:60; X[j,k]=parse(Float64,parts[j]); end
        lab[k] = parts[61]=="M" ? 1 : 0
    end
    for f in 1:60; mn,mx=extrema(@view X[f,:]); mx>mn && (X[f,:].=(X[f,:].-mn)./(mx-mn)); end
    Y = zeros(Float64, 2, n); for k in 1:n; Y[lab[k]+1,k]=1.0; end
    correr_significancia("Sonar", X, Y, lab, [0,1], [60,8,2], 200, 0.1, 0.55, 1)
end

# 3. Ionosphere
iono_path = joinpath(@__DIR__, "..", ".cache_ionosphere.csv")
if isfile(iono_path)
    lines = filter(l->!isempty(strip(l)), readlines(iono_path)); n = length(lines)
    X = zeros(Float64, 34, n); lab = zeros(Int, n)
    for (k, line) in enumerate(lines)
        parts = split(line, ',')
        for j in 1:34; X[j,k]=parse(Float64,parts[j]); end
        lab[k] = parts[35]=="g" ? 1 : 0
    end
    for f in 1:34; mn,mx=extrema(@view X[f,:]); mx>mn && (X[f,:].=(X[f,:].-mn)./(mx-mn)); end
    Y = zeros(Float64, 2, n); for k in 1:n; Y[lab[k]+1,k]=1.0; end
    correr_significancia("Ionosphere", X, Y, lab, [0,1], [34,16,2], 200, 0.1, 0.6, 1)
end

# 4. Adult Income
const CAT_COLS = [2,4,6,7,8,9,10,14]; const NUM_COLS = [1,3,5,11,12,13]
function parsear_adult(fp, skip=false)
    lines = readlines(fp); skip && (lines=lines[2:end])
    lines = filter(l->!isempty(strip(l))&&!occursin("?",l), lines)
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
    # Para Adult usamos train/test predefinido, no split aleatorio
    trX,trY=codificar_adult(p1,l1,cv); teX,teY=codificar_adult(p2,l2,cv)
    nf=size(trX,1)
    # Concatenar para usar split_estratificado con semilla fija
    allX=hcat(trX,teX); allY=hcat(trY,teY); allL=vcat(l1,l2)
    correr_significancia("Adult Income", allX, allY, allL, [0,1], [nf,16,2], 30, 0.1, 0.6, 2)
end

# 5. Wine
using MLDatasets: Wine; import DataFrames
ds=Wine(as_df=false); X=Float64.(ds.features); lab_s=vec(ds.targets)
for f in 1:size(X,1); mn,mx=extrema(@view X[f,:]); mx>mn && (X[f,:].=(X[f,:].-mn)./(mx-mn)); end
clases=sort(unique(lab_s)); ci=Dict(c=>i for (i,c) in enumerate(clases))
n=size(X,2); Y=zeros(Float64,3,n); lab_i=[ci[lab_s[k]] for k in 1:n]
for k in 1:n; Y[lab_i[k],k]=1.0; end
correr_significancia("Wine", X, Y, lab_i, sort(unique(lab_i)), [13,16,3], 100, 0.1, 0.4, 1)

# 6. Optical Digits
function parsear_digits(fp)
    lines=filter(l->!isempty(strip(l)),readlines(fp)); n=length(lines)
    F=zeros(Float64,64,n); L=zeros(Int,n)
    for (k,line) in enumerate(lines); parts=split(line,',')
        for j in 1:64; F[j,k]=parse(Float64,parts[j]); end; L[k]=parse(Int,parts[65]); end
    return F,L
end
dtr=joinpath(@__DIR__,"../.cache_digits_train.csv"); dte=joinpath(@__DIR__,"../.cache_digits_test.csv")
if isfile(dtr) && isfile(dte)
    tX,tL=parsear_digits(dtr); eX,eL=parsear_digits(dte)
    allX=hcat(tX,eX)./16.0; allL=vcat(tL,eL)
    n=size(allX,2); allY=zeros(Float64,10,n)
    for k in 1:n; allY[allL[k]+1,k]=1.0; end
    correr_significancia("Optical Digits", allX, allY, allL, 0:9, [64,32,10], 100, 0.1, 0.15, 2)
end

println("=" ^ 95)
println("  FIN")
println("=" ^ 95)
