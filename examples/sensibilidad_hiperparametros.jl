# =============================================================================
# Análisis de sensibilidad de hiperparámetros
# =============================================================================
#
# Ejecutar con:
#   julia --project=. -t auto examples/sensibilidad_hiperparametros.jl
#
# Varía un hiperparámetro a la vez (los demás fijos) y mide Accuracy test,
# topología final y reducción de parámetros. 3 seeds por configuración.
#
# Hiperparámetros analizados:
#   1. Tamaño de nube: 10, 25, 50, 100, 200
#   2. Umbral de acierto: 0.3, 0.4, 0.5, 0.6, 0.7
#   3. Neuronas a eliminar: 1, 2, 4
#
# Datasets: Sonar, Ionosphere, Wine (varianza suficiente)
# =============================================================================

using RandomCloud
using RandomCloud: RedNeuronal, entrenar!, EntrenarBuffers, evaluar, evaluar_f1
using Random
using Statistics: mean, std

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

const SEEDS = [42, 123, 789]

# Corre la nube con una configuración y devuelve (acc_mean, f1_mean, topo, reduccion, n_exitos)
function evaluar_config(trX, trY, teX, teY, topo, epocas, lr;
                        tamano_nube=50, umbral=0.5, neuronas_elim=1)
    params_i = contar_parametros(topo)
    accs = Float64[]; f1s = Float64[]; topos = Vector{Int}[]

    for s in SEEDS
        config = ConfiguracionNube(
            tamano_nube=tamano_nube, topologia_inicial=topo,
            umbral_acierto=umbral, neuronas_eliminar=neuronas_elim,
            epocas_refinamiento=epocas, tasa_aprendizaje=lr, semilla=s)
        motor = MotorNube(config, trX, trY)
        inf = ejecutar(motor)
        if inf.exitoso
            push!(accs, evaluar(inf.mejor_red, teX, teY))
            push!(f1s, evaluar_f1(inf.mejor_red, teX, teY))
            push!(topos, inf.topologia_final)
        end
    end

    n_ok = length(accs)
    if n_ok == 0
        return 0.0, 0.0, Int[], 0.0, 0
    end

    acc_m = mean(accs)
    f1_m = mean(f1s)
    # Topología más frecuente
    topo_f = topos[1]
    params_f = contar_parametros(topo_f)
    red = round((1.0 - params_f / params_i) * 100, digits=1)
    return acc_m, f1_m, topo_f, red, n_ok
end

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
sonar_trX, sonar_trY = sonar_X[:,str], sonar_Y[:,str]
sonar_teX, sonar_teY = sonar_X[:,ste], sonar_Y[:,ste]

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
iono_trX, iono_trY = iono_X[:,itr], iono_Y[:,itr]
iono_teX, iono_teY = iono_X[:,ite], iono_Y[:,ite]

# Wine
using MLDatasets: Wine; import DataFrames
ds=Wine(as_df=false); wine_X=Float64.(ds.features); wlab_s=vec(ds.targets)
for f in 1:size(wine_X,1); mn,mx=extrema(@view wine_X[f,:]); mx>mn && (wine_X[f,:].=(wine_X[f,:].-mn)./(mx-mn)); end
clases=sort(unique(wlab_s)); ci=Dict(c=>i for (i,c) in enumerate(clases))
nw=size(wine_X,2); wine_Y=zeros(Float64,3,nw); wlab_i=[ci[wlab_s[k]] for k in 1:nw]
for k in 1:nw; wine_Y[wlab_i[k],k]=1.0; end
rng_w=MersenneTwister(42); wtr=Int[]; wte=Int[]
for c in clases; idx=findall(==(c),wlab_s); perm=shuffle(rng_w,idx); nt=round(Int,0.8*length(perm))
    append!(wtr,perm[1:nt]); append!(wte,perm[nt+1:end]); end
shuffle!(rng_w,wtr); shuffle!(rng_w,wte)
wine_trX, wine_trY = wine_X[:,wtr], wine_Y[:,wtr]
wine_teX, wine_teY = wine_X[:,wte], wine_Y[:,wte]

println("=" ^ 95)
println("  ANÁLISIS DE SENSIBILIDAD DE HIPERPARÁMETROS (3 seeds por config)")
println("=" ^ 95)
println()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. TAMAÑO DE NUBE (10, 25, 50, 100, 200)
# ═══════════════════════════════════════════════════════════════════════════════
println("─" ^ 95)
println("  1. TAMAÑO DE NUBE — umbral y neuronas_elim fijos")
println("─" ^ 95)
println()

tamanos_nube = [10, 25, 50, 100, 200]

println("  Sonar [60,8,2] — umbral=0.55, elim=1, 200 épocas, LR=0.1")
println("  Nube  │ Acc (mean)  │ F1 (mean)  │ Topología       │ Reducción │ Éxitos")
println("  ──────┼─────────────┼────────────┼─────────────────┼───────────┼────────")
for tn in tamanos_nube
    acc, f1, topo, red, ok = evaluar_config(sonar_trX, sonar_trY, sonar_teX, sonar_teY,
        [60,8,2], 200, 0.1; tamano_nube=tn, umbral=0.55, neuronas_elim=1)
    tstr = ok > 0 ? string(topo) : "—"
    println("  $(rpad(tn, 5)) │ $(rpad(string(round(acc*100,digits=1))*"%", 11)) │ $(rpad(string(round(f1,digits=3)), 10)) │ $(rpad(tstr, 15)) │ $(rpad(string(red)*"%", 9)) │ $ok/$(length(SEEDS))")
end
println()

println("  Ionosphere [34,16,2] — umbral=0.6, elim=1, 200 épocas, LR=0.1")
println("  Nube  │ Acc (mean)  │ F1 (mean)  │ Topología       │ Reducción │ Éxitos")
println("  ──────┼─────────────┼────────────┼─────────────────┼───────────┼────────")
for tn in tamanos_nube
    acc, f1, topo, red, ok = evaluar_config(iono_trX, iono_trY, iono_teX, iono_teY,
        [34,16,2], 200, 0.1; tamano_nube=tn, umbral=0.6, neuronas_elim=1)
    tstr = ok > 0 ? string(topo) : "—"
    println("  $(rpad(tn, 5)) │ $(rpad(string(round(acc*100,digits=1))*"%", 11)) │ $(rpad(string(round(f1,digits=3)), 10)) │ $(rpad(tstr, 15)) │ $(rpad(string(red)*"%", 9)) │ $ok/$(length(SEEDS))")
end
println()

println("  Wine [13,16,3] — umbral=0.4, elim=1, 100 épocas, LR=0.1")
println("  Nube  │ Acc (mean)  │ F1 (mean)  │ Topología       │ Reducción │ Éxitos")
println("  ──────┼─────────────┼────────────┼─────────────────┼───────────┼────────")
for tn in tamanos_nube
    acc, f1, topo, red, ok = evaluar_config(wine_trX, wine_trY, wine_teX, wine_teY,
        [13,16,3], 100, 0.1; tamano_nube=tn, umbral=0.4, neuronas_elim=1)
    tstr = ok > 0 ? string(topo) : "—"
    println("  $(rpad(tn, 5)) │ $(rpad(string(round(acc*100,digits=1))*"%", 11)) │ $(rpad(string(round(f1,digits=3)), 10)) │ $(rpad(tstr, 15)) │ $(rpad(string(red)*"%", 9)) │ $ok/$(length(SEEDS))")
end
println()

# ═══════════════════════════════════════════════════════════════════════════════
# 2. UMBRAL DE ACIERTO (0.3, 0.4, 0.5, 0.6, 0.7)
# ═══════════════════════════════════════════════════════════════════════════════
println("─" ^ 95)
println("  2. UMBRAL DE ACIERTO — nube=50, neuronas_elim fijo")
println("─" ^ 95)
println()

umbrales = [0.3, 0.4, 0.5, 0.6, 0.7]

println("  Sonar [60,8,2] — nube=50, elim=1, 200 épocas, LR=0.1")
println("  Umbral │ Acc (mean)  │ F1 (mean)  │ Topología       │ Reducción │ Éxitos")
println("  ───────┼─────────────┼────────────┼─────────────────┼───────────┼────────")
for u in umbrales
    acc, f1, topo, red, ok = evaluar_config(sonar_trX, sonar_trY, sonar_teX, sonar_teY,
        [60,8,2], 200, 0.1; tamano_nube=50, umbral=u, neuronas_elim=1)
    tstr = ok > 0 ? string(topo) : "—"
    println("  $(rpad(u, 6)) │ $(rpad(string(round(acc*100,digits=1))*"%", 11)) │ $(rpad(string(round(f1,digits=3)), 10)) │ $(rpad(tstr, 15)) │ $(rpad(string(red)*"%", 9)) │ $ok/$(length(SEEDS))")
end
println()

println("  Ionosphere [34,16,2] — nube=50, elim=1, 200 épocas, LR=0.1")
println("  Umbral │ Acc (mean)  │ F1 (mean)  │ Topología       │ Reducción │ Éxitos")
println("  ───────┼─────────────┼────────────┼─────────────────┼───────────┼────────")
for u in umbrales
    acc, f1, topo, red, ok = evaluar_config(iono_trX, iono_trY, iono_teX, iono_teY,
        [34,16,2], 200, 0.1; tamano_nube=50, umbral=u, neuronas_elim=1)
    tstr = ok > 0 ? string(topo) : "—"
    println("  $(rpad(u, 6)) │ $(rpad(string(round(acc*100,digits=1))*"%", 11)) │ $(rpad(string(round(f1,digits=3)), 10)) │ $(rpad(tstr, 15)) │ $(rpad(string(red)*"%", 9)) │ $ok/$(length(SEEDS))")
end
println()

println("  Wine [13,16,3] — nube=50, elim=1, 100 épocas, LR=0.1")
println("  Umbral │ Acc (mean)  │ F1 (mean)  │ Topología       │ Reducción │ Éxitos")
println("  ───────┼─────────────┼────────────┼─────────────────┼───────────┼────────")
for u in umbrales
    acc, f1, topo, red, ok = evaluar_config(wine_trX, wine_trY, wine_teX, wine_teY,
        [13,16,3], 100, 0.1; tamano_nube=50, umbral=u, neuronas_elim=1)
    tstr = ok > 0 ? string(topo) : "—"
    println("  $(rpad(u, 6)) │ $(rpad(string(round(acc*100,digits=1))*"%", 11)) │ $(rpad(string(round(f1,digits=3)), 10)) │ $(rpad(tstr, 15)) │ $(rpad(string(red)*"%", 9)) │ $ok/$(length(SEEDS))")
end
println()

# ═══════════════════════════════════════════════════════════════════════════════
# 3. NEURONAS A ELIMINAR (1, 2, 4)
# ═══════════════════════════════════════════════════════════════════════════════
println("─" ^ 95)
println("  3. NEURONAS A ELIMINAR — nube=50, umbral fijo")
println("─" ^ 95)
println()

elims = [1, 2, 4]

println("  Sonar [60,8,2] — nube=50, umbral=0.55, 200 épocas, LR=0.1")
println("  Elim │ Acc (mean)  │ F1 (mean)  │ Topología       │ Reducción │ Éxitos")
println("  ─────┼─────────────┼────────────┼─────────────────┼───────────┼────────")
for e in elims
    acc, f1, topo, red, ok = evaluar_config(sonar_trX, sonar_trY, sonar_teX, sonar_teY,
        [60,8,2], 200, 0.1; tamano_nube=50, umbral=0.55, neuronas_elim=e)
    tstr = ok > 0 ? string(topo) : "—"
    println("  $(rpad(e, 4)) │ $(rpad(string(round(acc*100,digits=1))*"%", 11)) │ $(rpad(string(round(f1,digits=3)), 10)) │ $(rpad(tstr, 15)) │ $(rpad(string(red)*"%", 9)) │ $ok/$(length(SEEDS))")
end
println()

println("  Ionosphere [34,16,2] — nube=50, umbral=0.6, 200 épocas, LR=0.1")
println("  Elim │ Acc (mean)  │ F1 (mean)  │ Topología       │ Reducción │ Éxitos")
println("  ─────┼─────────────┼────────────┼─────────────────┼───────────┼────────")
for e in elims
    acc, f1, topo, red, ok = evaluar_config(iono_trX, iono_trY, iono_teX, iono_teY,
        [34,16,2], 200, 0.1; tamano_nube=50, umbral=0.6, neuronas_elim=e)
    tstr = ok > 0 ? string(topo) : "—"
    println("  $(rpad(e, 4)) │ $(rpad(string(round(acc*100,digits=1))*"%", 11)) │ $(rpad(string(round(f1,digits=3)), 10)) │ $(rpad(tstr, 15)) │ $(rpad(string(red)*"%", 9)) │ $ok/$(length(SEEDS))")
end
println()

println("  Wine [13,16,3] — nube=50, umbral=0.4, 100 épocas, LR=0.1")
println("  Elim │ Acc (mean)  │ F1 (mean)  │ Topología       │ Reducción │ Éxitos")
println("  ─────┼─────────────┼────────────┼─────────────────┼───────────┼────────")
for e in elims
    acc, f1, topo, red, ok = evaluar_config(wine_trX, wine_trY, wine_teX, wine_teY,
        [13,16,3], 100, 0.1; tamano_nube=50, umbral=0.4, neuronas_elim=e)
    tstr = ok > 0 ? string(topo) : "—"
    println("  $(rpad(e, 4)) │ $(rpad(string(round(acc*100,digits=1))*"%", 11)) │ $(rpad(string(round(f1,digits=3)), 10)) │ $(rpad(tstr, 15)) │ $(rpad(string(red)*"%", 9)) │ $ok/$(length(SEEDS))")
end
println()

println("=" ^ 95)
println("  FIN")
println("=" ^ 95)
