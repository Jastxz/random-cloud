# Evaluacion — Funciones para calcular métricas de una red sobre un dataset

# --- Clasificación: proporción de aciertos (argmax) ---

function evaluar(red::RedNeuronal, entradas::Matrix{Float64}, objetivos::Matrix{Float64};
                 acts::Union{Vector{Symbol}, Nothing}=nothing)
    n_muestras = size(entradas, 2)
    aciertos = 0
    n_capas = length(red.pesos)
    buffers = [Vector{Float64}(undef, red.topologia[i+1]) for i in 1:n_capas]
    if acts === nothing
        @inbounds for k in 1:n_muestras
            salida = feedforward!(red, @view(entradas[:, k]), buffers)
            if argmax(salida) == argmax(@view objetivos[:, k])
                aciertos += 1
            end
        end
    else
        @inbounds for k in 1:n_muestras
            salida = feedforward!(red, @view(entradas[:, k]), buffers, acts)
            if argmax(salida) == argmax(@view objetivos[:, k])
                aciertos += 1
            end
        end
    end
    return aciertos / n_muestras
end

# --- Regresión: R² (coeficiente de determinación) clamped a [0, 1] ---

function evaluar_regresion(red::RedNeuronal, entradas::Matrix{Float64}, objetivos::Matrix{Float64};
                           acts::Union{Vector{Symbol}, Nothing}=nothing)
    n_muestras = size(entradas, 2)
    n_salidas = size(objetivos, 1)
    n_capas = length(red.pesos)
    buffers = [Vector{Float64}(undef, red.topologia[i+1]) for i in 1:n_capas]

    media_obj = zeros(n_salidas)
    @inbounds for k in 1:n_muestras
        for j in 1:n_salidas
            media_obj[j] += objetivos[j, k]
        end
    end
    media_obj ./= n_muestras

    ss_res = 0.0
    ss_tot = 0.0
    if acts === nothing
        @inbounds for k in 1:n_muestras
            salida = feedforward!(red, @view(entradas[:, k]), buffers)
            for j in 1:n_salidas
                ss_res += (objetivos[j, k] - salida[j])^2
                ss_tot += (objetivos[j, k] - media_obj[j])^2
            end
        end
    else
        @inbounds for k in 1:n_muestras
            salida = feedforward!(red, @view(entradas[:, k]), buffers, acts)
            for j in 1:n_salidas
                ss_res += (objetivos[j, k] - salida[j])^2
                ss_tot += (objetivos[j, k] - media_obj[j])^2
            end
        end
    end

    if ss_tot == 0.0
        return ss_res == 0.0 ? 1.0 : 0.0
    end
    r2 = 1.0 - ss_res / ss_tot
    return clamp(r2, 0.0, 1.0)
end

# --- F1-Score (macro-averaged para multiclase, estándar para binario) ---

"""
    evaluar_f1(red, entradas, objetivos; acts=nothing) → Float64

Calcula el F1-score macro-averaged. Para cada clase calcula precision y recall,
luego promedia los F1 por clase. Rango: [0, 1].

Para problemas binarios (2 salidas), equivale al F1 de la clase positiva (índice 2)
promediado con el de la clase negativa.
"""
function evaluar_f1(red::RedNeuronal, entradas::Matrix{Float64}, objetivos::Matrix{Float64};
                    acts::Union{Vector{Symbol}, Nothing}=nothing)
    n_muestras = size(entradas, 2)
    n_clases = size(objetivos, 1)
    n_capas = length(red.pesos)
    buffers = [Vector{Float64}(undef, red.topologia[i+1]) for i in 1:n_capas]

    # Contadores por clase: true positives, false positives, false negatives
    tp = zeros(Int, n_clases)
    fp = zeros(Int, n_clases)
    fn = zeros(Int, n_clases)

    @inbounds for k in 1:n_muestras
        salida = acts === nothing ?
            feedforward!(red, @view(entradas[:, k]), buffers) :
            feedforward!(red, @view(entradas[:, k]), buffers, acts)
        pred = argmax(salida)
        real = argmax(@view objetivos[:, k])
        if pred == real
            tp[pred] += 1
        else
            fp[pred] += 1
            fn[real] += 1
        end
    end

    # Macro-averaged F1: promedio de F1 por clase
    f1_sum = 0.0
    clases_con_soporte = 0
    for c in 1:n_clases
        # Solo considerar clases que tienen al menos una muestra real
        if tp[c] + fn[c] > 0
            prec = tp[c] + fp[c] > 0 ? tp[c] / (tp[c] + fp[c]) : 0.0
            rec = tp[c] / (tp[c] + fn[c])
            f1_c = prec + rec > 0.0 ? 2.0 * prec * rec / (prec + rec) : 0.0
            f1_sum += f1_c
            clases_con_soporte += 1
        end
    end

    return clases_con_soporte > 0 ? f1_sum / clases_con_soporte : 0.0
end

# --- AUC (Area Under ROC Curve) para clasificación binaria ---

"""
    evaluar_auc(red, entradas, objetivos; acts=nothing) → Float64

Calcula el AUC-ROC para problemas binarios (2 salidas).
Usa la probabilidad de la clase positiva (índice 2) como score.
Implementación por el método del trapecio sobre la curva ROC.

Para problemas multiclase (>2 salidas), calcula macro-averaged AUC (one-vs-rest).
Rango: [0, 1]. AUC=0.5 → aleatorio, AUC=1.0 → perfecto.
"""
function evaluar_auc(red::RedNeuronal, entradas::Matrix{Float64}, objetivos::Matrix{Float64};
                     acts::Union{Vector{Symbol}, Nothing}=nothing)
    n_muestras = size(entradas, 2)
    n_clases = size(objetivos, 1)
    n_capas = length(red.pesos)
    buffers = [Vector{Float64}(undef, red.topologia[i+1]) for i in 1:n_capas]

    if n_clases == 2
        return _auc_binario(red, entradas, objetivos, buffers, acts)
    else
        # Macro-averaged AUC: one-vs-rest para cada clase
        auc_sum = 0.0
        clases_validas = 0
        for c in 1:n_clases
            auc_c = _auc_one_vs_rest(red, entradas, objetivos, buffers, acts, c)
            if auc_c >= 0.0  # -1 indica clase sin positivos o sin negativos
                auc_sum += auc_c
                clases_validas += 1
            end
        end
        return clases_validas > 0 ? auc_sum / clases_validas : 0.5
    end
end

# AUC binario: usa salida[2] como score de la clase positiva
function _auc_binario(red::RedNeuronal, entradas::Matrix{Float64},
                      objetivos::Matrix{Float64},
                      buffers::Vector{Vector{Float64}},
                      acts::Union{Vector{Symbol}, Nothing})
    n_muestras = size(entradas, 2)
    scores = Vector{Float64}(undef, n_muestras)
    labels = Vector{Bool}(undef, n_muestras)

    @inbounds for k in 1:n_muestras
        salida = acts === nothing ?
            feedforward!(red, @view(entradas[:, k]), buffers) :
            feedforward!(red, @view(entradas[:, k]), buffers, acts)
        scores[k] = salida[2]  # probabilidad de clase positiva
        labels[k] = objetivos[2, k] > 0.5
    end

    return _calcular_auc(scores, labels)
end

# AUC one-vs-rest para una clase específica
function _auc_one_vs_rest(red::RedNeuronal, entradas::Matrix{Float64},
                          objetivos::Matrix{Float64},
                          buffers::Vector{Vector{Float64}},
                          acts::Union{Vector{Symbol}, Nothing},
                          clase::Int)
    n_muestras = size(entradas, 2)
    scores = Vector{Float64}(undef, n_muestras)
    labels = Vector{Bool}(undef, n_muestras)

    @inbounds for k in 1:n_muestras
        salida = acts === nothing ?
            feedforward!(red, @view(entradas[:, k]), buffers) :
            feedforward!(red, @view(entradas[:, k]), buffers, acts)
        scores[k] = salida[clase]
        labels[k] = argmax(@view objetivos[:, k]) == clase
    end

    # Si no hay positivos o no hay negativos, AUC no está definido
    n_pos = count(labels)
    if n_pos == 0 || n_pos == n_muestras
        return -1.0
    end

    return _calcular_auc(scores, labels)
end

# Cálculo de AUC por método del trapecio (ordenar por score descendente)
function _calcular_auc(scores::Vector{Float64}, labels::Vector{Bool})
    n = length(scores)
    n_pos = count(labels)
    n_neg = n - n_pos

    if n_pos == 0 || n_neg == 0
        return 0.5  # indefinido, devolvemos aleatorio
    end

    # Ordenar por score descendente
    orden = sortperm(scores, rev=true)

    tp = 0.0
    fp = 0.0
    auc = 0.0
    tp_prev = 0.0
    fp_prev = 0.0

    @inbounds for i in 1:n
        idx = orden[i]
        if labels[idx]
            tp += 1.0
        else
            fp += 1.0
        end
        # Cuando cambia el score (o al final), calcular área del trapecio
        if i == n || scores[orden[i]] != scores[orden[min(i + 1, n)]]
            # Trapecio: (fp - fp_prev) * (tp + tp_prev) / 2
            auc += (fp - fp_prev) * (tp + tp_prev) / 2.0
            tp_prev = tp
            fp_prev = fp
        end
    end

    return auc / (n_pos * n_neg)
end
