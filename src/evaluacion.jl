# Evaluacion — Funciones para calcular métricas de una red sobre un dataset

# --- Clasificación: proporción de aciertos (argmax) ---

function evaluar(red::RedNeuronal, entradas::Matrix{Float64}, objetivos::Matrix{Float64})
    n_muestras = size(entradas, 2)
    aciertos = 0
    n_capas = length(red.pesos)
    buffers = [Vector{Float64}(undef, red.topologia[i+1]) for i in 1:n_capas]
    @inbounds for k in 1:n_muestras
        salida = feedforward!(red, @view(entradas[:, k]), buffers)
        if argmax(salida) == argmax(@view objetivos[:, k])
            aciertos += 1
        end
    end
    return aciertos / n_muestras
end

# --- Regresión: R² (coeficiente de determinación) clamped a [0, 1] ---

function evaluar_regresion(red::RedNeuronal, entradas::Matrix{Float64}, objetivos::Matrix{Float64})
    n_muestras = size(entradas, 2)
    n_salidas = size(objetivos, 1)
    n_capas = length(red.pesos)
    buffers = [Vector{Float64}(undef, red.topologia[i+1]) for i in 1:n_capas]

    # Calcular media de objetivos por componente
    media_obj = zeros(n_salidas)
    @inbounds for k in 1:n_muestras
        for j in 1:n_salidas
            media_obj[j] += objetivos[j, k]
        end
    end
    media_obj ./= n_muestras

    # Calcular SS_res y SS_tot
    ss_res = 0.0
    ss_tot = 0.0
    @inbounds for k in 1:n_muestras
        salida = feedforward!(red, @view(entradas[:, k]), buffers)
        for j in 1:n_salidas
            ss_res += (objetivos[j, k] - salida[j])^2
            ss_tot += (objetivos[j, k] - media_obj[j])^2
        end
    end

    # R² = 1 - SS_res/SS_tot, clamped a [0, 1]
    if ss_tot == 0.0
        return ss_res == 0.0 ? 1.0 : 0.0
    end
    r2 = 1.0 - ss_res / ss_tot
    return clamp(r2, 0.0, 1.0)
end
