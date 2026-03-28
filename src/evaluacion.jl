# Evaluacion — Funciones para calcular precisión de una red sobre un dataset

function evaluar(red::RedNeuronal, entradas::Matrix{Float64}, objetivos::Matrix{Float64})
    n_muestras = size(entradas, 2)
    aciertos = 0
    # Pre-alocar buffers para feedforward
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
