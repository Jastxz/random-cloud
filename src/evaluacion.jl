# Evaluacion — Funciones para calcular precisión de una red sobre un dataset

function evaluar(red::RedNeuronal, entradas::Matrix{Float64}, objetivos::Matrix{Float64})
    n_muestras = size(entradas, 2)
    aciertos = 0
    for k in 1:n_muestras
        salida = feedforward(red, entradas[:, k])
        if argmax(salida) == argmax(objetivos[:, k])
            aciertos += 1
        end
    end
    return aciertos / n_muestras
end
