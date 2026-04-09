# Activaciones — Funciones de activación y sus derivadas
#
# Cada activación se define como un símbolo (:sigmoid, :relu, :identidad)
# y se despacha con funciones inline para máximo rendimiento.

@inline sigmoid(x::Float64) = 1.0 / (1.0 + exp(-x))
@inline sigmoid_deriv_from_output(y::Float64) = y * (1.0 - y)

@inline relu(x::Float64) = max(0.0, x)
@inline relu_deriv_from_output(y::Float64) = y > 0.0 ? 1.0 : 0.0

@inline identidad(x::Float64) = x
@inline identidad_deriv_from_output(::Float64) = 1.0

# Despacho por símbolo — el compilador puede constant-fold si el símbolo es const
@inline function aplicar_activacion(x::Float64, act::Symbol)
    act === :relu && return relu(x)
    act === :identidad && return identidad(x)
    return sigmoid(x)
end

@inline function aplicar_derivada(y::Float64, act::Symbol)
    act === :relu && return relu_deriv_from_output(y)
    act === :identidad && return identidad_deriv_from_output(y)
    return sigmoid_deriv_from_output(y)
end

# Determinar activaciones por capa según configuración
# Regla: capas ocultas usan la activación configurada, capa de salida usa sigmoid (clasificación)
# o identidad (regresión)
function activaciones_por_capa(n_capas_pesos::Int, activacion::Symbol)
    acts = Vector{Symbol}(undef, n_capas_pesos)
    for i in 1:(n_capas_pesos - 1)
        acts[i] = activacion  # capas ocultas
    end
    # Capa de salida: sigmoid para :sigmoid y :relu, identidad para :identidad
    acts[n_capas_pesos] = activacion === :identidad ? :identidad : :sigmoid
    return acts
end


# Versiones genéricas para operaciones batch — compatibles con Float32 (GPU) y Float64 (CPU)
# Broadcastable sobre CuArray y Array gracias a parametrización en T<:AbstractFloat

@inline function aplicar_activacion_batch(x::T, act::Symbol) where T<:AbstractFloat
    act === :relu && return max(zero(T), x)
    act === :identidad && return x
    return one(T) / (one(T) + exp(-x))  # sigmoid
end

@inline function aplicar_derivada_batch(y::T, act::Symbol) where T<:AbstractFloat
    act === :relu && return y > zero(T) ? one(T) : zero(T)
    act === :identidad && return one(T)
    return y * (one(T) - y)  # sigmoid derivative from output
end
