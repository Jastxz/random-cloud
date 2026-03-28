# RedNeuronal — Red neuronal feedforward con pesos y biases

using LinearAlgebra: mul!

struct RedNeuronal
    topologia::Vector{Int}
    pesos::Vector{Matrix{Float64}}
    biases::Vector{Vector{Float64}}
end

function RedNeuronal(topologia::Vector{Int}, rng::AbstractRNG)
    topologia = copy(topologia)
    n_capas = length(topologia)
    pesos = [2.0 .* rand(rng, topologia[i+1], topologia[i]) .- 1.0
             for i in 1:(n_capas - 1)]
    biases = [2.0 .* rand(rng, topologia[i+1]) .- 1.0
              for i in 1:(n_capas - 1)]
    RedNeuronal(topologia, pesos, biases)
end

@inline sigmoid(x::Float64) = 1.0 / (1.0 + exp(-x))

@inline sigmoid_deriv(x::Float64) = x * (1.0 - x)

# --- Feedforward con pre-alocación de buffers ---

function feedforward(red::RedNeuronal, entrada::AbstractVector{Float64})
    x = entrada
    for i in 1:length(red.pesos)
        x = sigmoid.(red.pesos[i] * x .+ red.biases[i])
    end
    return x
end

# Versión in-place que reutiliza buffers pre-alocados (para el hot path)
# buffers[i] debe tener longitud topologia[i+1]
function feedforward!(red::RedNeuronal, entrada::AbstractVector{Float64},
                      buffers::Vector{Vector{Float64}})
    x = entrada
    @inbounds for i in 1:length(red.pesos)
        buf = buffers[i]
        mul!(buf, red.pesos[i], x)
        b = red.biases[i]
        @simd for j in eachindex(buf)
            buf[j] = sigmoid(buf[j] + b[j])
        end
        x = buf
    end
    return x
end

# --- Entrenamiento con pre-alocación ---

# Estructura para buffers reutilizables durante entrenamiento
struct EntrenarBuffers
    activaciones::Vector{Vector{Float64}}
    deltas::Vector{Vector{Float64}}
    grad_pesos::Vector{Matrix{Float64}}  # buffer para outer product
end

function EntrenarBuffers(topologia::Vector{Int})
    n_capas = length(topologia) - 1
    activaciones = [Vector{Float64}(undef, topologia[i]) for i in 1:length(topologia)]
    deltas = [Vector{Float64}(undef, topologia[i+1]) for i in 1:n_capas]
    grad_pesos = [Matrix{Float64}(undef, topologia[i+1], topologia[i]) for i in 1:n_capas]
    EntrenarBuffers(activaciones, deltas, grad_pesos)
end


function entrenar!(red::RedNeuronal, entrada::AbstractVector{Float64},
                   objetivo::AbstractVector{Float64}, lr::Float64)
    # Forward pass almacenando activaciones por capa
    n_capas = length(red.pesos)
    activaciones = Vector{Vector{Float64}}(undef, n_capas + 1)
    activaciones[1] = collect(entrada)
    for i in 1:n_capas
        activaciones[i + 1] = sigmoid.(red.pesos[i] * activaciones[i] .+ red.biases[i])
    end

    # Retropropagación
    delta = (activaciones[end] .- objetivo) .* sigmoid_deriv.(activaciones[end])

    for i in n_capas:-1:1
        red.pesos[i] .-= lr .* (delta * activaciones[i]')
        red.biases[i] .-= lr .* delta
        if i > 1
            delta = (red.pesos[i]' * delta) .* sigmoid_deriv.(activaciones[i])
        end
    end

    return nothing
end

# Versión optimizada que reutiliza buffers pre-alocados
function entrenar!(red::RedNeuronal, entrada::AbstractVector{Float64},
                   objetivo::AbstractVector{Float64}, lr::Float64,
                   bufs::EntrenarBuffers)
    n_capas = length(red.pesos)
    acts = bufs.activaciones

    # Forward pass: mul! evita alocación del resultado de W*x
    copyto!(acts[1], entrada)
    @inbounds for i in 1:n_capas
        buf = acts[i + 1]
        mul!(buf, red.pesos[i], acts[i])
        buf .= sigmoid.(buf .+ red.biases[i])
    end

    # Retropropagación: reutilizar buffers de deltas y grad_pesos
    delta = bufs.deltas[n_capas]
    delta .= (acts[n_capas + 1] .- objetivo) .* sigmoid_deriv.(acts[n_capas + 1])

    @inbounds for i in n_capas:-1:1
        # Outer product in-place → grad_pesos[i]
        grad = bufs.grad_pesos[i]
        mul!(grad, delta, acts[i]')

        # Actualizar pesos y biases (in-place con broadcasting fusionado)
        red.pesos[i] .-= lr .* grad
        red.biases[i] .-= lr .* delta

        # Delta para la capa anterior
        if i > 1
            delta_prev = bufs.deltas[i - 1]
            mul!(delta_prev, red.pesos[i]', delta)
            delta_prev .*= sigmoid_deriv.(acts[i])
            delta = delta_prev
        end
    end

    return nothing
end

function reconstruir(red::RedNeuronal, nueva_topologia::Vector{Int})
    n_capas = length(nueva_topologia)

    # Paso 1: Recortar pesos y biases a las dimensiones de la nueva topología
    nuevos_pesos = Matrix{Float64}[]
    nuevos_biases = Vector{Float64}[]
    for i in 1:(n_capas - 1)
        filas = nueva_topologia[i + 1]
        cols = nueva_topologia[i]
        push!(nuevos_pesos, red.pesos[i][1:filas, 1:cols])
        push!(nuevos_biases, red.biases[i][1:filas])
    end

    # Paso 2: Eliminar capas ocultas con 0 neuronas
    topo_filtrada = Int[nueva_topologia[1]]
    pesos_filtrados = Matrix{Float64}[]
    biases_filtrados = Vector{Float64}[]

    mantener = Bool[true]
    for j in 2:(n_capas - 1)
        push!(mantener, nueva_topologia[j] > 0)
    end
    push!(mantener, true)

    for j in 2:n_capas
        if mantener[j]
            push!(topo_filtrada, nueva_topologia[j])
        end
    end

    capas_mantenidas = [j for j in 1:n_capas if mantener[j]]
    for k in 1:(length(capas_mantenidas) - 1)
        from_idx = capas_mantenidas[k]
        to_idx = capas_mantenidas[k + 1]

        if to_idx == from_idx + 1
            push!(pesos_filtrados, nuevos_pesos[from_idx])
            push!(biases_filtrados, nuevos_biases[from_idx])
        else
            from_size = nueva_topologia[from_idx]
            to_size = nueva_topologia[to_idx]
            push!(pesos_filtrados, zeros(to_size, from_size))
            push!(biases_filtrados, zeros(to_size))
        end
    end

    return RedNeuronal(topo_filtrada, pesos_filtrados, biases_filtrados)
end
