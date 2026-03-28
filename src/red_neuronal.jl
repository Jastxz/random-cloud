# RedNeuronal — Red neuronal feedforward con pesos y biases

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

sigmoid(x::Float64) = 1.0 / (1.0 + exp(-x))

sigmoid_deriv(x::Float64) = x * (1.0 - x)

function feedforward(red::RedNeuronal, entrada::Vector{Float64})
    x = entrada
    for i in 1:length(red.pesos)
        x = sigmoid.(red.pesos[i] * x .+ red.biases[i])
    end
    return x
end

function entrenar!(red::RedNeuronal, entrada::Vector{Float64},
                   objetivo::Vector{Float64}, lr::Float64)
    # Forward pass almacenando activaciones por capa
    n_capas = length(red.pesos)
    activaciones = Vector{Vector{Float64}}(undef, n_capas + 1)
    activaciones[1] = entrada
    for i in 1:n_capas
        activaciones[i + 1] = sigmoid.(red.pesos[i] * activaciones[i] .+ red.biases[i])
    end

    # Retropropagación
    delta = (activaciones[end] .- objetivo) .* sigmoid_deriv.(activaciones[end])

    for i in n_capas:-1:1
        # Actualizar pesos y biases
        red.pesos[i] .-= lr .* (delta * activaciones[i]')
        red.biases[i] .-= lr .* delta
        # Calcular delta para la capa anterior (solo si no es la primera capa)
        if i > 1
            delta = (red.pesos[i]' * delta) .* sigmoid_deriv.(activaciones[i])
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
    # Filtrar la topología y los pesos/biases correspondientes
    # Una capa oculta es cualquier capa que no sea la primera ni la última
    topo_filtrada = Int[nueva_topologia[1]]
    pesos_filtrados = Matrix{Float64}[]
    biases_filtrados = Vector{Float64}[]

    # Identificar qué capas mantener (entrada, salida, y ocultas con >0 neuronas)
    mantener = Bool[true]  # capa de entrada siempre se mantiene
    for j in 2:(n_capas - 1)
        push!(mantener, nueva_topologia[j] > 0)
    end
    push!(mantener, true)  # capa de salida siempre se mantiene

    # Construir topología filtrada
    for j in 2:n_capas
        if mantener[j]
            push!(topo_filtrada, nueva_topologia[j])
        end
    end

    # Construir pesos y biases para la topología filtrada
    # Recorrer las transiciones entre capas mantenidas
    capas_mantenidas = [j for j in 1:n_capas if mantener[j]]
    for k in 1:(length(capas_mantenidas) - 1)
        from_idx = capas_mantenidas[k]
        to_idx = capas_mantenidas[k + 1]

        if to_idx == from_idx + 1
            # Capas adyacentes en la topología original: usar pesos recortados directamente
            push!(pesos_filtrados, nuevos_pesos[from_idx])
            push!(biases_filtrados, nuevos_biases[from_idx])
        else
            # Capas no adyacentes: hay capas con 0 neuronas en medio
            # Crear conexión directa con pesos cero (no hay pesos que preservar)
            from_size = nueva_topologia[from_idx]
            to_size = nueva_topologia[to_idx]
            push!(pesos_filtrados, zeros(to_size, from_size))
            push!(biases_filtrados, zeros(to_size))
        end
    end

    return RedNeuronal(topo_filtrada, pesos_filtrados, biases_filtrados)
end
