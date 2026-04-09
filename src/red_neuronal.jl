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

# --- Feedforward ---

function feedforward(red::RedNeuronal, entrada::AbstractVector{Float64},
                     acts::Vector{Symbol})
    x = entrada
    @inbounds for i in 1:length(red.pesos)
        act = acts[i]
        x = [aplicar_activacion(v, act) for v in red.pesos[i] * x .+ red.biases[i]]
    end
    return x
end

# Versión legacy sin activaciones (usa sigmoid en todas las capas)
function feedforward(red::RedNeuronal, entrada::AbstractVector{Float64})
    x = entrada
    for i in 1:length(red.pesos)
        x = sigmoid.(red.pesos[i] * x .+ red.biases[i])
    end
    return x
end

# Versión in-place que reutiliza buffers pre-alocados (para el hot path)
function feedforward!(red::RedNeuronal, entrada::AbstractVector{Float64},
                      buffers::Vector{Vector{Float64}},
                      acts::Vector{Symbol})
    x = entrada
    @inbounds for i in 1:length(red.pesos)
        buf = buffers[i]
        mul!(buf, red.pesos[i], x)
        b = red.biases[i]
        act = acts[i]
        @simd for j in eachindex(buf)
            buf[j] = aplicar_activacion(buf[j] + b[j], act)
        end
        x = buf
    end
    return x
end

# Versión legacy sin activaciones (sigmoid en todas las capas)
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

# --- Batched Feedforward ---
# Computes feedforward for all samples simultaneously via matrix operations.
# Works on both Matrix{Float64} (CPU) and CuMatrix{Float32} (GPU) via dispatch.

"""
    feedforward_batch(pesos, biases, X, acts) → Y

Compute feedforward for all samples simultaneously.
- pesos: Vector of weight matrices per layer
- biases: Vector of bias vectors per layer
- X: input data matrix (features × samples)
- acts: activation symbol per layer
Returns Y: output matrix (output_dim × samples)
"""
function feedforward_batch(
    pesos::Vector{<:AbstractMatrix{T}},
    biases::Vector{<:AbstractVector{T}},
    X::AbstractMatrix{T},
    acts::Vector{Symbol}
) where T<:AbstractFloat
    A = X
    for i in eachindex(pesos)
        Z = pesos[i] * A .+ biases[i]
        A = aplicar_activacion_batch.(Z, acts[i])
    end
    return A
end

# Convenience method: defaults to sigmoid activation on every layer (Req 1.4)
function feedforward_batch(
    pesos::Vector{<:AbstractMatrix{T}},
    biases::Vector{<:AbstractVector{T}},
    X::AbstractMatrix{T}
) where T<:AbstractFloat
    acts = fill(:sigmoid, length(pesos))
    return feedforward_batch(pesos, biases, X, acts)
end

# --- Entrenamiento con pre-alocación ---

struct EntrenarBuffers
    activaciones::Vector{Vector{Float64}}
    deltas::Vector{Vector{Float64}}
    grad_pesos::Vector{Matrix{Float64}}
end

function EntrenarBuffers(topologia::Vector{Int})
    n_capas = length(topologia) - 1
    activaciones = [Vector{Float64}(undef, topologia[i]) for i in 1:length(topologia)]
    deltas = [Vector{Float64}(undef, topologia[i+1]) for i in 1:n_capas]
    grad_pesos = [Matrix{Float64}(undef, topologia[i+1], topologia[i]) for i in 1:n_capas]
    EntrenarBuffers(activaciones, deltas, grad_pesos)
end

# Versión legacy sin activaciones (sigmoid)
function entrenar!(red::RedNeuronal, entrada::AbstractVector{Float64},
                   objetivo::AbstractVector{Float64}, lr::Float64)
    n_capas = length(red.pesos)
    activaciones = Vector{Vector{Float64}}(undef, n_capas + 1)
    activaciones[1] = collect(entrada)
    for i in 1:n_capas
        activaciones[i + 1] = sigmoid.(red.pesos[i] * activaciones[i] .+ red.biases[i])
    end
    delta = (activaciones[end] .- objetivo) .* sigmoid_deriv_from_output.(activaciones[end])
    for i in n_capas:-1:1
        red.pesos[i] .-= lr .* (delta * activaciones[i]')
        red.biases[i] .-= lr .* delta
        if i > 1
            delta = (red.pesos[i]' * delta) .* sigmoid_deriv_from_output.(activaciones[i])
        end
    end
    return nothing
end

# Versión optimizada con buffers (sigmoid)
function entrenar!(red::RedNeuronal, entrada::AbstractVector{Float64},
                   objetivo::AbstractVector{Float64}, lr::Float64,
                   bufs::EntrenarBuffers)
    n_capas = length(red.pesos)
    acts = bufs.activaciones
    copyto!(acts[1], entrada)
    @inbounds for i in 1:n_capas
        buf = acts[i + 1]
        mul!(buf, red.pesos[i], acts[i])
        buf .= sigmoid.(buf .+ red.biases[i])
    end
    delta = bufs.deltas[n_capas]
    delta .= (acts[n_capas + 1] .- objetivo) .* sigmoid_deriv_from_output.(acts[n_capas + 1])
    @inbounds for i in n_capas:-1:1
        grad = bufs.grad_pesos[i]
        mul!(grad, delta, acts[i]')
        red.pesos[i] .-= lr .* grad
        red.biases[i] .-= lr .* delta
        if i > 1
            delta_prev = bufs.deltas[i - 1]
            mul!(delta_prev, red.pesos[i]', delta)
            delta_prev .*= sigmoid_deriv_from_output.(acts[i])
            delta = delta_prev
        end
    end
    return nothing
end

# Versión optimizada con buffers Y activaciones configurables
function entrenar!(red::RedNeuronal, entrada::AbstractVector{Float64},
                   objetivo::AbstractVector{Float64}, lr::Float64,
                   bufs::EntrenarBuffers, acts::Vector{Symbol})
    n_capas = length(red.pesos)
    as = bufs.activaciones
    copyto!(as[1], entrada)
    @inbounds for i in 1:n_capas
        buf = as[i + 1]
        mul!(buf, red.pesos[i], as[i])
        act = acts[i]
        b = red.biases[i]
        @simd for j in eachindex(buf)
            buf[j] = aplicar_activacion(buf[j] + b[j], act)
        end
    end
    delta = bufs.deltas[n_capas]
    act_last = acts[n_capas]
    @inbounds @simd for j in eachindex(delta)
        delta[j] = (as[n_capas + 1][j] - objetivo[j]) * aplicar_derivada(as[n_capas + 1][j], act_last)
    end
    @inbounds for i in n_capas:-1:1
        grad = bufs.grad_pesos[i]
        mul!(grad, delta, as[i]')
        red.pesos[i] .-= lr .* grad
        red.biases[i] .-= lr .* delta
        if i > 1
            delta_prev = bufs.deltas[i - 1]
            mul!(delta_prev, red.pesos[i]', delta)
            act_i = acts[i - 1]
            @simd for j in eachindex(delta_prev)
                delta_prev[j] *= aplicar_derivada(as[i][j], act_i)
            end
            delta = delta_prev
        end
    end
    return nothing
end

# --- Mini-batch entrenamiento ---
# Acumula gradientes sobre un batch y actualiza una vez
function entrenar_batch!(red::RedNeuronal, entradas::Matrix{Float64},
                         objetivos::Matrix{Float64}, indices::AbstractVector{Int},
                         lr::Float64, bufs::EntrenarBuffers, acts::Vector{Symbol})
    n_capas = length(red.pesos)
    batch_size = length(indices)
    batch_size == 0 && return nothing
    lr_batch = lr / batch_size

    # Acumuladores de gradientes (reutilizamos grad_pesos para el último, necesitamos acumuladores separados)
    # Para simplicidad y 0-alloc: acumulamos directamente en pesos/biases con lr/batch_size
    # Esto es equivalente a: grad_acum = sum(grads) / batch_size; pesos -= lr * grad_acum
    # Reescrito como: pesos -= (lr/batch_size) * sum(grads)

    @inbounds for idx in indices
        as = bufs.activaciones
        copyto!(as[1], @view(entradas[:, idx]))

        # Forward
        for i in 1:n_capas
            buf = as[i + 1]
            mul!(buf, red.pesos[i], as[i])
            act = acts[i]
            b = red.biases[i]
            @simd for j in eachindex(buf)
                buf[j] = aplicar_activacion(buf[j] + b[j], act)
            end
        end

        # Backward
        delta = bufs.deltas[n_capas]
        act_last = acts[n_capas]
        @simd for j in eachindex(delta)
            delta[j] = (as[n_capas + 1][j] - objetivos[j, idx]) * aplicar_derivada(as[n_capas + 1][j], act_last)
        end

        for i in n_capas:-1:1
            grad = bufs.grad_pesos[i]
            mul!(grad, delta, as[i]')
            red.pesos[i] .-= lr_batch .* grad
            red.biases[i] .-= lr_batch .* delta
            if i > 1
                delta_prev = bufs.deltas[i - 1]
                mul!(delta_prev, red.pesos[i]', delta)
                act_i = acts[i - 1]
                @simd for j in eachindex(delta_prev)
                    delta_prev[j] *= aplicar_derivada(as[i][j], act_i)
                end
                delta = delta_prev
            end
        end
    end
    return nothing
end

# --- Batched matrix backpropagation ---
# Full-batch gradient computation via matrix ops — no sample loop.
# Numerically equivalent to entrenar_batch! (lr/B per sample accumulation).
# Works on both Matrix{Float64} (CPU) and CuMatrix{Float32} (GPU) via dispatch.

"""
    entrenar_batch_matmul!(pesos, biases, X_batch, Y_batch, lr, acts) → nothing

Full-batch matrix backpropagation.
- X_batch: (features × B) mini-batch input
- Y_batch: (outputs × B) mini-batch targets
All operations are matrix multiplications; no sample loop.
"""
function entrenar_batch_matmul!(
    pesos::Vector{<:AbstractMatrix{T}},
    biases::Vector{<:AbstractVector{T}},
    X_batch::AbstractMatrix{T},
    Y_batch::AbstractMatrix{T},
    lr::T,
    acts::Vector{Symbol}
) where T<:AbstractFloat
    n_capas = length(pesos)
    B = T(size(X_batch, 2))

    # --- Forward pass: store activations A[0..L] as matrices (neurons × B) ---
    activaciones = Vector{AbstractMatrix{T}}(undef, n_capas + 1)
    activaciones[1] = X_batch
    for i in 1:n_capas
        Z = pesos[i] * activaciones[i] .+ biases[i]
        activaciones[i + 1] = aplicar_activacion_batch.(Z, acts[i])
    end

    # --- Backward pass ---
    # δ[L] = (A[L] - Y) .* f'(A[L])
    delta = (activaciones[n_capas + 1] .- Y_batch) .* aplicar_derivada_batch.(activaciones[n_capas + 1], acts[n_capas])

    for i in n_capas:-1:1
        # ∇W[i] = (1/B) * δ * A[i-1]'
        grad_w = (one(T) / B) .* (delta * activaciones[i]')
        # ∇b[i] = (1/B) * sum(δ, dims=2)
        grad_b = vec(sum(delta, dims=2)) .* (one(T) / B)

        # Update weights and biases
        pesos[i] .-= lr .* grad_w
        biases[i] .-= lr .* grad_b

        # Propagate delta to previous layer (if not first layer)
        if i > 1
            delta = (pesos[i]' * delta) .* aplicar_derivada_batch.(activaciones[i], acts[i - 1])
        end
    end

    return nothing
end

function reconstruir(red::RedNeuronal, nueva_topologia::Vector{Int})
    n_capas = length(nueva_topologia)

    nuevos_pesos = Matrix{Float64}[]
    nuevos_biases = Vector{Float64}[]
    for i in 1:(n_capas - 1)
        filas = nueva_topologia[i + 1]
        cols = nueva_topologia[i]
        push!(nuevos_pesos, red.pesos[i][1:filas, 1:cols])
        push!(nuevos_biases, red.biases[i][1:filas])
    end

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
