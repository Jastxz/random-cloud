# gpu_backend.jl — GPU utilities for RandomCloud CUDA extension
#
# Implements: a_gpu, de_gpu, estimar_vram, verificar_gpu, _ejecutar_gpu

# --- Data transfer utilities ---

"""
    RandomCloud.a_gpu(x::AbstractArray{Float64}) → CuArray{Float32}

Convert Float64 array to Float32 and transfer to GPU device memory.
"""
function RandomCloud.a_gpu(x::AbstractArray{Float64})
    return CUDA.cu(Float32.(x))
end

"""
    RandomCloud.de_gpu(x::CuArray{Float32}) → Array{Float64}

Transfer CuArray back to host memory and convert Float32 → Float64.
"""
function RandomCloud.de_gpu(x::CuArray{Float32})
    return Float64.(Array(x))
end

# --- GPU verification ---

"""
    RandomCloud.verificar_gpu() → nothing

Check that CUDA is functional and a device is available.
Throws an informative `ArgumentError` if not.
"""
function RandomCloud.verificar_gpu()
    if !CUDA.functional()
        throw(ArgumentError(
            "No CUDA-capable GPU detected. Set gpu=false or install a CUDA driver."
        ))
    end
    return nothing
end

# --- VRAM estimation ---

"""
    RandomCloud.estimar_vram(config::ConfiguracionNube, entradas::Matrix{Float64}) → Float64

Estimate peak GPU memory usage in bytes for a full cloud run.
Accounts for input data, target data, per-layer weights/biases, activation buffers,
and a 20% overhead margin for the CUDA allocator.

Throws `ArgumentError` if the estimate exceeds 3.5 GB.
"""
function RandomCloud.estimar_vram(config::ConfiguracionNube, entradas::Matrix{Float64})
    n_features = size(entradas, 1)
    n_samples = size(entradas, 2)
    topo = config.topologia_inicial
    N = config.tamano_nube

    total = 0
    # Input data
    total += n_features * n_samples * sizeof(Float32)
    # Target data (output layer × samples)
    total += topo[end] * n_samples * sizeof(Float32)

    # Per-layer weights and biases
    for i in 1:(length(topo) - 1)
        neurons_out = topo[i + 1]
        neurons_in = topo[i]
        total += neurons_out * neurons_in * N * sizeof(Float32)  # weights
        total += neurons_out * N * sizeof(Float32)               # biases
    end

    # Activation buffers (forward + backward, per-network during evaluation)
    max_layer = maximum(topo[2:end])
    total += max_layer * n_samples * 2 * sizeof(Float32)

    # 20% overhead for CUDA allocator fragmentation
    estimated = total * 1.2

    if estimated > 3.5e9  # 3.5 GB
        error_msg = "Estimated VRAM: $(round(estimated / 1e9, digits=2)) GB exceeds limit (3.5 GB). " *
                    "Reduce tamano_nube (currently $(N)) or use a smaller dataset " *
                    "(currently $(n_samples) samples × $(n_features) features)."
        throw(ArgumentError(error_msg))
    end

    return estimated
end

# --- Adaptive batching strategy ---

"""
    _elegir_estrategia(config, n_features, n_samples) → Symbol

Determine GPU execution strategy based on estimated VRAM usage.
Returns `:gpu_batch` if the problem fits in VRAM, `:error` otherwise.
"""
function _elegir_estrategia(config::ConfiguracionNube, n_features::Int, n_samples::Int)
    topo = config.topologia_inicial
    N = config.tamano_nube

    total = 0
    total += n_features * n_samples * sizeof(Float32)
    total += topo[end] * n_samples * sizeof(Float32)
    for i in 1:(length(topo) - 1)
        neurons_out = topo[i + 1]
        neurons_in = topo[i]
        total += neurons_out * neurons_in * N * sizeof(Float32)
        total += neurons_out * N * sizeof(Float32)
    end
    max_layer = maximum(topo[2:end])
    total += max_layer * n_samples * 2 * sizeof(Float32)
    estimated = total * 1.2

    if estimated < 3.5e9 * 0.85
        return :gpu_batch
    else
        return :error
    end
end

# --- GPU execution path ---

"""
    RandomCloud._ejecutar_gpu(motor::MotorNube) → InformeNube

Full GPU execution path for the Random Cloud Method.

1. Verify GPU availability and estimate VRAM (fail fast)
2. Run exploration phase on CPU (topology changes per network)
3. Transfer best network's weights to GPU for refinement
4. Run batched backpropagation on GPU
5. Transfer final results back to host
6. Record gpu_tiempo_ms and pico_vram_mb in InformeNube
"""
function RandomCloud._ejecutar_gpu(motor::MotorNube)
    config = motor.config
    entradas = motor.entradas
    objetivos = motor.objetivos
    fn_eval = motor.fn_evaluar

    # --- Fail fast: verify GPU and estimate VRAM ---
    RandomCloud.verificar_gpu()
    RandomCloud.estimar_vram(config, entradas)

    estrategia = _elegir_estrategia(config, size(entradas, 1), size(entradas, 2))
    if estrategia === :error
        throw(ArgumentError(
            "Problem too large for GPU. Reduce tamano_nube or dataset size."
        ))
    end

    use_acts = config.activacion !== :sigmoid
    rng = MersenneTwister(config.semilla)
    t_inicio = time_ns()
    gpu_t_inicio = time_ns()

    nube = [RedNeuronal(config.topologia_inicial, rng) for _ in 1:config.tamano_nube]

    # Compute base activations
    n_capas_inicial = length(config.topologia_inicial) - 1
    acts_base = activaciones_por_capa(n_capas_inicial, config.activacion)

    N = config.tamano_nube
    resultados = Vector{ResultadoExploracion}(undef, N)

    # --- Exploration phase: CPU (topology changes per network make GPU batching complex) ---
    if use_acts
        for j in 1:N
            resultados[j] = _explorar_red(nube[j], entradas, objetivos,
                                          config.umbral_acierto, config.neuronas_eliminar,
                                          fn_eval, acts_base)
        end
    else
        for j in 1:N
            resultados[j] = _explorar_red(nube[j], entradas, objetivos,
                                          config.umbral_acierto, config.neuronas_eliminar, fn_eval)
        end
    end

    mejor_red = nothing
    mejor_precision = 0.0
    total_evaluaciones = 0
    total_reducciones = 0

    for res in resultados
        total_evaluaciones += res.evaluaciones
        total_reducciones += res.reducciones
        if res.mejor_red !== nothing && res.mejor_precision > mejor_precision
            mejor_red = res.mejor_red
            mejor_precision = res.mejor_precision
        end
    end

    # Track peak VRAM
    pico_vram = 0.0

    # --- Refinement phase: GPU ---
    if mejor_red !== nothing
        n_muestras = size(entradas, 2)
        acts_red = activaciones_por_capa(length(mejor_red.pesos), config.activacion)

        # Transfer data to GPU
        entradas_gpu = RandomCloud.a_gpu(entradas)
        objetivos_gpu = RandomCloud.a_gpu(objetivos)

        # Transfer best network weights to GPU as Float32
        pesos_gpu = [CUDA.cu(Float32.(mejor_red.pesos[l])) for l in eachindex(mejor_red.pesos)]
        biases_gpu = [CUDA.cu(Float32.(mejor_red.biases[l])) for l in eachindex(mejor_red.biases)]

        lr = Float32(config.tasa_aprendizaje)

        # Record VRAM after transfers
        pico_vram = max(pico_vram, Float64(CUDA.memory_status().total_bytes - CUDA.memory_status().free_bytes))

        if config.batch_size > 0 && n_muestras > config.batch_size
            # Mini-batch training on GPU
            indices = collect(1:n_muestras)
            rng_shuffle = MersenneTwister(config.semilla + 1)
            for _ in 1:config.epocas_refinamiento
                shuffle!(rng_shuffle, indices)
                @inbounds for start in 1:config.batch_size:n_muestras
                    fin = min(start + config.batch_size - 1, n_muestras)
                    batch_idx = indices[start:fin]
                    X_batch = entradas_gpu[:, batch_idx]
                    Y_batch = objetivos_gpu[:, batch_idx]
                    entrenar_batch_matmul!(pesos_gpu, biases_gpu, X_batch, Y_batch, lr, acts_red)
                end
            end
        else
            # Full-batch training on GPU
            entrenar_batch_matmul!(pesos_gpu, biases_gpu, entradas_gpu, objetivos_gpu, lr, acts_red)
        end

        # Record peak VRAM after training
        pico_vram = max(pico_vram, Float64(CUDA.memory_status().total_bytes - CUDA.memory_status().free_bytes))

        # Transfer weights back to CPU
        for l in eachindex(mejor_red.pesos)
            mejor_red.pesos[l] .= RandomCloud.de_gpu(pesos_gpu[l])
            mejor_red.biases[l] .= RandomCloud.de_gpu(biases_gpu[l])
        end

        # Re-evaluate on CPU with Float64 precision
        if use_acts
            mejor_precision = fn_eval(mejor_red, entradas, objetivos; acts=acts_red)
        else
            mejor_precision = fn_eval(mejor_red, entradas, objetivos)
        end

        es_exitoso = mejor_precision >= config.umbral_acierto

        gpu_t_fin = time_ns()
        gpu_tiempo_ms = (gpu_t_fin - gpu_t_inicio) / 1_000_000.0
        pico_vram_mb = pico_vram / (1024.0 * 1024.0)

        t_fin = time_ns()
        tiempo_ms = (t_fin - t_inicio) / 1_000_000.0

        if es_exitoso
            return InformeNube(mejor_red, mejor_precision, copy(mejor_red.topologia),
                               total_evaluaciones, total_reducciones, tiempo_ms, true,
                               gpu_tiempo_ms, pico_vram_mb)
        else
            return InformeNube(nothing, mejor_precision, nothing,
                               total_evaluaciones, total_reducciones, tiempo_ms, false,
                               gpu_tiempo_ms, pico_vram_mb)
        end
    end

    gpu_t_fin = time_ns()
    gpu_tiempo_ms = (gpu_t_fin - gpu_t_inicio) / 1_000_000.0
    pico_vram_mb = pico_vram / (1024.0 * 1024.0)

    t_fin = time_ns()
    tiempo_ms = (t_fin - t_inicio) / 1_000_000.0
    return InformeNube(nothing, mejor_precision, nothing,
                       total_evaluaciones, total_reducciones, tiempo_ms, false,
                       gpu_tiempo_ms, pico_vram_mb)
end
