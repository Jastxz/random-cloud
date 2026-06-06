# Fase2Estructura — Exploración estructural: colapso y redistribución de capas
#
# Operaciones que transforman la topología de una red existente
# reutilizando sus pesos (nunca genera pesos aleatorios nuevos).

# ─────────────────────────────────────────────────────────────────────────────
# Tipos
# ─────────────────────────────────────────────────────────────────────────────

"""
Tipo de operación estructural que produjo un candidato.
"""
@enum TipoOperacion begin
    OP_COLAPSO          # Eliminación de capa con composición matricial
    OP_SPLIT_1          # Redistribución profundidad 1 (1 capa → 2)
    OP_SPLIT_2          # Redistribución profundidad 2 (1 capa → 3)
    OP_COMBINADA        # Colapso + split en la misma red
end

"""
Candidato estructural: una topología alternativa con sus pesos derivados.
"""
struct CandidatoEstructural
    red::RedNeuronal
    operacion::TipoOperacion
    descripcion::String
end

"""
Resultado de la exploración estructural (Fase 2).
"""
struct ResultadoFase2
    mejor_red::Union{RedNeuronal, Nothing}
    mejor_precision::Float64
    topologia_final::Union{Vector{Int}, Nothing}
    operacion::Union{TipoOperacion, Nothing}
    candidatos_evaluados::Int
    candidatos_refinados::Int
end

# ─────────────────────────────────────────────────────────────────────────────
# Colapso de capas
# ─────────────────────────────────────────────────────────────────────────────

"""
    colapsar_capa(red::RedNeuronal, idx_capa::Int) → RedNeuronal

Elimina la capa oculta en posición `idx_capa` (1-indexed sobre capas ocultas,
es decir idx_capa=1 es la primera capa oculta).

Los pesos se componen: W_nueva = W[idx+1] × W[idx]
Los biases se absorben: b_nueva = W[idx+1] × b[idx] + b[idx+1]
"""
function colapsar_capa(red::RedNeuronal, idx_capa::Int)
    n_capas_pesos = length(red.pesos)
    # idx_capa refiere a la capa oculta (pesos[idx_capa] conecta capa anterior → esta)
    # Para eliminarla, componemos pesos[idx_capa] con pesos[idx_capa+1]

    @assert 1 <= idx_capa <= n_capas_pesos - 1 "idx_capa debe estar en [1, $(n_capas_pesos-1)]"

    nuevos_pesos = Matrix{Float64}[]
    nuevos_biases = Vector{Float64}[]
    nueva_topologia = Int[]

    # Topología: eliminamos la capa oculta idx_capa+1 en la topología original
    # red.topologia = [entrada, oculta1, oculta2, ..., salida]
    # La capa oculta idx_capa corresponde a topologia[idx_capa+1]
    for i in eachindex(red.topologia)
        if i != idx_capa + 1
            push!(nueva_topologia, red.topologia[i])
        end
    end

    for i in 1:n_capas_pesos
        if i == idx_capa
            # Componer esta capa con la siguiente
            W_compuesta = red.pesos[i + 1] * red.pesos[i]
            b_compuesta = red.pesos[i + 1] * red.biases[i] .+ red.biases[i + 1]
            push!(nuevos_pesos, W_compuesta)
            push!(nuevos_biases, b_compuesta)
        elseif i == idx_capa + 1
            # Ya absorbida en la composición anterior — saltar
            continue
        else
            push!(nuevos_pesos, copy(red.pesos[i]))
            push!(nuevos_biases, copy(red.biases[i]))
        end
    end

    return RedNeuronal(nueva_topologia, nuevos_pesos, nuevos_biases)
end

# ─────────────────────────────────────────────────────────────────────────────
# Redistribución de capas (split)
# ─────────────────────────────────────────────────────────────────────────────

"""
    redistribuir_capa(red::RedNeuronal, idx_capa::Int, anchos::Vector{Int};
                      invertir::Bool=false) → RedNeuronal

Divide la capa oculta `idx_capa` en múltiples sub-capas con los anchos especificados.

- `anchos`: vector con el ancho de cada sub-capa. sum(anchos) debe ser ≤ ancho original.
- `invertir`: si true, invierte el orden de las filas al particionar.

Los pesos de la primera sub-capa son las filas correspondientes de W[idx_capa].
Las conexiones entre sub-capas son matrices identidad.
La conexión de la última sub-capa con la capa siguiente usa las columnas correspondientes
de W[idx_capa+1].
"""
function redistribuir_capa(red::RedNeuronal, idx_capa::Int, anchos::Vector{Int};
                           invertir::Bool=false)
    n_capas_pesos = length(red.pesos)
    @assert 1 <= idx_capa <= n_capas_pesos - 1 "idx_capa debe estar en [1, $(n_capas_pesos-1)]"

    ancho_original = red.topologia[idx_capa + 1]
    total_anchos = sum(anchos)
    @assert total_anchos <= ancho_original "sum(anchos)=$total_anchos > ancho_original=$ancho_original"
    @assert all(a -> a >= 1, anchos) "todos los anchos deben ser ≥ 1"

    n_subcapas = length(anchos)
    W_entrada = red.pesos[idx_capa]      # (ancho_original × n_in)
    b_entrada = red.biases[idx_capa]     # (ancho_original,)
    W_salida = red.pesos[idx_capa + 1]   # (n_out × ancho_original)

    # Determinar el orden de filas
    if invertir
        filas_orden = collect(ancho_original:-1:1)
    else
        filas_orden = collect(1:ancho_original)
    end

    # Particionar las filas según anchos
    rangos = Vector{UnitRange{Int}}(undef, n_subcapas)
    offset = 0
    for s in 1:n_subcapas
        rangos[s] = (offset + 1):(offset + anchos[s])
        offset += anchos[s]
    end

    # Construir nuevos pesos y biases
    nuevos_pesos = Matrix{Float64}[]
    nuevos_biases = Vector{Float64}[]
    nueva_topologia = Int[]

    # Copiar capas anteriores a idx_capa sin cambios
    for i in 1:idx_capa - 1
        push!(nueva_topologia, red.topologia[i])
        push!(nuevos_pesos, copy(red.pesos[i]))
        push!(nuevos_biases, copy(red.biases[i]))
    end
    # Añadir la capa de entrada (previa a la primera sub-capa)
    push!(nueva_topologia, red.topologia[idx_capa])

    # Primera sub-capa: filas seleccionadas de W_entrada
    filas_sub1 = filas_orden[rangos[1]]
    push!(nuevos_pesos, W_entrada[filas_sub1, :])
    push!(nuevos_biases, b_entrada[filas_sub1])
    push!(nueva_topologia, anchos[1])

    # Sub-capas intermedias: conexión identidad
    for s in 2:n_subcapas
        filas_sub = filas_orden[rangos[s]]
        # Conexión entre sub-capa s-1 y sub-capa s: identidad (anchos[s] × anchos[s-1])
        # Usamos una identidad rectangular si los anchos difieren
        W_inter = zeros(anchos[s], anchos[s - 1])
        dim_min = min(anchos[s], anchos[s - 1])
        for j in 1:dim_min
            W_inter[j, j] = 1.0
        end
        b_inter = zeros(anchos[s])
        push!(nuevos_pesos, W_inter)
        push!(nuevos_biases, b_inter)
        push!(nueva_topologia, anchos[s])
    end

    # Conexión de la última sub-capa con la capa siguiente:
    # Tomamos las columnas correspondientes de W_salida
    # Las columnas corresponden a las neuronas de la capa original que ahora
    # están en la última sub-capa
    filas_ultima = filas_orden[rangos[end]]
    W_salida_new = W_salida[:, filas_ultima]
    b_salida_new = copy(red.biases[idx_capa + 1])
    push!(nuevos_pesos, W_salida_new)
    push!(nuevos_biases, b_salida_new)

    # Copiar capas posteriores a idx_capa+1 sin cambios
    for i in idx_capa + 2:n_capas_pesos
        push!(nuevos_pesos, copy(red.pesos[i]))
        push!(nuevos_biases, copy(red.biases[i]))
    end
    # Completar topología con las capas posteriores
    for i in (idx_capa + 2):length(red.topologia)
        push!(nueva_topologia, red.topologia[i])
    end

    return RedNeuronal(nueva_topologia, nuevos_pesos, nuevos_biases)
end

# ─────────────────────────────────────────────────────────────────────────────
# Generación de candidatos
# ─────────────────────────────────────────────────────────────────────────────

"""
    generar_divisiones_split1(ancho::Int, ancho_minimo::Int) → Vector{Vector{Int}}

Genera todas las divisiones válidas de una capa de ancho `ancho` en 2 sub-capas,
donde cada sub-capa tiene al menos `ancho_minimo` neuronas.
"""
function generar_divisiones_split1(ancho::Int, ancho_minimo::Int)
    divisiones = Vector{Int}[]
    for a in ancho_minimo:(ancho - ancho_minimo)
        b = ancho - a
        if b >= ancho_minimo
            push!(divisiones, [a, b])
        end
    end
    return divisiones
end

"""
    generar_divisiones_split2(ancho::Int, ancho_minimo::Int) → Vector{Vector{Int}}

Genera todas las divisiones válidas de una capa de ancho `ancho` en 3 sub-capas,
donde cada sub-capa tiene al menos `ancho_minimo` neuronas.
"""
function generar_divisiones_split2(ancho::Int, ancho_minimo::Int)
    divisiones = Vector{Int}[]
    for a in ancho_minimo:(ancho - 2 * ancho_minimo)
        for b in ancho_minimo:(ancho - a - ancho_minimo)
            c = ancho - a - b
            if c >= ancho_minimo
                push!(divisiones, [a, b, c])
            end
        end
    end
    return divisiones
end

"""
    generar_candidatos(red::RedNeuronal, max_prof_split::Int, ancho_min::Int) → Vector{CandidatoEstructural}

Genera todos los candidatos estructurales a partir de una red dada:
- Colapsos de cada capa oculta.
- Splits de profundidad 1 y 2 (según max_prof_split) con variantes normal/invertida.
"""
function generar_candidatos(red::RedNeuronal, max_prof_split::Int, ancho_min::Int)
    candidatos = CandidatoEstructural[]
    n_capas_ocultas = length(red.pesos) - 1  # capas que se pueden colapsar/redistribuir

    # --- Colapsos ---
    for i in 1:n_capas_ocultas
        red_colapsada = colapsar_capa(red, i)
        desc = "colapso capa $(i) (ancho=$(red.topologia[i+1]))"
        push!(candidatos, CandidatoEstructural(red_colapsada, OP_COLAPSO, desc))
    end

    # --- Splits profundidad 1 ---
    for i in 1:n_capas_ocultas
        ancho = red.topologia[i + 1]
        if ancho < 2 * ancho_min
            continue  # no se puede dividir respetando ancho mínimo
        end
        divisiones = generar_divisiones_split1(ancho, ancho_min)
        for div in divisiones
            # Variante normal
            red_split = redistribuir_capa(red, i, div; invertir=false)
            desc = "split1 capa $(i) ($(ancho)→$(div)) normal"
            push!(candidatos, CandidatoEstructural(red_split, OP_SPLIT_1, desc))
            # Variante invertida
            red_split_inv = redistribuir_capa(red, i, div; invertir=true)
            desc_inv = "split1 capa $(i) ($(ancho)→$(div)) invertido"
            push!(candidatos, CandidatoEstructural(red_split_inv, OP_SPLIT_1, desc_inv))
        end
    end

    # --- Splits profundidad 2 ---
    if max_prof_split >= 2
        for i in 1:n_capas_ocultas
            ancho = red.topologia[i + 1]
            if ancho < 3 * ancho_min
                continue
            end
            divisiones = generar_divisiones_split2(ancho, ancho_min)
            for div in divisiones
                # Variante normal
                red_split = redistribuir_capa(red, i, div; invertir=false)
                desc = "split2 capa $(i) ($(ancho)→$(div)) normal"
                push!(candidatos, CandidatoEstructural(red_split, OP_SPLIT_2, desc))
                # Variante invertida
                red_split_inv = redistribuir_capa(red, i, div; invertir=true)
                desc_inv = "split2 capa $(i) ($(ancho)→$(div)) invertido"
                push!(candidatos, CandidatoEstructural(red_split_inv, OP_SPLIT_2, desc_inv))
            end
        end
    end

    return candidatos
end

# ─────────────────────────────────────────────────────────────────────────────
# Orquestador de Fase 2
# ─────────────────────────────────────────────────────────────────────────────

"""
    explorar_estructura(red_fase1::RedNeuronal, entradas::Matrix{Float64},
                        objetivos::Matrix{Float64}, config::ConfiguracionNube,
                        fn_eval::Function) → ResultadoFase2

Ejecuta la Fase 2: genera candidatos estructurales a partir de la red ganadora
de Fase 1, los evalúa sin entrenamiento, selecciona los K mejores,
los refina con backprop, y retorna el mejor resultado.
"""
function explorar_estructura(red_fase1::RedNeuronal, entradas::Matrix{Float64},
                             objetivos::Matrix{Float64}, config::ConfiguracionNube,
                             fn_eval::Function)
    # Generar candidatos
    candidatos = generar_candidatos(red_fase1, config.max_profundidad_split,
                                    config.ancho_minimo_split)

    if isempty(candidatos)
        return ResultadoFase2(nothing, 0.0, nothing, nothing, 0, 0)
    end

    # Evaluar todos sin entrenamiento
    use_acts = config.activacion !== :sigmoid
    precisiones = Vector{Float64}(undef, length(candidatos))

    for (j, cand) in enumerate(candidatos)
        n_capas_cand = length(cand.red.pesos)
        if use_acts
            acts = activaciones_por_capa(n_capas_cand, config.activacion)
            precisiones[j] = fn_eval(cand.red, entradas, objetivos; acts=acts)
        else
            precisiones[j] = fn_eval(cand.red, entradas, objetivos)
        end
    end

    # Seleccionar top-K
    K = min(config.n_candidatos_estructura, length(candidatos))
    orden = sortperm(precisiones, rev=true)
    top_k_indices = orden[1:K]

    # Refinar cada top-K con backprop
    mejor_precision = 0.0
    mejor_red = nothing
    mejor_operacion = nothing

    for idx in top_k_indices
        cand = candidatos[idx]
        red_copia = _copiar_red(cand.red)
        n_capas_cand = length(red_copia.pesos)
        acts_red = activaciones_por_capa(n_capas_cand, config.activacion)

        # Refinamiento con backprop
        n_muestras = size(entradas, 2)
        bufs = EntrenarBuffers(red_copia.topologia)

        if config.batch_size > 0 && n_muestras > config.batch_size
            indices = collect(1:n_muestras)
            rng_shuffle = MersenneTwister(config.semilla + 2)
            for _ in 1:config.epocas_refinamiento
                shuffle!(rng_shuffle, indices)
                @inbounds for start in 1:config.batch_size:n_muestras
                    fin = min(start + config.batch_size - 1, n_muestras)
                    batch_idx = @view indices[start:fin]
                    if use_acts
                        entrenar_batch!(red_copia, entradas, objetivos, batch_idx,
                                        config.tasa_aprendizaje, bufs, acts_red)
                    else
                        for k in batch_idx
                            entrenar!(red_copia, @view(entradas[:, k]), @view(objetivos[:, k]),
                                      config.tasa_aprendizaje, bufs)
                        end
                    end
                end
            end
        else
            if use_acts
                for _ in 1:config.epocas_refinamiento
                    @inbounds for k in 1:n_muestras
                        entrenar!(red_copia, @view(entradas[:, k]), @view(objetivos[:, k]),
                                  config.tasa_aprendizaje, bufs, acts_red)
                    end
                end
            else
                for _ in 1:config.epocas_refinamiento
                    @inbounds for k in 1:n_muestras
                        entrenar!(red_copia, @view(entradas[:, k]), @view(objetivos[:, k]),
                                  config.tasa_aprendizaje, bufs)
                    end
                end
            end
        end

        # Evaluar tras refinamiento
        if use_acts
            p = fn_eval(red_copia, entradas, objetivos; acts=acts_red)
        else
            p = fn_eval(red_copia, entradas, objetivos)
        end

        if p > mejor_precision
            mejor_precision = p
            mejor_red = red_copia
            mejor_operacion = cand.operacion
        end
    end

    topologia_final = mejor_red !== nothing ? copy(mejor_red.topologia) : nothing

    return ResultadoFase2(mejor_red, mejor_precision, topologia_final,
                          mejor_operacion, length(candidatos), K)
end

# ─────────────────────────────────────────────────────────────────────────────
# Utilidades internas
# ─────────────────────────────────────────────────────────────────────────────

"""Copia profunda de una RedNeuronal."""
function _copiar_red(red::RedNeuronal)
    RedNeuronal(
        copy(red.topologia),
        [copy(w) for w in red.pesos],
        [copy(b) for b in red.biases]
    )
end
