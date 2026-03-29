# MotorNube — Orquestador del Método de la Nube Aleatoria

const FnEvaluar = Function

mutable struct MotorNube
    config::ConfiguracionNube
    entradas::Matrix{Float64}
    objetivos::Matrix{Float64}
    fn_evaluar::FnEvaluar
end

# Constructor por defecto: usa evaluar (clasificación)
MotorNube(config, entradas, objetivos) = MotorNube(config, entradas, objetivos, evaluar)

struct ResultadoExploracion
    mejor_red::Union{RedNeuronal, Nothing}
    mejor_precision::Float64
    evaluaciones::Int
    reducciones::Int
end

# Explorar sub-topologías de una red (función pura)
function _explorar_red(red::RedNeuronal, entradas::Matrix{Float64},
                       objetivos::Matrix{Float64}, umbral::Float64,
                       neuronas_eliminar::Int, fn_eval::Function,
                       acts_base::Vector{Symbol})
    politica = PoliticaSecuencial()
    mejor_red = nothing
    mejor_precision = 0.0
    evaluaciones = 0
    reducciones = 0
    r_actual = red
    t_actual = copy(r_actual.topologia)

    while true
        # Recalcular activaciones para la topología actual
        n_capas = length(r_actual.pesos)
        acts = activaciones_por_capa(n_capas, acts_base[1])  # usa la misma activación base
        p = fn_eval(r_actual, entradas, objetivos; acts=acts)
        evaluaciones += 1
        if p > umbral && p > mejor_precision
            mejor_red = r_actual
            mejor_precision = p
        end
        t_nueva = siguiente_reduccion(politica, t_actual, neuronas_eliminar)
        if t_nueva === nothing
            break
        end
        r_actual = reconstruir(r_actual, t_nueva)
        reducciones += 1
        t_actual = copy(r_actual.topologia)
    end
    return ResultadoExploracion(mejor_red, mejor_precision, evaluaciones, reducciones)
end

# Versión legacy sin activaciones (backward compatible)
function _explorar_red(red::RedNeuronal, entradas::Matrix{Float64},
                       objetivos::Matrix{Float64}, umbral::Float64,
                       neuronas_eliminar::Int, fn_eval::Function)
    politica = PoliticaSecuencial()
    mejor_red = nothing
    mejor_precision = 0.0
    evaluaciones = 0
    reducciones = 0
    r_actual = red
    t_actual = copy(r_actual.topologia)

    while true
        p = fn_eval(r_actual, entradas, objetivos)
        evaluaciones += 1
        if p > umbral && p > mejor_precision
            mejor_red = r_actual
            mejor_precision = p
        end
        t_nueva = siguiente_reduccion(politica, t_actual, neuronas_eliminar)
        if t_nueva === nothing
            break
        end
        r_actual = reconstruir(r_actual, t_nueva)
        reducciones += 1
        t_actual = copy(r_actual.topologia)
    end
    return ResultadoExploracion(mejor_red, mejor_precision, evaluaciones, reducciones)
end

function ejecutar(motor::MotorNube)
    config = motor.config
    entradas = motor.entradas
    objetivos = motor.objetivos
    fn_eval = motor.fn_evaluar
    use_acts = config.activacion !== :sigmoid

    rng = MersenneTwister(config.semilla)
    t_inicio = time_ns()

    nube = [RedNeuronal(config.topologia_inicial, rng) for _ in 1:config.tamano_nube]

    # Calcular activaciones base
    n_capas_inicial = length(config.topologia_inicial) - 1
    acts_base = activaciones_por_capa(n_capas_inicial, config.activacion)

    N = config.tamano_nube
    resultados = Vector{ResultadoExploracion}(undef, N)

    if use_acts
        if Threads.nthreads() > 1
            Threads.@threads for j in 1:N
                resultados[j] = _explorar_red(nube[j], entradas, objetivos,
                                              config.umbral_acierto, config.neuronas_eliminar,
                                              fn_eval, acts_base)
            end
        else
            for j in 1:N
                resultados[j] = _explorar_red(nube[j], entradas, objetivos,
                                              config.umbral_acierto, config.neuronas_eliminar,
                                              fn_eval, acts_base)
            end
        end
    else
        if Threads.nthreads() > 1
            Threads.@threads for j in 1:N
                resultados[j] = _explorar_red(nube[j], entradas, objetivos,
                                              config.umbral_acierto, config.neuronas_eliminar, fn_eval)
            end
        else
            for j in 1:N
                resultados[j] = _explorar_red(nube[j], entradas, objetivos,
                                              config.umbral_acierto, config.neuronas_eliminar, fn_eval)
            end
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

    if mejor_red !== nothing
        n_muestras = size(entradas, 2)
        bufs = EntrenarBuffers(mejor_red.topologia)
        acts_red = activaciones_por_capa(length(mejor_red.pesos), config.activacion)

        if config.batch_size > 0 && n_muestras > config.batch_size
            # Mini-batch training
            indices = collect(1:n_muestras)
            rng_shuffle = MersenneTwister(config.semilla + 1)
            for _ in 1:config.epocas_refinamiento
                shuffle!(rng_shuffle, indices)
                @inbounds for start in 1:config.batch_size:n_muestras
                    fin = min(start + config.batch_size - 1, n_muestras)
                    batch_idx = @view indices[start:fin]
                    if use_acts
                        entrenar_batch!(mejor_red, entradas, objetivos, batch_idx,
                                        config.tasa_aprendizaje, bufs, acts_red)
                    else
                        # Fallback sample-by-sample para sigmoid (legacy path)
                        for k in batch_idx
                            entrenar!(mejor_red, @view(entradas[:, k]), @view(objetivos[:, k]),
                                      config.tasa_aprendizaje, bufs)
                        end
                    end
                end
            end
        else
            # Sample-by-sample (original behavior)
            if use_acts
                for _ in 1:config.epocas_refinamiento
                    @inbounds for k in 1:n_muestras
                        entrenar!(mejor_red, @view(entradas[:, k]), @view(objetivos[:, k]),
                                  config.tasa_aprendizaje, bufs, acts_red)
                    end
                end
            else
                for _ in 1:config.epocas_refinamiento
                    @inbounds for k in 1:n_muestras
                        entrenar!(mejor_red, @view(entradas[:, k]), @view(objetivos[:, k]),
                                  config.tasa_aprendizaje, bufs)
                    end
                end
            end
        end

        if use_acts
            mejor_precision = fn_eval(mejor_red, entradas, objetivos; acts=acts_red)
        else
            mejor_precision = fn_eval(mejor_red, entradas, objetivos)
        end
        es_exitoso = mejor_precision >= config.umbral_acierto

        t_fin = time_ns()
        tiempo_ms = (t_fin - t_inicio) / 1_000_000.0

        if es_exitoso
            return InformeNube(mejor_red, mejor_precision, copy(mejor_red.topologia),
                               total_evaluaciones, total_reducciones, tiempo_ms, true)
        else
            return InformeNube(nothing, mejor_precision, nothing,
                               total_evaluaciones, total_reducciones, tiempo_ms, false)
        end
    end

    t_fin = time_ns()
    tiempo_ms = (t_fin - t_inicio) / 1_000_000.0
    return InformeNube(nothing, mejor_precision, nothing,
                       total_evaluaciones, total_reducciones, tiempo_ms, false)
end
