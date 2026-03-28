# MotorNube — Orquestador del Método de la Nube Aleatoria

mutable struct MotorNube
    config::ConfiguracionNube
    entradas::Matrix{Float64}
    objetivos::Matrix{Float64}
end

function ejecutar(motor::MotorNube)
    config = motor.config
    entradas = motor.entradas
    objetivos = motor.objetivos

    # 1. Inicializar RNG local con semilla de la configuración
    rng = MersenneTwister(config.semilla)

    # 2. Registrar tiempo de inicio
    t_inicio = time_ns()

    # 3. Inicializar mejor red, mejor precisión y contadores
    mejor_red = nothing
    mejor_precision = 0.0
    total_evaluaciones = 0
    total_reducciones = 0

    politica = PoliticaSecuencial()

    # 4. Generar nube de N redes con topología inicial
    nube = [RedNeuronal(config.topologia_inicial, rng) for _ in 1:config.tamano_nube]

    # 5. Para cada red en la nube, explorar todas las sub-topologías
    for red_j in nube
        r_actual = red_j
        t_actual = copy(r_actual.topologia)

        while true
            p = evaluar(r_actual, entradas, objetivos)
            total_evaluaciones += 1

            if p > config.umbral_acierto && p > mejor_precision
                mejor_red = r_actual
                mejor_precision = p
            end

            t_nueva = siguiente_reduccion(politica, t_actual, config.neuronas_eliminar)

            if t_nueva === nothing
                break
            end

            r_actual = reconstruir(r_actual, t_nueva)
            total_reducciones += 1
            t_actual = copy(r_actual.topologia)
        end
    end

    # 6. Si se encontró red viable: refinar UNA SOLA VEZ con backpropagation
    if mejor_red !== nothing
        n_muestras = size(entradas, 2)

        # Pre-alocar buffers para entrenamiento (evita allocations en el hot loop)
        bufs = EntrenarBuffers(mejor_red.topologia)

        for _ in 1:config.epocas_refinamiento
            @inbounds for k in 1:n_muestras
                entrenar!(mejor_red, @view(entradas[:, k]), @view(objetivos[:, k]),
                          config.tasa_aprendizaje, bufs)
            end
        end

        mejor_precision = evaluar(mejor_red, entradas, objetivos)
        es_exitoso = mejor_precision >= config.umbral_acierto

        t_fin = time_ns()
        tiempo_ms = (t_fin - t_inicio) / 1_000_000.0

        if es_exitoso
            return InformeNube(
                mejor_red, mejor_precision, copy(mejor_red.topologia),
                total_evaluaciones, total_reducciones, tiempo_ms, true
            )
        else
            return InformeNube(
                nothing, mejor_precision, nothing,
                total_evaluaciones, total_reducciones, tiempo_ms, false
            )
        end
    end

    # 7. Ninguna red superó el umbral
    t_fin = time_ns()
    tiempo_ms = (t_fin - t_inicio) / 1_000_000.0

    return InformeNube(
        nothing, mejor_precision, nothing,
        total_evaluaciones, total_reducciones, tiempo_ms, false
    )
end
