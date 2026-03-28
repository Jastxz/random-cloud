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

        # Bucle de exploración de sub-topologías
        while true
            # i. Evaluar red actual sin entrenamiento
            p = evaluar(r_actual, entradas, objetivos)
            total_evaluaciones += 1

            # ii. Si supera umbral Y es mejor que la mejor encontrada → guardar
            if p > config.umbral_acierto && p > mejor_precision
                mejor_red = r_actual
                mejor_precision = p
            end

            # iii. Calcular siguiente reducción
            t_nueva = siguiente_reduccion(politica, t_actual, config.neuronas_eliminar)

            # iv. Si no hay más reducciones posibles → siguiente red
            if t_nueva === nothing
                break
            end

            # v. Reconstruir red con topología reducida
            r_actual = reconstruir(r_actual, t_nueva)
            total_reducciones += 1

            # vi. Actualizar topología actual (use the reconstructed network's actual
            #     topology, which may have fewer layers if 0-neuron layers were collapsed)
            t_actual = copy(r_actual.topologia)
        end
    end

    # 6. Si se encontró red viable: refinar UNA SOLA VEZ con backpropagation
    if mejor_red !== nothing
        n_muestras = size(entradas, 2)
        for _epoca in 1:config.epocas_refinamiento
            for k in 1:n_muestras
                entrenar!(mejor_red, entradas[:, k], objetivos[:, k], config.tasa_aprendizaje)
            end
        end

        # Evaluar precisión final tras refinamiento
        mejor_precision = evaluar(mejor_red, entradas, objetivos)

        # Verificar si la precisión post-refinamiento aún supera el umbral
        es_exitoso = mejor_precision >= config.umbral_acierto

        # Registrar tiempo final
        t_fin = time_ns()
        tiempo_ms = (t_fin - t_inicio) / 1_000_000.0

        if es_exitoso
            return InformeNube(
                mejor_red,
                mejor_precision,
                copy(mejor_red.topologia),
                total_evaluaciones,
                total_reducciones,
                tiempo_ms,
                true
            )
        else
            return InformeNube(
                nothing,
                mejor_precision,
                nothing,
                total_evaluaciones,
                total_reducciones,
                tiempo_ms,
                false
            )
        end
    end

    # 7. Ninguna red superó el umbral
    t_fin = time_ns()
    tiempo_ms = (t_fin - t_inicio) / 1_000_000.0

    return InformeNube(
        nothing,
        mejor_precision,
        nothing,
        total_evaluaciones,
        total_reducciones,
        tiempo_ms,
        false
    )
end
