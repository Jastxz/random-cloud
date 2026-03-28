# =============================================================================
# Ejemplo: Resolver el problema XOR con el Método de la Nube Aleatoria
# =============================================================================
#
# Ejecutar con:
#   julia --project=. examples/xor.jl
#
# Este script demuestra el uso completo de RandomCloud.jl para encontrar
# una arquitectura mínima de red neuronal que resuelva el problema XOR.
# =============================================================================

using RandomCloud

# --- Dataset XOR (formato column-major: cada columna es una muestra) ---
# 4 muestras, 2 entradas
entradas = [0.0 0.0 1.0 1.0;
            0.0 1.0 0.0 1.0]

# 4 muestras, 2 salidas (one-hot encoding)
# [1,0] = clase 0 (entradas iguales), [0,1] = clase 1 (entradas diferentes)
objetivos = [1.0 0.0 0.0 1.0;
             0.0 1.0 1.0 0.0]

# --- Configuración del método ---
config = ConfiguracionNube(
    tamano_nube = 50,              # 50 redes aleatorias en la nube
    topologia_inicial = [2, 8, 4, 2],  # 2 entradas → 8 ocultas → 4 ocultas → 2 salidas
    umbral_acierto = 0.5,         # Precisión mínima aceptable (50%)
    neuronas_eliminar = 1,        # Eliminar 1 neurona por reducción
    epocas_refinamiento = 2000,   # Épocas de backpropagation para refinamiento
    tasa_aprendizaje = 0.5,       # Tasa de aprendizaje para el refinamiento
    semilla = 42                  # Semilla para reproducibilidad
)

println("=" ^ 60)
println("  Método de la Nube Aleatoria — Problema XOR")
println("=" ^ 60)
println()
println("Configuración:")
println("  Tamaño de nube:       $(config.tamano_nube)")
println("  Topología inicial:    $(config.topologia_inicial)")
println("  Umbral de acierto:    $(config.umbral_acierto)")
println("  Neuronas a eliminar:  $(config.neuronas_eliminar)")
println("  Épocas refinamiento:  $(config.epocas_refinamiento)")
println("  Tasa de aprendizaje:  $(config.tasa_aprendizaje)")
println("  Semilla:              $(config.semilla)")
println()

# --- Ejecutar el método ---
println("Ejecutando...")
println()

motor = MotorNube(config, entradas, objetivos)
informe = ejecutar(motor)

# --- Imprimir resultados ---
println("-" ^ 60)
println("  Resultados")
println("-" ^ 60)
println()
println("  Exitoso:                $(informe.exitoso)")
println("  Precisión:              $(round(informe.precision * 100, digits=2))%")
println("  Redes evaluadas:        $(informe.total_redes_evaluadas)")
println("  Reducciones realizadas:  $(informe.total_reducciones)")
println("  Tiempo de ejecución:    $(round(informe.tiempo_ejecucion_ms, digits=2)) ms")

if informe.exitoso
    println()
    println("  Topología final:        $(informe.topologia_final)")

    # Calcular reducción de parámetros
    function contar_parametros(topologia::Vector{Int})
        total = 0
        for i in 1:(length(topologia) - 1)
            total += topologia[i + 1] * topologia[i]  # pesos
            total += topologia[i + 1]                  # biases
        end
        return total
    end

    params_original = contar_parametros(config.topologia_inicial)
    params_final = contar_parametros(informe.topologia_final)
    reduccion = (1.0 - params_final / params_original) * 100.0

    println()
    println("  Parámetros originales:  $params_original")
    println("  Parámetros finales:     $params_final")
    println("  Reducción:              $(round(reduccion, digits=2))%")
end

println()
println("=" ^ 60)
