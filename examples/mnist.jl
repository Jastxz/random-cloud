# =============================================================================
# Experimento: Método de la Nube Aleatoria sobre MNIST
# =============================================================================
#
# Ejecutar con:
#   julia --project=. examples/mnist.jl
#
# Usa un subconjunto de MNIST (1000 muestras por defecto) para evaluar
# la capacidad del método de encontrar arquitecturas mínimas en un
# problema de clasificación real (dígitos manuscritos 0-9).
#
# MNIST: 28×28 píxeles = 784 entradas, 10 clases de salida.
# =============================================================================

using RandomCloud
using MLDatasets: MNIST
using Random

# --- Parámetros del experimento ---
const N_MUESTRAS = 60_000     # dataset completo de entrenamiento
const SEMILLA = 42

println("=" ^ 65)
println("  Método de la Nube Aleatoria — MNIST ($N_MUESTRAS muestras)")
println("=" ^ 65)
println()

# --- Cargar y preparar datos ---
println("Cargando MNIST...")
dataset = MNIST(Float64, :train)

# Seleccionar subconjunto aleatorio
rng = MersenneTwister(SEMILLA)
indices = randperm(rng, length(dataset))[1:N_MUESTRAS]

# Aplanar imágenes 28×28 → vector de 784, formato column-major (cada columna = muestra)
raw_features = dataset[indices].features   # 28×28×N_MUESTRAS
entradas = reshape(raw_features, 784, N_MUESTRAS)

# One-hot encoding de labels (0-9 → vector de 10)
labels = dataset[indices].targets
objetivos = zeros(Float64, 10, N_MUESTRAS)
for k in 1:N_MUESTRAS
    objetivos[labels[k] + 1, k] = 1.0   # labels van de 0 a 9
end

println("  Entradas:  $(size(entradas)) — $(size(entradas,1)) píxeles × $(size(entradas,2)) muestras")
println("  Objetivos: $(size(objetivos)) — 10 clases × $(size(objetivos,2)) muestras")
println("  Clases:    0-9 (dígitos manuscritos)")
println()


# --- Función auxiliar para contar parámetros ---
function contar_parametros(topologia::Vector{Int})
    total = 0
    for i in 1:(length(topologia) - 1)
        total += topologia[i + 1] * topologia[i] + topologia[i + 1]
    end
    return total
end

# --- Ejecutar experimento ---
# Topología: 784 entradas → capas ocultas → 10 salidas
# Con 784 entradas, incluso una capa oculta pequeña genera muchos parámetros.
# [784, 32, 16, 10] = 25,898 parámetros

config = ConfiguracionNube(
    tamano_nube = 100,                    # nube grande para 784 dimensiones
    topologia_inicial = [784, 32, 16, 10], # arquitectura compacta para MNIST
    umbral_acierto = 0.15,                # ~15% (azar = 10% para 10 clases)
    neuronas_eliminar = 2,                # reducción de 2 en 2 para ir más rápido
    epocas_refinamiento = 50,             # pocas épocas para el subconjunto
    tasa_aprendizaje = 0.1,
    semilla = SEMILLA
)

params_inicial = contar_parametros(config.topologia_inicial)

println("Configuración:")
println("  Tamaño de nube:       $(config.tamano_nube)")
println("  Topología inicial:    $(config.topologia_inicial)")
println("  Parámetros iniciales: $params_inicial")
println("  Umbral de acierto:    $(config.umbral_acierto) (azar ≈ 10%)")
println("  Neuronas a eliminar:  $(config.neuronas_eliminar)")
println("  Épocas refinamiento:  $(config.epocas_refinamiento)")
println("  Tasa de aprendizaje:  $(config.tasa_aprendizaje)")
println("  Semilla:              $(config.semilla)")
println()

println("Ejecutando el Método de la Nube Aleatoria...")
println("  (esto puede tardar unos minutos)")
println()

motor = MotorNube(config, entradas, objetivos)
informe = ejecutar(motor)

# --- Resultados ---
println("-" ^ 65)
println("  Resultados")
println("-" ^ 65)
println()
println("  Exitoso:                 $(informe.exitoso)")
println("  Precisión:               $(round(informe.precision * 100, digits=2))%")
println("  Redes evaluadas:         $(informe.total_redes_evaluadas)")
println("  Reducciones realizadas:  $(informe.total_reducciones)")
println("  Tiempo de ejecución:     $(round(informe.tiempo_ejecucion_ms / 1000, digits=2)) s")

if informe.exitoso
    params_final = contar_parametros(informe.topologia_final)
    reduccion = (1.0 - params_final / params_inicial) * 100.0

    println()
    println("  Topología final:         $(informe.topologia_final)")
    println("  Parámetros finales:      $params_final")
    println("  Reducción:               $(round(reduccion, digits=2))%")
else
    println()
    println("  No se encontró red viable con umbral $(config.umbral_acierto).")
    println("  Sugerencias: aumentar tamano_nube, bajar umbral_acierto,")
    println("  o usar topología inicial más grande.")
end

println()
println("=" ^ 65)
