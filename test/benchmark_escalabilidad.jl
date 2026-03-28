# =============================================================================
# Benchmark de Escalabilidad — RandomCloud.jl
# =============================================================================
#
# Ejecutar con:
#   julia --project=. test/benchmark_escalabilidad.jl
#
# Si BenchmarkTools no está disponible, instalar con:
#   julia --project=. -e 'import Pkg; Pkg.add("BenchmarkTools")'
#
# Compara el rendimiento del Método de la Nube Aleatoria con distintos
# tamaños de nube para medir cómo escala el tiempo de ejecución.
#
# Requisitos: 10.5
# =============================================================================

using RandomCloud
using BenchmarkTools

# --- Dataset XOR (formato column-major) ---
const ENTRADAS_XOR = [0.0 0.0 1.0 1.0;
                      0.0 1.0 0.0 1.0]

const OBJETIVOS_XOR = [1.0 0.0 0.0 1.0;
                       0.0 1.0 1.0 0.0]

# --- Configuración base (pequeña para benchmarks rápidos) ---
const TOPOLOGIA = [2, 4, 2]
const UMBRAL = 0.5
const EPOCAS = 100
const LR = 0.1
const SEMILLA = 42

# --- Tamaños de nube a comparar ---
const TAMANOS = [5, 10, 20, 50]

# --- Función auxiliar para ejecutar con un tamaño dado ---
function ejecutar_con_tamano(n::Int)
    config = ConfiguracionNube(
        tamano_nube = n,
        topologia_inicial = TOPOLOGIA,
        umbral_acierto = UMBRAL,
        epocas_refinamiento = EPOCAS,
        tasa_aprendizaje = LR,
        semilla = SEMILLA
    )
    motor = MotorNube(config, ENTRADAS_XOR, OBJETIVOS_XOR)
    return ejecutar(motor)
end

# --- Ejecutar benchmarks ---
println("=" ^ 60)
println("  Benchmark de Escalabilidad — RandomCloud.jl")
println("=" ^ 60)
println()
println("Configuración fija:")
println("  Topología:           $TOPOLOGIA")
println("  Umbral:              $UMBRAL")
println("  Épocas refinamiento: $EPOCAS")
println("  Tasa aprendizaje:    $LR")
println("  Semilla:             $SEMILLA")
println()
println("-" ^ 60)

for n in TAMANOS
    println()
    println("  Tamaño de nube: $n")
    println()
    resultado = @benchmark ejecutar_con_tamano($n) samples=5 evals=1
    display(resultado)
    println()
    println("-" ^ 60)
end

println()
println("Benchmark completado.")
