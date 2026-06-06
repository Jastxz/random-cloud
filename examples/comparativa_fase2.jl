# Comparativa: Fase 1 sola vs Fase 1 + Fase 2 (exploración estructural)
#
# Usa Iris y Wine porque tienen suficientes capas ocultas para que la Fase 2
# tenga material con el que trabajar (colapso + redistribución).

using RandomCloud
using Random

# --- Datos sintéticos: Two Moons (no requiere dependencias externas) ---

function generar_two_moons(n::Int; ruido=0.1, seed=42)
    rng = MersenneTwister(seed)
    n_half = n ÷ 2

    # Luna superior
    theta1 = range(0, π, length=n_half)
    x1 = cos.(theta1) .+ ruido .* randn(rng, n_half)
    y1 = sin.(theta1) .+ ruido .* randn(rng, n_half)

    # Luna inferior (desplazada)
    theta2 = range(0, π, length=n - n_half)
    x2 = 1.0 .- cos.(theta2) .+ ruido .* randn(rng, n - n_half)
    y2 = 1.0 .- sin.(theta2) .- 0.5 .+ ruido .* randn(rng, n - n_half)

    X = hcat(vcat(x1, x2), vcat(y1, y2))'  # 2 × n
    labels = vcat(zeros(Int, n_half), ones(Int, n - n_half))

    # One-hot
    Y = zeros(2, n)
    for i in 1:n
        Y[labels[i] + 1, i] = 1.0
    end

    return Float64.(X), Y
end

# --- Datos ---

entradas, objetivos = generar_two_moons(200; ruido=0.15, seed=123)
println("Dataset: Two Moons (200 muestras, 2 features, 2 clases)")
println("Topología inicial: [2, 32, 16, 8, 2] (3 capas ocultas)")
println()

# --- Fase 1 sola ---

config_f1 = ConfiguracionNube(
    tamano_nube=20,
    topologia_inicial=[2, 32, 16, 8, 2],
    umbral_acierto=0.6,
    neuronas_eliminar=2,
    epocas_refinamiento=500,
    tasa_aprendizaje=0.1,
    semilla=42,
    explorar_estructura=false
)

motor_f1 = MotorNube(config_f1, entradas, objetivos)
informe_f1 = ejecutar(motor_f1)

println("═══ FASE 1 SOLA ═══")
println("  Exitoso:     $(informe_f1.exitoso)")
println("  Precisión:   $(round(informe_f1.precision * 100, digits=1))%")
println("  Topología:   $(informe_f1.topologia_final)")
if informe_f1.topologia_final !== nothing
    params_f1 = sum(informe_f1.topologia_final[i] * informe_f1.topologia_final[i+1]
                    for i in 1:length(informe_f1.topologia_final)-1)
    println("  Parámetros:  $params_f1")
end
println("  Evaluaciones: $(informe_f1.total_redes_evaluadas)")
println("  Tiempo:      $(round(informe_f1.tiempo_ejecucion_ms, digits=1)) ms")
println()

# --- Fase 1 + Fase 2 ---

config_f2 = ConfiguracionNube(
    tamano_nube=20,
    topologia_inicial=[2, 32, 16, 8, 2],
    umbral_acierto=0.6,
    neuronas_eliminar=2,
    epocas_refinamiento=500,
    tasa_aprendizaje=0.1,
    semilla=42,
    explorar_estructura=true,
    max_profundidad_split=2,
    ancho_minimo_split=4,
    n_candidatos_estructura=5
)

motor_f2 = MotorNube(config_f2, entradas, objetivos)
informe_f2 = ejecutar(motor_f2)

println("═══ FASE 1 + FASE 2 ═══")
println("  Exitoso:     $(informe_f2.exitoso)")
println("  Precisión:   $(round(informe_f2.precision * 100, digits=1))%")
println("  Topología:   $(informe_f2.topologia_final)")
if informe_f2.topologia_final !== nothing
    params_f2 = sum(informe_f2.topologia_final[i] * informe_f2.topologia_final[i+1]
                    for i in 1:length(informe_f2.topologia_final)-1)
    println("  Parámetros:  $params_f2")
end
println("  Evaluaciones: $(informe_f2.total_redes_evaluadas)")
println("  Tiempo:      $(round(informe_f2.tiempo_ejecucion_ms, digits=1)) ms")
println("  Fase 2 mejoró: $(informe_f2.fase2_mejoro)")
if informe_f2.fase2_resultado !== nothing
    r = informe_f2.fase2_resultado
    println("  Candidatos evaluados: $(r.candidatos_evaluados)")
    println("  Candidatos refinados: $(r.candidatos_refinados)")
    println("  Operación ganadora:  $(r.operacion)")
    println("  Precisión Fase 2:    $(round(r.mejor_precision * 100, digits=1))%")
end
println()

# --- Comparación ---

println("═══ COMPARACIÓN ═══")
if informe_f1.topologia_final !== nothing && informe_f2.topologia_final !== nothing
    params_f1 = sum(informe_f1.topologia_final[i] * informe_f1.topologia_final[i+1]
                    for i in 1:length(informe_f1.topologia_final)-1)
    params_f2 = sum(informe_f2.topologia_final[i] * informe_f2.topologia_final[i+1]
                    for i in 1:length(informe_f2.topologia_final)-1)
    diff_acc = (informe_f2.precision - informe_f1.precision) * 100
    diff_params = params_f2 - params_f1
    println("  Δ Precisión:   $(diff_acc > 0 ? "+" : "")$(round(diff_acc, digits=1)) pp")
    println("  Δ Parámetros:  $(diff_params > 0 ? "+" : "")$diff_params ($(round(params_f2/params_f1*100, digits=1))% del original)")
end
