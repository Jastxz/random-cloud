# InformeNube — Resultados de una ejecución del método

struct InformeNube
    mejor_red::Union{RedNeuronal, Nothing}
    precision::Float64
    topologia_final::Union{Vector{Int}, Nothing}
    total_redes_evaluadas::Int
    total_reducciones::Int
    tiempo_ejecucion_ms::Float64
    exitoso::Bool
    gpu_tiempo_ms::Float64
    pico_vram_mb::Float64
    # Fase 2
    fase2_resultado::Union{ResultadoFase2, Nothing}
    fase2_mejoro::Bool
end

# Backward-compatible: 9 args (GPU path, sin Fase 2)
function InformeNube(mejor_red, precision, topologia_final, total_redes_evaluadas,
                     total_reducciones, tiempo_ejecucion_ms, exitoso,
                     gpu_tiempo_ms::Float64, pico_vram_mb::Float64)
    InformeNube(mejor_red, precision, topologia_final, total_redes_evaluadas,
                total_reducciones, tiempo_ejecucion_ms, exitoso,
                gpu_tiempo_ms, pico_vram_mb, nothing, false)
end

# Backward-compatible: 7 args (legacy, sin GPU ni Fase 2)
function InformeNube(mejor_red, precision, topologia_final, total_redes_evaluadas,
                     total_reducciones, tiempo_ejecucion_ms, exitoso)
    InformeNube(mejor_red, precision, topologia_final, total_redes_evaluadas,
                total_reducciones, tiempo_ejecucion_ms, exitoso,
                0.0, 0.0, nothing, false)
end
