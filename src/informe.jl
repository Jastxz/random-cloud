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
end

# Backward-compatible constructor: accepts original 7 arguments, defaults new fields to 0.0
function InformeNube(mejor_red, precision, topologia_final, total_redes_evaluadas, total_reducciones, tiempo_ejecucion_ms, exitoso)
    InformeNube(mejor_red, precision, topologia_final, total_redes_evaluadas, total_reducciones, tiempo_ejecucion_ms, exitoso, 0.0, 0.0)
end
