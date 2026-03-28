# InformeNube — Resultados de una ejecución del método

struct InformeNube
    mejor_red::Union{RedNeuronal, Nothing}
    precision::Float64
    topologia_final::Union{Vector{Int}, Nothing}
    total_redes_evaluadas::Int
    total_reducciones::Int
    tiempo_ejecucion_ms::Float64
    exitoso::Bool
end
