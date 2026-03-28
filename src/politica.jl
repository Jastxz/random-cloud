# PoliticaEliminacion — Sistema de políticas de reducción topológica

"""
Tipo abstracto que define la interfaz para políticas de eliminación de neuronas.
Nuevas políticas se crean definiendo un subtipo y un método `siguiente_reduccion`.
"""
abstract type PoliticaEliminacion end

"""
Política concreta que elimina neuronas desde la última capa oculta hacia la primera.
"""
struct PoliticaSecuencial <: PoliticaEliminacion end

"""
    siguiente_reduccion(::PoliticaSecuencial, topologia::Vector{Int}, n::Int) → Union{Vector{Int}, Nothing}

Retorna una nueva topología con `n` neuronas menos en la última capa oculta que tenga
neuronas disponibles (> 0), o `nothing` si no es posible reducir.

La capa de entrada (índice 1) y la capa de salida (último índice) no se modifican.
"""
function siguiente_reduccion(::PoliticaSecuencial, topologia::Vector{Int}, n::Int)
    # Si todas las capas ocultas tienen 0 neuronas → nothing
    capas_ocultas = @view topologia[2:end-1]
    if all(c -> c == 0, capas_ocultas)
        return nothing
    end

    # Copiar la topología
    nueva = copy(topologia)

    # Buscar la última capa oculta con neuronas > 0 (desde length-1 hasta 2)
    idx = nothing
    for i in (length(nueva) - 1):-1:2
        if nueva[i] > 0
            idx = i
            break
        end
    end

    # Restar n neuronas (mínimo 0)
    nueva[idx] = max(0, nueva[idx] - n)

    # Si tras la reducción todas las capas ocultas quedan en 0 → nothing
    if all(c -> c == 0, @view nueva[2:end-1])
        return nothing
    end

    return nueva
end
