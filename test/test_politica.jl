# Tests unitarios y PBT para PoliticaSecuencial

using RandomCloud: PoliticaEliminacion, PoliticaSecuencial, siguiente_reduccion

@testset "PoliticaSecuencial" begin

    @testset "PoliticaSecuencial es subtipo de PoliticaEliminacion" begin
        @test PoliticaSecuencial <: PoliticaEliminacion
        @test PoliticaSecuencial() isa PoliticaEliminacion
    end

    @testset "Reducción normal — última capa oculta" begin
        pol = PoliticaSecuencial()
        resultado = siguiente_reduccion(pol, [2, 4, 3, 1], 1)
        @test resultado == [2, 4, 2, 1]
    end

    @testset "Reducciones sucesivas" begin
        pol = PoliticaSecuencial()
        # Primera reducción: reduce última capa oculta (3 → 2)
        r1 = siguiente_reduccion(pol, [2, 4, 3, 1], 1)
        @test r1 == [2, 4, 2, 1]
        # Segunda: 2 → 1
        r2 = siguiente_reduccion(pol, r1, 1)
        @test r2 == [2, 4, 1, 1]
        # Tercera: 1 → 0, pero still has hidden layer 4
        r3 = siguiente_reduccion(pol, r2, 1)
        @test r3 == [2, 4, 0, 1]
        # Cuarta: ahora reduce la primera capa oculta (4 → 3)
        r4 = siguiente_reduccion(pol, r3, 1)
        @test r4 == [2, 3, 0, 1]
    end

    @testset "Retorna nothing cuando todas las capas ocultas son 0" begin
        pol = PoliticaSecuencial()
        @test siguiente_reduccion(pol, [2, 0, 0, 1], 1) === nothing
    end

    @testset "Retorna nothing cuando reducción dejaría todas las capas ocultas en 0" begin
        pol = PoliticaSecuencial()
        # Solo queda 1 neurona en una capa oculta; restar 1 las deja todas en 0
        @test siguiente_reduccion(pol, [2, 1, 0, 1], 1) === nothing
        # Caso con n > neuronas restantes
        @test siguiente_reduccion(pol, [2, 0, 1, 1], 2) === nothing
    end

    @testset "Topología original no se modifica" begin
        pol = PoliticaSecuencial()
        original = [2, 4, 3, 1]
        copia = copy(original)
        siguiente_reduccion(pol, original, 1)
        @test original == copia
    end

    @testset "Capa de entrada y salida permanecen iguales" begin
        pol = PoliticaSecuencial()
        topologia = [5, 8, 6, 3]
        resultado = siguiente_reduccion(pol, topologia, 2)
        @test resultado[1] == topologia[1]
        @test resultado[end] == topologia[end]
    end

end

# Feature: random-cloud-julia, Property 9: Propiedades de reducción válida
# **Validates: Requirements 3.3, 3.6, 3.7, 3.8**
using Supposition
using Supposition: Data

@testset "PBT Propiedad 9: Propiedades de reducción válida" begin

    # Generator for topologies with hidden layers >= 2 so that reducing by 1
    # won't make ALL hidden layers 0 (ensuring non-nothing results most of the time)
    hidden_gen = Data.Vectors(Data.Integers(2, 10); min_size=1, max_size=4)
    topo_gen = @composed function valid_topology_p9(
        input_size = Data.Integers(1, 10),
        hidden = hidden_gen,
        output_size = Data.Integers(1, 10)
    )
        return vcat([input_size], hidden, [output_size])
    end

    @check max_examples=100 function prop_reduccion_valida(topo = topo_gen)
        pol = PoliticaSecuencial()
        n = 1  # Use n=1 to maximize non-nothing results

        # Save a copy of the original topology before calling siguiente_reduccion
        topo_original = copy(topo)

        resultado = siguiente_reduccion(pol, topo, n)

        # (d) The original topology must NOT be modified regardless of result
        topo == topo_original || return false

        # If result is nothing, that's valid — skip remaining assertions
        if resultado === nothing
            return true
        end

        # (a) Input layer (first) must be same as original
        resultado[1] == topo_original[1] || return false

        # (b) Output layer (last) must be same as original
        resultado[end] == topo_original[end] || return false

        # (c) Exactly one hidden layer was reduced (differs from original)
        capas_ocultas_orig = topo_original[2:end-1]
        capas_ocultas_res = resultado[2:end-1]
        length(capas_ocultas_res) == length(capas_ocultas_orig) || return false

        diferencias = sum(capas_ocultas_res[i] != capas_ocultas_orig[i] for i in 1:length(capas_ocultas_orig))
        diferencias == 1 || return false

        return true
    end
end

# Feature: random-cloud-julia, Property 10: Reducción imposible → nothing
# **Validates: Requirements 3.4, 3.5**

@testset "PBT Propiedad 10: siguiente_reduccion retorna nothing cuando no hay reducción posible" begin

    # Case 1: All hidden layers are 0 → [input, 0, 0, ..., 0, output]
    num_hidden_gen_c1 = Data.Integers(1, 4)
    topo_all_zeros_gen = @composed function all_zeros_topology(
        input_size = Data.Integers(1, 10),
        num_hidden = num_hidden_gen_c1,
        output_size = Data.Integers(1, 10)
    )
        return vcat([input_size], zeros(Int, num_hidden), [output_size])
    end

    n_gen = Data.Integers(1, 5)

    @check max_examples=100 function prop_all_hidden_zero_returns_nothing(
        topo = topo_all_zeros_gen,
        n = n_gen
    )
        pol = PoliticaSecuencial()
        resultado = siguiente_reduccion(pol, topo, n)
        resultado === nothing || return false
        return true
    end

    # Case 2: Reduction would make all hidden layers 0
    # One hidden layer has value 1, all others 0, and n >= that value
    num_hidden_gen_c2 = Data.Integers(1, 4)
    topo_reduction_impossible_gen = @composed function reduction_impossible_topology(
        input_size = Data.Integers(1, 10),
        num_hidden = num_hidden_gen_c2,
        output_size = Data.Integers(1, 10),
        active_idx = Data.Integers(1, 4)
    )
        hidden = zeros(Int, num_hidden)
        # Place value 1 in one hidden layer (clamp index to valid range)
        idx = clamp(active_idx, 1, num_hidden)
        hidden[idx] = 1
        return vcat([input_size], hidden, [output_size])
    end

    # n >= 1 ensures the single non-zero hidden layer (value=1) will be reduced to 0
    n_ge1_gen = Data.Integers(1, 5)

    @check max_examples=100 function prop_reduction_would_zero_all_returns_nothing(
        topo = topo_reduction_impossible_gen,
        n = n_ge1_gen
    )
        pol = PoliticaSecuencial()
        resultado = siguiente_reduccion(pol, topo, n)
        resultado === nothing || return false
        return true
    end
end
