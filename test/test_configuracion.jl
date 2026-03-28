# Tests unitarios y PBT para ConfiguracionNube

@testset "ConfiguracionNube" begin

    @testset "Valores por defecto" begin
        config = ConfiguracionNube()
        @test config.tamano_nube == 10
        @test config.topologia_inicial == [2, 4, 1]
        @test config.umbral_acierto == 0.5
        @test config.neuronas_eliminar == 1
        @test config.epocas_refinamiento == 1000
        @test config.tasa_aprendizaje == 0.1
        @test config.semilla == 42
    end

    @testset "Validación tamano_nube < 1" begin
        @test_throws ArgumentError ConfiguracionNube(tamano_nube=0)
        @test_throws ArgumentError ConfiguracionNube(tamano_nube=-5)
        try
            ConfiguracionNube(tamano_nube=0)
        catch e
            @test occursin("tamano_nube debe ser ≥ 1", e.msg)
            @test occursin("0", e.msg)
        end
    end

    @testset "Validación topologia_inicial < 3 capas" begin
        @test_throws ArgumentError ConfiguracionNube(topologia_inicial=[2, 1])
        @test_throws ArgumentError ConfiguracionNube(topologia_inicial=[5])
        try
            ConfiguracionNube(topologia_inicial=[2, 1])
        catch e
            @test occursin("al menos 3 capas", e.msg)
        end
    end

    @testset "Validación umbral_acierto fuera de [0.0, 1.0]" begin
        @test_throws ArgumentError ConfiguracionNube(umbral_acierto=-0.1)
        @test_throws ArgumentError ConfiguracionNube(umbral_acierto=1.1)
        try
            ConfiguracionNube(umbral_acierto=-0.1)
        catch e
            @test occursin("umbral_acierto debe estar en [0.0, 1.0]", e.msg)
            @test occursin("-0.1", e.msg)
        end
        try
            ConfiguracionNube(umbral_acierto=1.1)
        catch e
            @test occursin("1.1", e.msg)
        end
    end

    @testset "Validación neuronas_eliminar < 1" begin
        @test_throws ArgumentError ConfiguracionNube(neuronas_eliminar=0)
        @test_throws ArgumentError ConfiguracionNube(neuronas_eliminar=-3)
        try
            ConfiguracionNube(neuronas_eliminar=0)
        catch e
            @test occursin("neuronas_eliminar debe ser ≥ 1", e.msg)
            @test occursin("0", e.msg)
        end
    end

    @testset "Validación capas ocultas < 1 neurona" begin
        @test_throws ArgumentError ConfiguracionNube(topologia_inicial=[2, 0, 1])
        @test_throws ArgumentError ConfiguracionNube(topologia_inicial=[2, 3, 0, 1])
        try
            ConfiguracionNube(topologia_inicial=[2, 0, 1])
        catch e
            @test occursin("las capas ocultas requieren al menos 1 neurona", e.msg)
        end
    end

    @testset "Copia defensiva de topologia_inicial" begin
        topo = [2, 4, 1]
        config = ConfiguracionNube(topologia_inicial=topo)
        topo[2] = 99
        @test config.topologia_inicial == [2, 4, 1]
        @test config.topologia_inicial[2] == 4
    end

end

# Feature: random-cloud-julia, Property 3: Copia defensiva de topología
# **Validates: Requirements 1.8, 2.3**
using Supposition
using Supposition: Data

@testset "PBT Propiedad 3: Copia defensiva de topología" begin

    # Generator for valid topology: [input, hidden..., output] with ≥ 3 elements, hidden ≥ 1
    hidden_gen = Data.Vectors(Data.Integers(1, 10); min_size=1, max_size=4)
    topo_gen = @composed function valid_topology(
        input_size = Data.Integers(1, 10),
        hidden = hidden_gen,
        output_size = Data.Integers(1, 10)
    )
        return vcat([input_size], hidden, [output_size])
    end

    @check max_examples=100 function prop_copia_defensiva_configuracion(topo = topo_gen)
        # Save a copy of the original topology before construction
        topo_original = copy(topo)

        # Build ConfiguracionNube with the topology
        config = ConfiguracionNube(topologia_inicial=topo)

        # Verify the stored topology matches the original
        config.topologia_inicial == topo_original || return false

        # Mutate the original vector
        topo[2] = topo[2] + 100

        # Verify the stored topology was NOT affected by the mutation
        config.topologia_inicial == topo_original && config.topologia_inicial[2] == topo_original[2]
    end
end
