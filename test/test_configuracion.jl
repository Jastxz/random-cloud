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

# Feature: gpu-batched-cloud, Property 7: ConfiguracionNube backward compatibility
# **Validates: Requirements 5.1, 5.2**
@testset "PBT Property 7: ConfiguracionNube backward compatibility" begin

    hidden_gen = Data.Vectors(Data.Integers(1, 30); min_size=1, max_size=3)
    topo_gen = @composed function gen_topo(
        n_in = Data.Integers(1, 20),
        hidden = hidden_gen,
        n_out = Data.Integers(1, 10)
    )
        return vcat([n_in], hidden, [n_out])
    end

    activacion_gen = Data.SampledFrom([:sigmoid, :relu, :identidad])

    umbral_gen = filter(x -> !isnan(x) && !isinf(x), Data.Floats{Float64}(minimum=0.0, maximum=1.0))
    lr_gen = filter(x -> !isnan(x) && !isinf(x), Data.Floats{Float64}(minimum=0.001, maximum=1.0))

    @check max_examples=100 function prop_backward_compat(
        tamano_nube = Data.Integers(1, 50),
        topologia = topo_gen,
        umbral = umbral_gen,
        neuronas_eliminar = Data.Integers(1, 10),
        epocas = Data.Integers(1, 5000),
        lr = lr_gen,
        semilla = Data.Integers(1, 100000),
        activacion = activacion_gen,
        batch_size = Data.Integers(0, 100)
    )
        # Construct WITHOUT specifying gpu
        config = ConfiguracionNube(
            tamano_nube=tamano_nube,
            topologia_inicial=topologia,
            umbral_acierto=umbral,
            neuronas_eliminar=neuronas_eliminar,
            epocas_refinamiento=epocas,
            tasa_aprendizaje=lr,
            semilla=semilla,
            activacion=activacion,
            batch_size=batch_size
        )

        # gpu must default to false
        config.gpu == false || return false

        # All other fields must match provided values
        config.tamano_nube == tamano_nube || return false
        config.topologia_inicial == topologia || return false
        config.umbral_acierto == umbral || return false
        config.neuronas_eliminar == neuronas_eliminar || return false
        config.epocas_refinamiento == epocas || return false
        config.tasa_aprendizaje == lr || return false
        config.semilla == semilla || return false
        config.activacion == activacion || return false
        config.batch_size == batch_size || return false

        return true
    end
end

# Feature: gpu-batched-cloud, Task 8.4: Unit tests for ConfiguracionNube and InformeNube modifications
using RandomCloud: MotorNube, InformeNube, GPU_AVAILABLE

@testset "Unit tests: ConfiguracionNube and InformeNube modifications" begin

    @testset "gpu=true without CUDA raises ArgumentError" begin
        @test GPU_AVAILABLE[] == false
        @test_throws ArgumentError ConfiguracionNube(gpu=true)
        try
            ConfiguracionNube(gpu=true)
        catch e
            @test occursin("CUDA.jl must be added to use gpu=true", e.msg)
        end
    end

    @testset "MotorNube 2-arg constructor (config, X, Y)" begin
        config = ConfiguracionNube(topologia_inicial=[2, 3, 1])
        X = Float64[0 0 1 1; 0 1 0 1]
        Y = Float64[0 1 1 0]
        motor = MotorNube(config, X, Y)
        @test motor.config === config
        @test motor.entradas === X
        @test motor.objetivos === Y
    end

    @testset "MotorNube 3-arg constructor (config, X, Y, fn)" begin
        config = ConfiguracionNube(topologia_inicial=[2, 3, 1])
        X = Float64[0 0 1 1; 0 1 0 1]
        Y = Float64[0 1 1 0]
        fn_custom = (red, ent, obj; acts=nothing) -> 0.5
        motor = MotorNube(config, X, Y, fn_custom)
        @test motor.config === config
        @test motor.entradas === X
        @test motor.objetivos === Y
        @test motor.fn_evaluar === fn_custom
    end

    @testset "InformeNube backward-compatible constructor defaults new fields to 0.0" begin
        informe = InformeNube(nothing, 0.0, nothing, 0, 0, 0.0, false)
        @test informe.gpu_tiempo_ms == 0.0
        @test informe.pico_vram_mb == 0.0
        @test informe.mejor_red === nothing
        @test informe.precision == 0.0
        @test informe.exitoso == false
    end

end
