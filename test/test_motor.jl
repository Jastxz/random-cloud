# Tests unitarios y PBT para MotorNube

using RandomCloud: MotorNube, ejecutar, InformeNube, ConfiguracionNube

# Dataset XOR (column-major, 2 salidas para clasificación)
const XOR_ENTRADAS = [0.0 0.0 1.0 1.0;
                      0.0 1.0 0.0 1.0]
const XOR_OBJETIVOS = [1.0 0.0 0.0 1.0;
                       0.0 1.0 1.0 0.0]

@testset "MotorNube" begin

    @testset "XOR exitoso con topología [2, 8, 4, 2]" begin
        config = ConfiguracionNube(
            tamano_nube = 50,
            topologia_inicial = [2, 8, 4, 2],
            umbral_acierto = 0.5,
            semilla = 42,
            epocas_refinamiento = 2000,
            tasa_aprendizaje = 0.5
        )
        motor = MotorNube(config, XOR_ENTRADAS, XOR_OBJETIVOS)
        informe = ejecutar(motor)

        @test informe.exitoso == true
        @test informe.precision > 0.0
        @test informe.total_redes_evaluadas > 0
        @test informe.tiempo_ejecucion_ms > 0.0
        @test informe.mejor_red !== nothing
        @test informe.topologia_final !== nothing
        @test informe.total_reducciones >= 0
    end

    @testset "Caso de fallo: umbral_acierto = 1.0 con topología mínima" begin
        config = ConfiguracionNube(
            tamano_nube = 2,
            topologia_inicial = [2, 2, 2],
            umbral_acierto = 1.0,
            semilla = 42,
            epocas_refinamiento = 10
        )
        motor = MotorNube(config, XOR_ENTRADAS, XOR_OBJETIVOS)
        informe = ejecutar(motor)

        @test informe.exitoso == false
        @test informe.mejor_red === nothing
        @test informe.topologia_final === nothing
    end

end

# Feature: random-cloud-julia, Property 12: Ejecución determinista
# **Validates: Requirements 8.1, 6.3**
using Supposition
using Supposition: Data

@testset "PBT Propiedad 12: Ejecución determinista" begin

    seed_gen = Data.Integers(1, 10000)

    @check max_examples=100 function prop_ejecucion_determinista(semilla = seed_gen)
        config = ConfiguracionNube(
            tamano_nube = 5,
            topologia_inicial = [2, 4, 2],
            umbral_acierto = 0.5,
            epocas_refinamiento = 50,
            tasa_aprendizaje = 0.1,
            semilla = semilla
        )

        motor1 = MotorNube(config, XOR_ENTRADAS, XOR_OBJETIVOS)
        informe1 = ejecutar(motor1)

        motor2 = MotorNube(config, XOR_ENTRADAS, XOR_OBJETIVOS)
        informe2 = ejecutar(motor2)

        informe1.topologia_final == informe2.topologia_final || return false
        informe1.precision == informe2.precision || return false
        informe1.exitoso == informe2.exitoso || return false

        return true
    end
end

# Feature: random-cloud-julia, Property 14: Exitoso → precisión ≥ umbral
# **Validates: Requirements 6.5**

@testset "PBT Propiedad 14: Ejecución exitosa implica precisión ≥ umbral" begin

    seed_gen = Data.Integers(1, 10000)
    umbral_gen = Data.Floats{Float64}(minimum=0.25, maximum=0.75)

    @check max_examples=100 function prop_exitoso_implica_precision_ge_umbral(semilla = seed_gen, umbral = umbral_gen)
        config = ConfiguracionNube(
            tamano_nube = 10,
            topologia_inicial = [2, 4, 2],
            umbral_acierto = umbral,
            epocas_refinamiento = 100,
            tasa_aprendizaje = 0.3,
            semilla = semilla
        )

        motor = MotorNube(config, XOR_ENTRADAS, XOR_OBJETIVOS)
        informe = ejecutar(motor)

        # Si exitoso == true, la precisión debe ser ≥ umbral_acierto
        # Si exitoso == false, la propiedad es vacuamente verdadera
        if informe.exitoso
            informe.precision >= umbral || return false
        end

        return true
    end
end

# Feature: random-cloud-julia, Property 15: Fallido → campos nothing
# **Validates: Requirements 5.4, 6.9**

@testset "PBT Propiedad 15: Ejecución fallida implica campos nothing" begin

    seed_gen = Data.Integers(1, 10000)
    umbral_gen = Data.Floats{Float64}(minimum=0.99, maximum=1.0)

    @check max_examples=100 function prop_fallido_implica_nothing(semilla = seed_gen, umbral = umbral_gen)
        config = ConfiguracionNube(
            tamano_nube = 2,
            topologia_inicial = [2, 2, 2],
            umbral_acierto = umbral,
            epocas_refinamiento = 10,
            tasa_aprendizaje = 0.1,
            semilla = semilla
        )

        motor = MotorNube(config, XOR_ENTRADAS, XOR_OBJETIVOS)
        informe = ejecutar(motor)

        # Si exitoso == false, mejor_red y topologia_final deben ser nothing
        # Si exitoso == true, la propiedad es vacuamente verdadera
        if !informe.exitoso
            informe.mejor_red === nothing || return false
            informe.topologia_final === nothing || return false
        end

        return true
    end
end

# Feature: random-cloud-julia, Property 16: Metadatos consistentes
# **Validates: Requirements 6.10, 6.11**

@testset "PBT Propiedad 16: Metadatos del informe son consistentes" begin

    seed_gen = Data.Integers(1, 10000)
    tamano_gen = Data.Integers(2, 10)

    @check max_examples=100 function prop_metadatos_consistentes(semilla = seed_gen, tamano = tamano_gen)
        config = ConfiguracionNube(
            tamano_nube = tamano,
            topologia_inicial = [2, 4, 2],
            umbral_acierto = 0.5,
            epocas_refinamiento = 10,
            tasa_aprendizaje = 0.1,
            semilla = semilla
        )

        motor = MotorNube(config, XOR_ENTRADAS, XOR_OBJETIVOS)
        informe = ejecutar(motor)

        informe.tiempo_ejecucion_ms > 0 || return false
        informe.total_redes_evaluadas >= tamano || return false

        return true
    end
end
