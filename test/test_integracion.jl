# Test de integración: flujo completo del método
# Valida: Requisitos 5.1, 5.2, 5.3, 5.4, 10.4

@testset "Integración — Flujo completo" begin

    # Dataset XOR (column-major)
    entradas = [0.0 0.0 1.0 1.0;
                0.0 1.0 0.0 1.0]
    objetivos = [1.0 0.0 0.0 1.0;
                 0.0 1.0 1.0 0.0]

    @testset "Caso exitoso — XOR con parámetros favorables" begin
        config = ConfiguracionNube(
            tamano_nube = 50,
            topologia_inicial = [2, 8, 4, 2],
            umbral_acierto = 0.5,
            neuronas_eliminar = 1,
            epocas_refinamiento = 2000,
            tasa_aprendizaje = 0.5,
            semilla = 42
        )

        motor = MotorNube(config, entradas, objetivos)
        informe = ejecutar(motor)

        # Req 5.3: ejecución exitosa → exitoso == true
        @test informe.exitoso == true

        # Req 5.3: mejor_red encontrada (es una RedNeuronal)
        @test informe.mejor_red !== nothing
        @test informe.mejor_red isa RedNeuronal

        # Req 5.3: topologia_final presente
        @test informe.topologia_final !== nothing
        @test informe.topologia_final isa Vector{Int}

        # Req 5.1: precision en rango válido
        @test informe.precision > 0.0
        @test informe.precision <= 1.0

        # Req 5.1: contadores válidos
        @test informe.total_redes_evaluadas > 0
        @test informe.total_reducciones >= 0

        # Req 5.1: tiempo de ejecución registrado
        @test informe.tiempo_ejecucion_ms > 0.0
    end

    @testset "Caso fallido — parámetros imposibles" begin
        config = ConfiguracionNube(
            tamano_nube = 2,
            topologia_inicial = [2, 2, 2],
            umbral_acierto = 1.0,
            neuronas_eliminar = 1,
            epocas_refinamiento = 10,
            tasa_aprendizaje = 0.1,
            semilla = 42
        )

        motor = MotorNube(config, entradas, objetivos)
        informe = ejecutar(motor)

        # Req 5.4: ejecución fallida → exitoso == false
        @test informe.exitoso == false

        # Req 5.4: mejor_red y topologia_final son nothing
        @test informe.mejor_red === nothing
        @test informe.topologia_final === nothing

        # Contadores y tiempo siguen siendo válidos
        @test informe.total_redes_evaluadas > 0
        @test informe.tiempo_ejecucion_ms > 0.0
    end

    @testset "Inmutabilidad de InformeNube" begin
        # Req 5.2: InformeNube es inmutable
        config = ConfiguracionNube(
            tamano_nube = 5,
            topologia_inicial = [2, 4, 2],
            umbral_acierto = 0.5,
            semilla = 42
        )
        motor = MotorNube(config, entradas, objetivos)
        informe = ejecutar(motor)

        @test_throws ErrorException informe.exitoso = !informe.exitoso
    end
end
