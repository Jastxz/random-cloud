# Tests unitarios y PBT para evaluar

using Random
using RandomCloud: RedNeuronal, feedforward, evaluar

@testset "Evaluacion" begin

    @testset "Caso conocido: proporción correcta con red y datos conocidos" begin
        # Crear una red con semilla fija para resultados deterministas
        rng = MersenneTwister(42)
        red = RedNeuronal([2, 4, 2], rng)

        # Dataset XOR: 4 muestras, 2 entradas, 2 salidas (column-major)
        entradas = [0.0 0.0 1.0 1.0;
                    0.0 1.0 0.0 1.0]
        objetivos = [1.0 0.0 0.0 1.0;
                     0.0 1.0 1.0 0.0]

        # Calcular manualmente cuántas muestras clasifica correctamente
        aciertos_manual = 0
        n_muestras = size(entradas, 2)
        for k in 1:n_muestras
            salida = feedforward(red, entradas[:, k])
            if argmax(salida) == argmax(objetivos[:, k])
                aciertos_manual += 1
            end
        end
        proporcion_esperada = aciertos_manual / n_muestras

        resultado = evaluar(red, entradas, objetivos)

        @test resultado == proporcion_esperada
        @test 0.0 <= resultado <= 1.0
    end

    @testset "100% acierto: red entrenada en XOR" begin
        # Crear red y entrenarla en XOR hasta que clasifique todo correctamente
        rng = MersenneTwister(123)
        red = RedNeuronal([2, 8, 2], rng)

        entradas = [0.0 0.0 1.0 1.0;
                    0.0 1.0 0.0 1.0]
        objetivos = [1.0 0.0 0.0 1.0;
                     0.0 1.0 1.0 0.0]

        # Entrenar suficientes épocas para resolver XOR
        using RandomCloud: entrenar!
        for epoca in 1:5000
            for k in 1:size(entradas, 2)
                entrenar!(red, entradas[:, k], objetivos[:, k], 0.5)
            end
        end

        resultado = evaluar(red, entradas, objetivos)
        @test resultado == 1.0
    end

    @testset "0% acierto: todas las predicciones incorrectas" begin
        # Crear una red con semilla fija
        rng = MersenneTwister(42)
        red = RedNeuronal([2, 4, 2], rng)

        # Primero, determinar qué predice la red para cada entrada
        entradas = [0.0 0.0 1.0 1.0;
                    0.0 1.0 0.0 1.0]

        # Construir objetivos que sean opuestos a lo que la red predice
        n_muestras = size(entradas, 2)
        objetivos = zeros(2, n_muestras)
        for k in 1:n_muestras
            salida = feedforward(red, entradas[:, k])
            pred = argmax(salida)
            # Poner el 1.0 en la clase opuesta a la predicción
            clase_incorrecta = pred == 1 ? 2 : 1
            objetivos[clase_incorrecta, k] = 1.0
        end

        resultado = evaluar(red, entradas, objetivos)
        @test resultado == 0.0
    end

end
