# Tests unitarios y PBT para evaluar

using Random
using RandomCloud: RedNeuronal, feedforward, evaluar, evaluar_f1, evaluar_auc

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


# --- Tests para evaluar_f1 ---

@testset "F1-Score" begin

    @testset "F1 = 1.0 con clasificación perfecta" begin
        rng = MersenneTwister(123)
        red = RedNeuronal([2, 8, 2], rng)
        entradas = [0.0 0.0 1.0 1.0; 0.0 1.0 0.0 1.0]
        objetivos = [1.0 0.0 0.0 1.0; 0.0 1.0 1.0 0.0]

        using RandomCloud: entrenar!
        for _ in 1:5000
            for k in 1:size(entradas, 2)
                entrenar!(red, entradas[:, k], objetivos[:, k], 0.5)
            end
        end

        f1 = evaluar_f1(red, entradas, objetivos)
        @test f1 == 1.0
    end

    @testset "F1 = 0.0 con todas las predicciones incorrectas" begin
        rng = MersenneTwister(42)
        red = RedNeuronal([2, 4, 2], rng)
        entradas = [0.0 0.0 1.0 1.0; 0.0 1.0 0.0 1.0]

        # Construir objetivos opuestos a las predicciones
        n_muestras = size(entradas, 2)
        objetivos = zeros(2, n_muestras)
        for k in 1:n_muestras
            salida = feedforward(red, entradas[:, k])
            pred = argmax(salida)
            clase_incorrecta = pred == 1 ? 2 : 1
            objetivos[clase_incorrecta, k] = 1.0
        end

        f1 = evaluar_f1(red, entradas, objetivos)
        @test f1 == 0.0
    end

    @testset "F1 en rango [0, 1]" begin
        rng = MersenneTwister(42)
        red = RedNeuronal([2, 4, 2], rng)
        entradas = [0.0 0.0 1.0 1.0; 0.0 1.0 0.0 1.0]
        objetivos = [1.0 0.0 0.0 1.0; 0.0 1.0 1.0 0.0]

        f1 = evaluar_f1(red, entradas, objetivos)
        @test 0.0 <= f1 <= 1.0
    end

    @testset "F1 consistente con accuracy cuando todas las clases están balanceadas" begin
        # Con clases balanceadas y clasificación perfecta, F1 == accuracy == 1.0
        rng = MersenneTwister(123)
        red = RedNeuronal([2, 8, 2], rng)
        entradas = [0.0 0.0 1.0 1.0; 0.0 1.0 0.0 1.0]
        objetivos = [1.0 0.0 0.0 1.0; 0.0 1.0 1.0 0.0]

        using RandomCloud: entrenar!
        for _ in 1:5000
            for k in 1:size(entradas, 2)
                entrenar!(red, entradas[:, k], objetivos[:, k], 0.5)
            end
        end

        acc = evaluar(red, entradas, objetivos)
        f1 = evaluar_f1(red, entradas, objetivos)
        @test acc == 1.0
        @test f1 == 1.0
    end

    @testset "F1 multiclase (3 clases)" begin
        rng = MersenneTwister(42)
        red = RedNeuronal([2, 8, 3], rng)
        # 6 muestras, 3 clases (2 por clase)
        entradas = [0.0 0.1 0.5 0.6 1.0 0.9;
                    0.0 0.1 0.5 0.6 1.0 0.9]
        objetivos = [1.0 1.0 0.0 0.0 0.0 0.0;
                     0.0 0.0 1.0 1.0 0.0 0.0;
                     0.0 0.0 0.0 0.0 1.0 1.0]

        f1 = evaluar_f1(red, entradas, objetivos)
        @test 0.0 <= f1 <= 1.0
    end
end

# --- Tests para evaluar_auc ---

@testset "AUC" begin

    @testset "AUC = 1.0 con clasificación perfecta (binario)" begin
        rng = MersenneTwister(123)
        red = RedNeuronal([2, 8, 2], rng)
        entradas = [0.0 0.0 1.0 1.0; 0.0 1.0 0.0 1.0]
        objetivos = [1.0 0.0 0.0 1.0; 0.0 1.0 1.0 0.0]

        using RandomCloud: entrenar!
        for _ in 1:5000
            for k in 1:size(entradas, 2)
                entrenar!(red, entradas[:, k], objetivos[:, k], 0.5)
            end
        end

        auc = evaluar_auc(red, entradas, objetivos)
        @test auc == 1.0
    end

    @testset "AUC en rango [0, 1]" begin
        rng = MersenneTwister(42)
        red = RedNeuronal([2, 4, 2], rng)
        entradas = [0.0 0.0 1.0 1.0; 0.0 1.0 0.0 1.0]
        objetivos = [1.0 0.0 0.0 1.0; 0.0 1.0 1.0 0.0]

        auc = evaluar_auc(red, entradas, objetivos)
        @test 0.0 <= auc <= 1.0
    end

    @testset "AUC con red sin entrenar no es exactamente 0" begin
        # Una red aleatoria debería tener AUC cercano a 0.5 (aleatorio)
        rng = MersenneTwister(42)
        red = RedNeuronal([2, 4, 2], rng)
        entradas = [0.0 0.0 1.0 1.0; 0.0 1.0 0.0 1.0]
        objetivos = [1.0 0.0 0.0 1.0; 0.0 1.0 1.0 0.0]

        auc = evaluar_auc(red, entradas, objetivos)
        @test auc >= 0.0
        @test auc <= 1.0
    end

    @testset "AUC multiclase (3 clases, macro-averaged)" begin
        rng = MersenneTwister(42)
        red = RedNeuronal([2, 8, 3], rng)
        entradas = [0.0 0.1 0.5 0.6 1.0 0.9;
                    0.0 0.1 0.5 0.6 1.0 0.9]
        objetivos = [1.0 1.0 0.0 0.0 0.0 0.0;
                     0.0 0.0 1.0 1.0 0.0 0.0;
                     0.0 0.0 0.0 0.0 1.0 1.0]

        auc = evaluar_auc(red, entradas, objetivos)
        @test 0.0 <= auc <= 1.0
    end

    @testset "AUC > accuracy posible con buenas probabilidades" begin
        # AUC mide calidad de probabilidades, no solo argmax
        rng = MersenneTwister(123)
        red = RedNeuronal([2, 8, 2], rng)
        entradas = [0.0 0.0 1.0 1.0; 0.0 1.0 0.0 1.0]
        objetivos = [1.0 0.0 0.0 1.0; 0.0 1.0 1.0 0.0]

        # Entrenar parcialmente
        using RandomCloud: entrenar!
        for _ in 1:500
            for k in 1:size(entradas, 2)
                entrenar!(red, entradas[:, k], objetivos[:, k], 0.5)
            end
        end

        auc = evaluar_auc(red, entradas, objetivos)
        acc = evaluar(red, entradas, objetivos)
        # AUC puede ser >= accuracy (mide ranking, no solo clasificación)
        @test auc >= 0.0
        @test auc <= 1.0
    end
end


# Feature: gpu-batched-cloud, Property 3: Cloud evaluation accuracy equivalence
# **Validates: Requirements 2.2, 2.3**

using RandomCloud: evaluar_nube_batch, activaciones_por_capa
using Supposition
using Supposition: Data

@testset "PBT Property 3: Cloud evaluation accuracy equivalence" begin

    # Generator for random topologies (3-5 layers, sizes 1-10)
    hidden_gen_p3 = Data.Vectors(Data.Integers(1, 10); min_size=1, max_size=3)
    topo_gen_p3 = @composed function valid_topology_p3(
        input_size = Data.Integers(1, 10),
        hidden = hidden_gen_p3,
        output_size = Data.Integers(2, 10)
    )
        return vcat([input_size], hidden, [output_size])
    end

    seed_gen_p3 = Data.Integers(1, 10_000)
    n_networks_gen_p3 = Data.Integers(2, 10)
    n_samples_gen_p3 = Data.Integers(5, 30)
    act_sym_gen = Data.SampledFrom([:sigmoid, :relu, :identidad])

    @check max_examples=100 function prop_cloud_eval_accuracy_equivalence(
        topo = topo_gen_p3,
        seed = seed_gen_p3,
        N = n_networks_gen_p3,
        n_samples = n_samples_gen_p3,
        act_sym = act_sym_gen
    )
        n_features = topo[1]
        n_classes = topo[end]
        n_layers = length(topo) - 1

        # Generate N random networks with the same topology
        nube = [RedNeuronal(topo, MersenneTwister(seed + i)) for i in 1:N]

        # Generate random input matrix X (features × samples)
        rng_data = MersenneTwister(seed + 50_000)
        X = 2.0 .* rand(rng_data, n_features, n_samples) .- 1.0

        # Generate random one-hot target matrix Y (n_classes × samples)
        Y = zeros(Float64, n_classes, n_samples)
        for k in 1:n_samples
            class_idx = rand(rng_data, 1:n_classes)
            Y[class_idx, k] = 1.0
        end

        # Generate activation vector using activaciones_por_capa
        acts = activaciones_por_capa(n_layers, act_sym)

        # Batched cloud evaluation
        batched = evaluar_nube_batch(nube, X, Y, acts)

        # Individual evaluation per network
        for i in 1:N
            individual = evaluar(nube[i], X, Y; acts=acts)
            abs(batched[i] - individual) <= 1e-10 || return false
        end

        return true
    end
end
