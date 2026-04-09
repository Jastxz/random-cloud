# Tests unitarios y PBT para RedNeuronal

using Random
using RandomCloud: RedNeuronal, feedforward, entrenar!, reconstruir

@testset "RedNeuronal" begin

    @testset "Construcción: dimensiones de pesos y biases" begin
        red = RedNeuronal([2, 4, 1], MersenneTwister(42))

        @test red.topologia == [2, 4, 1]
        @test length(red.pesos) == 2
        @test length(red.biases) == 2

        # pesos[1] es 4×2 (capa oculta × entrada)
        @test size(red.pesos[1]) == (4, 2)
        # pesos[2] es 1×4 (salida × capa oculta)
        @test size(red.pesos[2]) == (1, 4)
        # biases[1] longitud 4
        @test length(red.biases[1]) == 4
        # biases[2] longitud 1
        @test length(red.biases[2]) == 1
    end

    @testset "Copia defensiva de topología" begin
        topo = [2, 4, 1]
        red = RedNeuronal(topo, MersenneTwister(42))
        topo[2] = 99
        @test red.topologia == [2, 4, 1]
    end

    @testset "Feedforward: dimensión de salida y rango (0,1)" begin
        red = RedNeuronal([2, 4, 1], MersenneTwister(42))
        salida = feedforward(red, [0.5, 0.3])

        # Dimensión igual a última capa
        @test length(salida) == 1
        # Valores en (0, 1) por sigmoid
        @test all(0.0 .< salida .< 1.0)

        # Con topología más grande
        red2 = RedNeuronal([3, 5, 2], MersenneTwister(7))
        salida2 = feedforward(red2, [0.1, 0.2, 0.9])
        @test length(salida2) == 2
        @test all(0.0 .< salida2 .< 1.0)
    end

    @testset "entrenar! modifica pesos" begin
        red = RedNeuronal([2, 4, 1], MersenneTwister(42))

        # Guardar copia de pesos antes de entrenar
        pesos_antes = [copy(w) for w in red.pesos]
        biases_antes = [copy(b) for b in red.biases]

        entrenar!(red, [1.0, 0.0], [1.0], 0.1)

        # Al menos un peso o bias debe haber cambiado
        algun_peso_cambio = any(
            red.pesos[i] != pesos_antes[i] for i in 1:length(red.pesos)
        )
        algun_bias_cambio = any(
            red.biases[i] != biases_antes[i] for i in 1:length(red.biases)
        )
        @test algun_peso_cambio || algun_bias_cambio
    end

    @testset "reconstruir con topología reducida" begin
        red = RedNeuronal([2, 4, 3, 1], MersenneTwister(42))

        red_rec = reconstruir(red, [2, 3, 2, 1])

        # Verificar topología resultante
        @test red_rec.topologia == [2, 3, 2, 1]

        # Verificar dimensiones recortadas
        @test size(red_rec.pesos[1]) == (3, 2)
        @test size(red_rec.pesos[2]) == (2, 3)
        @test size(red_rec.pesos[3]) == (1, 2)
        @test length(red_rec.biases[1]) == 3
        @test length(red_rec.biases[2]) == 2
        @test length(red_rec.biases[3]) == 1

        # Verificar que los pesos preservados son submatrices de los originales
        @test red_rec.pesos[1] == red.pesos[1][1:3, 1:2]
        @test red_rec.pesos[2] == red.pesos[2][1:2, 1:3]
        @test red_rec.pesos[3] == red.pesos[3][1:1, 1:2]
        @test red_rec.biases[1] == red.biases[1][1:3]
        @test red_rec.biases[2] == red.biases[2][1:2]
        @test red_rec.biases[3] == red.biases[3][1:1]

        # Verificar que la red original no fue modificada
        @test red.topologia == [2, 4, 3, 1]
        @test size(red.pesos[1]) == (4, 2)
    end

    @testset "reconstruir con capa oculta en 0 neuronas (edge case 2.12)" begin
        red = RedNeuronal([2, 4, 3, 1], MersenneTwister(42))

        red_rec = reconstruir(red, [2, 0, 2, 1])

        # La capa con 0 neuronas se elimina, topología colapsa
        @test red_rec.topologia == [2, 2, 1]
        @test length(red_rec.pesos) == 2
        @test length(red_rec.biases) == 2

        # Dimensiones de la red colapsada
        @test size(red_rec.pesos[1]) == (2, 2)
        @test size(red_rec.pesos[2]) == (1, 2)
        @test length(red_rec.biases[1]) == 2
        @test length(red_rec.biases[2]) == 1
    end

end

# Feature: random-cloud-julia, Property 7: Entrenamiento modifica pesos
# **Validates: Requirements 2.6**
using Supposition
using Supposition: Data

@testset "PBT Propiedad 7: Entrenamiento modifica pesos" begin

    # Generator for valid topology: at least 3 layers, hidden layers ≥ 1, sizes 1-8
    hidden_gen = Data.Vectors(Data.Integers(1, 8); min_size=1, max_size=3)
    topo_gen = @composed function valid_topology(
        input_size = Data.Integers(1, 8),
        hidden = hidden_gen,
        output_size = Data.Integers(1, 8)
    )
        return vcat([input_size], hidden, [output_size])
    end

    seed_gen = Data.Integers(1, 10_000)

    @check max_examples=100 function prop_entrenar_modifica_pesos(
        topo = topo_gen,
        seed = seed_gen
    )
        rng = MersenneTwister(seed)
        red = RedNeuronal(topo, rng)

        n_entrada = topo[1]
        n_salida = topo[end]

        # Generate random input and target in [0,1] using a deterministic RNG from seed
        rng2 = MersenneTwister(seed + 10_000)
        entrada = rand(rng2, n_entrada)
        objetivo = rand(rng2, n_salida)

        # Save copies of weights and biases before training
        pesos_antes = [copy(w) for w in red.pesos]
        biases_antes = [copy(b) for b in red.biases]

        # Train with lr=0.1
        entrenar!(red, entrada, objetivo, 0.1)

        # Verify at least one weight or bias changed
        algun_peso_cambio = any(
            red.pesos[i] != pesos_antes[i] for i in 1:length(red.pesos)
        )
        algun_bias_cambio = any(
            red.biases[i] != biases_antes[i] for i in 1:length(red.biases)
        )

        algun_peso_cambio || algun_bias_cambio
    end
end

# Feature: random-cloud-julia, Property 8: Reconstruir preserva pesos
# **Validates: Requirements 2.9, 2.10, 2.11**

@testset "PBT Propiedad 8: Reconstruir preserva pesos mediante recorte de submatrices" begin

    # Generator for valid topology: at least 3 layers, all sizes 2-8
    hidden_gen = Data.Vectors(Data.Integers(2, 8); min_size=1, max_size=3)
    topo_gen = @composed function valid_topology(
        input_size = Data.Integers(2, 8),
        hidden = hidden_gen,
        output_size = Data.Integers(2, 8)
    )
        return vcat([input_size], hidden, [output_size])
    end

    seed_gen = Data.Integers(1, 10_000)

    @check max_examples=100 function prop_reconstruir_preserva_pesos(
        topo = topo_gen,
        seed = seed_gen
    )
        rng = MersenneTwister(seed)
        red = RedNeuronal(topo, rng)

        # Generate a valid reduced topology:
        # - Input and output layers stay the same
        # - Hidden layers are between 1 and original size (no zeros for simple case)
        rng2 = MersenneTwister(seed + 50_000)
        reduced_topo = copy(topo)
        for j in 2:(length(topo) - 1)
            reduced_topo[j] = rand(rng2, 1:topo[j])
        end

        # Save copies of original weights and biases before reconstruir
        pesos_orig = [copy(w) for w in red.pesos]
        biases_orig = [copy(b) for b in red.biases]
        topo_orig = copy(red.topologia)

        # Call reconstruir
        red_rec = reconstruir(red, reduced_topo)

        # Verify topology
        red_rec.topologia == reduced_topo || return false

        # Verify for each layer: pesos and biases are the top-left submatrix of originals
        n_layers = length(reduced_topo) - 1
        for i in 1:n_layers
            filas = reduced_topo[i + 1]
            cols = reduced_topo[i]
            # Check pesos: should be top-left submatrix
            red_rec.pesos[i] == pesos_orig[i][1:filas, 1:cols] || return false
            # Check biases: should be first t'[i+1] components
            red_rec.biases[i] == biases_orig[i][1:filas] || return false
        end

        # Verify original red was NOT modified
        red.topologia == topo_orig || return false
        for i in 1:length(pesos_orig)
            red.pesos[i] == pesos_orig[i] || return false
            red.biases[i] == biases_orig[i] || return false
        end

        return true
    end
end

# Feature: random-cloud-julia, Property 13: Semillas diferentes → redes diferentes
# **Validates: Requirements 8.2**

@testset "PBT Propiedad 13: Semillas diferentes producen redes diferentes" begin

    # Generator for valid topology: at least 3 layers, sizes 1-8
    hidden_gen = Data.Vectors(Data.Integers(1, 8); min_size=1, max_size=3)
    topo_gen = @composed function valid_topology_p13(
        input_size = Data.Integers(1, 8),
        hidden = hidden_gen,
        output_size = Data.Integers(1, 8)
    )
        return vcat([input_size], hidden, [output_size])
    end

    # Generator for two different seeds
    seed_pair_gen = @composed function different_seeds(
        s1 = Data.Integers(1, 100_000),
        s2 = Data.Integers(1, 100_000)
    )
        # Ensure seeds are different
        if s1 == s2
            s2 = s1 + 1
        end
        return (s1, s2)
    end

    @check max_examples=100 function prop_semillas_diferentes_redes_diferentes(
        topo = topo_gen,
        seeds = seed_pair_gen
    )
        seed1, seed2 = seeds

        red1 = RedNeuronal(topo, MersenneTwister(seed1))
        red2 = RedNeuronal(topo, MersenneTwister(seed2))

        # At least one weight matrix must differ between the two networks
        alguna_diferencia = any(
            red1.pesos[i] != red2.pesos[i] for i in 1:length(red1.pesos)
        )

        return alguna_diferencia
    end
end

# Feature: gpu-batched-cloud, Property 1: Batched feedforward equivalence
# **Validates: Requirements 1.1, 1.2, 1.3**

using RandomCloud: feedforward_batch

@testset "PBT Property 1: Batched feedforward equivalence" begin

    # Generator for random topologies (3-5 layers, reasonable sizes)
    hidden_gen_p1 = Data.Vectors(Data.Integers(1, 30); min_size=1, max_size=3)
    topo_gen_p1 = @composed function valid_topology_p1(
        input_size = Data.Integers(1, 20),
        hidden = hidden_gen_p1,
        output_size = Data.Integers(1, 10)
    )
        return vcat([input_size], hidden, [output_size])
    end

    seed_gen_p1 = Data.Integers(1, 10_000)
    n_samples_gen = Data.Integers(1, 50)
    act_gen = Data.Vectors(Data.SampledFrom([:sigmoid, :relu, :identidad]); min_size=1, max_size=4)

    @check max_examples=100 function prop_batched_feedforward_equivalence(
        topo = topo_gen_p1,
        seed = seed_gen_p1,
        n_samples = n_samples_gen,
        raw_acts = act_gen
    )
        rng = MersenneTwister(seed)
        red = RedNeuronal(topo, rng)

        n_features = topo[1]
        n_layers = length(topo) - 1

        # Build activation vector matching the number of layers
        acts = [raw_acts[mod1(i, length(raw_acts))] for i in 1:n_layers]

        # Generate random input matrix X (features × samples)
        rng2 = MersenneTwister(seed + 20_000)
        X = 2.0 .* rand(rng2, n_features, n_samples) .- 1.0

        # Batched feedforward
        Y_batch = feedforward_batch(red.pesos, red.biases, X, acts)

        # Column-by-column feedforward
        for j in 1:n_samples
            y_single = feedforward(red, X[:, j], acts)
            for k in eachindex(y_single)
                abs(Y_batch[k, j] - y_single[k]) <= 1e-10 || return false
            end
        end

        return true
    end
end

# Feature: gpu-batched-cloud, Task 2.3: Unit tests for feedforward_batch edge cases
# **Validates: Requirements 1.1, 1.2, 1.3, 1.4**

@testset "feedforward_batch edge cases" begin

    @testset "Single-sample input (features × 1 matrix)" begin
        red = RedNeuronal([3, 5, 2], MersenneTwister(42))
        X = rand(MersenneTwister(99), 3, 1)  # 3 features, 1 sample
        acts = [:sigmoid, :sigmoid]

        Y = feedforward_batch(red.pesos, red.biases, X, acts)

        @test size(Y) == (2, 1)
        # Verify matches single-vector feedforward
        y_single = feedforward(red, X[:, 1], acts)
        @test all(abs.(Y[:, 1] .- y_single) .<= 1e-10)
    end

    @testset "Single-layer network" begin
        # Topology [4, 3] — just one weight layer, no hidden layers
        topo = [4, 3]
        red = RedNeuronal(topo, MersenneTwister(77))
        X = rand(MersenneTwister(88), 4, 5)  # 4 features, 5 samples
        acts = [:sigmoid]

        Y = feedforward_batch(red.pesos, red.biases, X, acts)

        @test size(Y) == (3, 5)
        # Verify each column matches single feedforward
        for j in 1:5
            y_single = feedforward(red, X[:, j], acts)
            @test all(abs.(Y[:, j] .- y_single) .<= 1e-10)
        end
    end

    @testset "Sigmoid default when acts omitted" begin
        red = RedNeuronal([2, 4, 1], MersenneTwister(42))
        X = rand(MersenneTwister(55), 2, 10)

        # Call without acts — should default to sigmoid on all layers
        Y_default = feedforward_batch(red.pesos, red.biases, X)
        # Call with explicit sigmoid acts
        Y_explicit = feedforward_batch(red.pesos, red.biases, X, [:sigmoid, :sigmoid])

        @test size(Y_default) == size(Y_explicit)
        @test all(abs.(Y_default .- Y_explicit) .<= 1e-15)
    end

    @testset "Identity activation produces linear output" begin
        # With :identidad on all layers, output = W_L * ... * W_1 * X + accumulated biases
        topo = [3, 4, 2]
        red = RedNeuronal(topo, MersenneTwister(33))
        X = rand(MersenneTwister(44), 3, 6)
        acts = [:identidad, :identidad]

        Y = feedforward_batch(red.pesos, red.biases, X, acts)

        # Manually compute the linear transformation layer by layer
        A = X
        for i in eachindex(red.pesos)
            A = red.pesos[i] * A .+ red.biases[i]
        end

        @test size(Y) == (2, 6)
        @test all(abs.(Y .- A) .<= 1e-10)
    end

end

# Feature: gpu-batched-cloud, Property 6: Batched backprop weight-update equivalence
# **Validates: Requirements 4.2, 4.3**

using RandomCloud: entrenar_batch_matmul!

@testset "PBT Property 6: Batched backprop weight-update equivalence" begin

    # Generator for random topologies (3-5 layers, small sizes for numerical stability)
    hidden_gen_p6 = Data.Vectors(Data.Integers(1, 10); min_size=1, max_size=3)
    topo_gen_p6 = @composed function valid_topology_p6(
        input_size = Data.Integers(1, 8),
        hidden = hidden_gen_p6,
        output_size = Data.Integers(1, 5)
    )
        return vcat([input_size], hidden, [output_size])
    end

    seed_gen_p6 = Data.Integers(1, 10_000)
    n_samples_gen_p6 = Data.Integers(1, 5)

    @check max_examples=100 function prop_batched_backprop_weight_update_equivalence(
        topo = topo_gen_p6,
        seed = seed_gen_p6,
        B = n_samples_gen_p6
    )
        n_layers = length(topo) - 1
        n_features = topo[1]
        n_output = topo[end]
        acts = fill(:sigmoid, n_layers)

        # Generate random input and target data
        rng_data = MersenneTwister(seed + 30_000)
        X = rand(rng_data, n_features, B)
        Y = rand(rng_data, n_output, B)

        # Use a small learning rate to minimize sequential-vs-batch divergence
        lr = 1e-4

        # --- Sequential path: entrenar! per sample with lr/B ---
        red_seq = RedNeuronal(topo, MersenneTwister(seed))
        lr_per_sample = lr / B
        for k in 1:B
            entrenar!(red_seq, X[:, k], Y[:, k], lr_per_sample)
        end

        # --- Batched path: entrenar_batch_matmul! ---
        red_batch = RedNeuronal(topo, MersenneTwister(seed))
        entrenar_batch_matmul!(red_batch.pesos, red_batch.biases, X, Y, lr, acts)

        # Compare weights element-wise
        for i in 1:n_layers
            for idx in eachindex(red_seq.pesos[i])
                abs(red_seq.pesos[i][idx] - red_batch.pesos[i][idx]) <= 1e-8 || return false
            end
            for idx in eachindex(red_seq.biases[i])
                abs(red_seq.biases[i][idx] - red_batch.biases[i][idx]) <= 1e-8 || return false
            end
        end

        return true
    end
end
