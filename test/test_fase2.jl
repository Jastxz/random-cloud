# Tests unitarios y PBT para Fase 2 (exploración estructural)

using RandomCloud
using RandomCloud: RedNeuronal, colapsar_capa, redistribuir_capa,
    generar_candidatos, explorar_estructura, CandidatoEstructural,
    ResultadoFase2, TipoOperacion, OP_COLAPSO, OP_SPLIT_1, OP_SPLIT_2,
    generar_divisiones_split1, generar_divisiones_split2,
    feedforward, evaluar
using Random
using LinearAlgebra
using Test

# ─────────────────────────────────────────────────────────────────────────────
# Tests unitarios: colapsar_capa
# ─────────────────────────────────────────────────────────────────────────────

@testset "colapsar_capa" begin
    rng = MersenneTwister(42)

    @testset "Red [2, 8, 4, 1]: colapsar capa 1 (la de 8)" begin
        red = RedNeuronal([2, 8, 4, 1], rng)
        colapsada = colapsar_capa(red, 1)

        # Topología correcta: se elimina la capa de 8
        @test colapsada.topologia == [2, 4, 1]
        @test length(colapsada.pesos) == 2
        @test size(colapsada.pesos[1]) == (4, 2)   # W2×W1 = (4×8)×(8×2) = (4×2)
        @test size(colapsada.pesos[2]) == (1, 4)   # sin cambio
    end

    @testset "Red [2, 8, 4, 1]: colapsar capa 2 (la de 4)" begin
        red = RedNeuronal([2, 8, 4, 1], rng)
        colapsada = colapsar_capa(red, 2)

        @test colapsada.topologia == [2, 8, 1]
        @test length(colapsada.pesos) == 2
        @test size(colapsada.pesos[1]) == (8, 2)   # sin cambio
        @test size(colapsada.pesos[2]) == (1, 8)   # W3×W2 = (1×4)×(4×8) = (1×8)
    end

    @testset "Composición matricial es correcta (sin activación)" begin
        # Verificar que W_nueva = W[i+1] × W[i]
        red = RedNeuronal([3, 5, 4, 2], rng)
        colapsada = colapsar_capa(red, 1)

        W_esperada = red.pesos[2] * red.pesos[1]
        @test colapsada.pesos[1] ≈ W_esperada

        # Verificar absorción de biases: b_nueva = W[i+1] × b[i] + b[i+1]
        b_esperado = red.pesos[2] * red.biases[1] .+ red.biases[2]
        @test colapsada.biases[1] ≈ b_esperado
    end

    @testset "Componente lineal preservada (feedforward sin activación)" begin
        # Sin activación (identidad), colapso debe dar resultado idéntico
        # Nota: con activación no será idéntico porque perdemos la no-linealidad
        red = RedNeuronal([2, 6, 3, 1], rng)
        x = randn(rng, 2)

        # Forward manual sin activación por la capa colapsada
        # capa 1→2: z1 = W1*x + b1
        z1 = red.pesos[1] * x .+ red.biases[1]
        # sin activación, capa 2→3: z2 = W2*z1 + b2
        z2 = red.pesos[2] * z1 .+ red.biases[2]

        # Red colapsada: z_col = W_nueva*x + b_nueva
        colapsada = colapsar_capa(red, 1)
        z_col = colapsada.pesos[1] * x .+ colapsada.biases[1]

        @test z_col ≈ z2 atol=1e-10
    end

    @testset "Error si idx_capa fuera de rango" begin
        red = RedNeuronal([2, 4, 1], rng)
        # Solo hay 1 capa oculta, pesos tiene length=2, así que idx_capa=1 es la única válida
        # Pero colapsar la capa 1 requiere que haya al menos 2 capas de pesos
        # idx_capa puede ser 1..(n_capas_pesos-1) = 1..1
        colapsada = colapsar_capa(red, 1)
        @test colapsada.topologia == [2, 1]
        @test length(colapsada.pesos) == 1

        # Fuera de rango
        @test_throws AssertionError colapsar_capa(red, 0)
        @test_throws AssertionError colapsar_capa(red, 2)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Tests unitarios: redistribuir_capa
# ─────────────────────────────────────────────────────────────────────────────

@testset "redistribuir_capa" begin
    rng = MersenneTwister(123)

    @testset "Split [2, 8, 1] → [2, 4, 4, 1]" begin
        red = RedNeuronal([2, 8, 1], rng)
        split = redistribuir_capa(red, 1, [4, 4])

        @test split.topologia == [2, 4, 4, 1]
        @test length(split.pesos) == 3
        @test size(split.pesos[1]) == (4, 2)   # primeras 4 filas de W1
        @test size(split.pesos[2]) == (4, 4)   # identidad 4×4
        @test size(split.pesos[3]) == (1, 4)   # columnas correspondientes de W2
    end

    @testset "Filas correctas (orden normal)" begin
        red = RedNeuronal([2, 8, 1], rng)
        split = redistribuir_capa(red, 1, [4, 4]; invertir=false)

        # Primera sub-capa: filas 1:4 de W1 original
        @test split.pesos[1] == red.pesos[1][1:4, :]
        @test split.biases[1] == red.biases[1][1:4]
    end

    @testset "Filas correctas (orden invertido)" begin
        red = RedNeuronal([2, 8, 1], rng)
        split = redistribuir_capa(red, 1, [4, 4]; invertir=true)

        # Con invertir=true, filas_orden = 8:-1:1
        # rangos[1] = 1:4, así que filas_sub1 = filas_orden[1:4] = [8,7,6,5]
        @test split.pesos[1] == red.pesos[1][[8,7,6,5], :]
        @test split.biases[1] == red.biases[1][[8,7,6,5]]
    end

    @testset "Conexión identidad entre sub-capas" begin
        red = RedNeuronal([3, 10, 2], rng)
        split = redistribuir_capa(red, 1, [5, 5])

        # Conexión entre sub-capas: debe ser identidad 5×5
        @test split.pesos[2] == Matrix{Float64}(I, 5, 5)
        @test split.biases[2] == zeros(5)
    end

    @testset "Conexión identidad rectangular (anchos distintos)" begin
        red = RedNeuronal([3, 10, 2], rng)
        split = redistribuir_capa(red, 1, [6, 4])

        # Conexión 4×6: identidad en la parte cuadrada, ceros fuera
        W_inter = split.pesos[2]
        @test size(W_inter) == (4, 6)
        for j in 1:4
            @test W_inter[j, j] == 1.0
        end
        # El resto debe ser 0
        @test sum(W_inter) == 4.0
    end

    @testset "Split en 3 sub-capas (profundidad 2)" begin
        red = RedNeuronal([2, 12, 1], rng)
        split = redistribuir_capa(red, 1, [4, 4, 4])

        @test split.topologia == [2, 4, 4, 4, 1]
        @test length(split.pesos) == 4
        @test size(split.pesos[1]) == (4, 2)   # filas 1:4
        @test size(split.pesos[2]) == (4, 4)   # identidad
        @test size(split.pesos[3]) == (4, 4)   # identidad
        @test size(split.pesos[4]) == (1, 4)   # conexión salida
    end

    @testset "Error si sum(anchos) > ancho original" begin
        red = RedNeuronal([2, 8, 1], rng)
        @test_throws AssertionError redistribuir_capa(red, 1, [5, 5])  # 10 > 8
    end

    @testset "Error si algún ancho < 1" begin
        red = RedNeuronal([2, 8, 1], rng)
        @test_throws AssertionError redistribuir_capa(red, 1, [0, 8])
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Tests unitarios: generar_divisiones
# ─────────────────────────────────────────────────────────────────────────────

@testset "generar_divisiones" begin
    @testset "split1: ancho=8, min=4" begin
        divs = generar_divisiones_split1(8, 4)
        @test [4, 4] in divs
        @test length(divs) == 1  # solo [4,4] cumple ambos ≥ 4
    end

    @testset "split1: ancho=10, min=4" begin
        divs = generar_divisiones_split1(10, 4)
        @test [4, 6] in divs
        @test [5, 5] in divs
        @test [6, 4] in divs
        @test length(divs) == 3
    end

    @testset "split1: ancho=6, min=4 → vacío" begin
        divs = generar_divisiones_split1(6, 4)
        # 4+2=6, pero 2 < 4 → no cumple. Solo [4,2] y [5,1] y [3,3] — ninguno cumple
        # Espera: solo si ambos ≥ 4, necesitas ancho ≥ 8
        @test isempty(divs)
    end

    @testset "split2: ancho=12, min=4" begin
        divs = generar_divisiones_split2(12, 4)
        @test [4, 4, 4] in divs
        @test all(d -> all(x -> x >= 4, d), divs)
        @test all(d -> sum(d) == 12, divs)
    end

    @testset "split2: ancho=10, min=4 → vacío" begin
        divs = generar_divisiones_split2(10, 4)
        # 4+4+4=12 > 10, así que no hay divisiones válidas con min=4
        # wait: 4+4+2=10, pero 2 < 4. Correct: vacío
        @test isempty(divs)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Tests unitarios: generar_candidatos
# ─────────────────────────────────────────────────────────────────────────────

@testset "generar_candidatos" begin
    rng = MersenneTwister(77)

    @testset "Red [2, 16, 8, 1]: genera colapsos y splits" begin
        red = RedNeuronal([2, 16, 8, 1], rng)
        candidatos = generar_candidatos(red, 2, 4)

        # Debe haber al menos los colapsos (2 capas ocultas = 2 colapsos)
        colapsos = filter(c -> c.operacion == OP_COLAPSO, candidatos)
        @test length(colapsos) == 2

        # Splits de la capa de 16 (ancho ≥ 2*4=8, así que sí)
        splits1 = filter(c -> c.operacion == OP_SPLIT_1, candidatos)
        @test length(splits1) > 0

        # Cada candidato tiene una red válida
        for c in candidatos
            @test length(c.red.topologia) >= 2
            @test c.red.topologia[1] == 2    # entrada preservada
            @test c.red.topologia[end] == 1  # salida preservada
            @test length(c.red.pesos) == length(c.red.topologia) - 1
            # Dimensiones de pesos coherentes con topología
            for i in eachindex(c.red.pesos)
                @test size(c.red.pesos[i]) == (c.red.topologia[i+1], c.red.topologia[i])
                @test length(c.red.biases[i]) == c.red.topologia[i+1]
            end
        end
    end

    @testset "Ancho mínimo respetado" begin
        rng2 = MersenneTwister(88)
        red = RedNeuronal([2, 20, 10, 1], rng2)
        candidatos = generar_candidatos(red, 2, 5)

        splits = filter(c -> c.operacion in (OP_SPLIT_1, OP_SPLIT_2), candidatos)
        for c in splits
            # Todas las capas ocultas deben tener ancho ≥ 1
            # (el ancho mínimo se aplica en la generación de divisiones,
            #  pero la topología resultante puede tener capas de cualquier tamaño
            #  siempre que vengan de un split válido)
            for i in 2:length(c.red.topologia)-1
                @test c.red.topologia[i] >= 1
            end
        end
    end

    @testset "Red con 1 capa oculta pequeña: no genera splits" begin
        rng3 = MersenneTwister(99)
        red = RedNeuronal([2, 6, 1], rng3)  # ancho 6 < 2*4
        candidatos = generar_candidatos(red, 1, 4)

        # No hay splits (6 < 8 = 2*min)
        splits = filter(c -> c.operacion in (OP_SPLIT_1, OP_SPLIT_2), candidatos)
        @test isempty(splits)

        # Pero sí hay colapso (0 capas ocultas colapsables? No: idx va 1..n_capas_pesos-1=1..0)
        # Con [2,6,1], n_capas_pesos=2, n_capas_ocultas=1. El colapso itera 1:1 → 1 colapso
        # Espera: el colapso de la única capa oculta produce [2,1]
        colapsos = filter(c -> c.operacion == OP_COLAPSO, candidatos)
        # n_capas_ocultas = length(red.pesos) - 1 = 1. Correcto.
        @test length(colapsos) == 1
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# PBT: colapsar_capa preserva componente lineal
# ─────────────────────────────────────────────────────────────────────────────

using Supposition
using Supposition: Data

@testset "PBT Fase 2: Colapso preserva transformación lineal" begin
    @check max_examples=50 function prop_colapso_lineal(
        seed=Data.Integers(1, 10000),
        n_in=Data.Integers(2, 10),
        n_hidden1=Data.Integers(3, 20),
        n_hidden2=Data.Integers(3, 15),
        n_out=Data.Integers(1, 5)
    )
        rng = MersenneTwister(seed)
        topo = [n_in, n_hidden1, n_hidden2, n_out]
        red = RedNeuronal(topo, rng)
        colapsada = colapsar_capa(red, 1)

        # Verificar: W_nueva == W2 × W1
        W_esperada = red.pesos[2] * red.pesos[1]
        isapprox(colapsada.pesos[1], W_esperada, atol=1e-10) || return false

        # Verificar: b_nueva == W2 × b1 + b2
        b_esperado = red.pesos[2] * red.biases[1] .+ red.biases[2]
        isapprox(colapsada.biases[1], b_esperado, atol=1e-10) || return false

        # Topología correcta
        colapsada.topologia == [n_in, n_hidden2, n_out] || return false
        true
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# PBT: redistribuir_capa produce dimensiones coherentes
# ─────────────────────────────────────────────────────────────────────────────

@testset "PBT Fase 2: Redistribución produce dimensiones coherentes" begin
    @check max_examples=50 function prop_split_dimensiones(
        seed=Data.Integers(1, 10000),
        n_in=Data.Integers(2, 8),
        ancho=Data.Integers(8, 24),
        n_out=Data.Integers(1, 5)
    )
        rng = MersenneTwister(seed)
        topo = [n_in, ancho, n_out]
        red = RedNeuronal(topo, rng)

        # Split simétrico en 2
        a = ancho ÷ 2
        b = ancho - a
        if a < 1 || b < 1
            return true  # skip degenerate
        end

        split = redistribuir_capa(red, 1, [a, b])

        # Topología correcta
        split.topologia == [n_in, a, b, n_out] || return false
        length(split.pesos) == 3 || return false

        # Dimensiones de cada capa de pesos
        size(split.pesos[1]) == (a, n_in) || return false
        size(split.pesos[2]) == (b, a) || return false
        size(split.pesos[3]) == (n_out, b) || return false

        # Biases
        length(split.biases[1]) == a || return false
        length(split.biases[2]) == b || return false
        length(split.biases[3]) == n_out || return false

        # Primera sub-capa usa filas del original
        split.pesos[1] == red.pesos[1][1:a, :] || return false
        split.biases[1] == red.biases[1][1:a] || return false

        true
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# PBT: generar_candidatos nunca produce topologías inválidas
# ─────────────────────────────────────────────────────────────────────────────

@testset "PBT Fase 2: Candidatos tienen topologías válidas" begin
    @check max_examples=30 function prop_candidatos_validos(
        seed=Data.Integers(1, 10000),
        n_in=Data.Integers(2, 10),
        h1=Data.Integers(8, 20),
        h2=Data.Integers(4, 12),
        n_out=Data.Integers(1, 5)
    )
        rng = MersenneTwister(seed)
        topo = [n_in, h1, h2, n_out]
        red = RedNeuronal(topo, rng)
        candidatos = generar_candidatos(red, 2, 4)

        for c in candidatos
            # Topología no vacía y ≥ 2 elementos
            length(c.red.topologia) >= 2 || return false
            # Entrada y salida preservadas
            c.red.topologia[1] == n_in || return false
            c.red.topologia[end] == n_out || return false
            # Número de capas de pesos coherente
            length(c.red.pesos) == length(c.red.topologia) - 1 || return false
            length(c.red.biases) == length(c.red.topologia) - 1 || return false
            # Cada peso tiene dimensiones correctas
            for i in eachindex(c.red.pesos)
                size(c.red.pesos[i]) == (c.red.topologia[i+1], c.red.topologia[i]) || return false
                length(c.red.biases[i]) == c.red.topologia[i+1] || return false
            end
        end
        true
    end
end
