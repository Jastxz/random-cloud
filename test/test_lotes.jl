# Tests PBT para lotes.jl — Cloud weight packing / unpacking utilities

using Random
using RandomCloud: RedNeuronal, empaquetar_pesos, reempaquetar_pesos
using Supposition
using Supposition: Data

# Feature: gpu-batched-cloud, Property 2: Weight packing round-trip
# **Validates: Requirements 2.1**

@testset "PBT Property 2: Weight packing round-trip" begin

    # Generator for random topologies (3-5 layers, sizes 1-10)
    hidden_gen_p2 = Data.Vectors(Data.Integers(1, 10); min_size=1, max_size=3)
    topo_gen_p2 = @composed function valid_topology_p2(
        input_size = Data.Integers(1, 10),
        hidden = hidden_gen_p2,
        output_size = Data.Integers(1, 10)
    )
        return vcat([input_size], hidden, [output_size])
    end

    seed_gen_p2 = Data.Integers(1, 10_000)
    n_networks_gen = Data.Integers(2, 20)

    @check max_examples=100 function prop_weight_packing_roundtrip(
        topo = topo_gen_p2,
        seed = seed_gen_p2,
        N = n_networks_gen
    )
        # Generate N random networks with the same topology
        nube = [RedNeuronal(topo, MersenneTwister(seed + i)) for i in 1:N]

        # Pack via empaquetar_pesos
        W3ds, B3ds = empaquetar_pesos(nube, Float64)

        n_capas = length(nube[1].pesos)

        # For each network i and each layer l, assert bitwise equality
        for i in 1:N
            for l in 1:n_capas
                W3ds[l][:, :, i] == nube[i].pesos[l] || return false
                B3ds[l][:, 1, i] == nube[i].biases[l] || return false
            end
        end

        return true
    end
end

# Feature: gpu-batched-cloud, Property 4: Re-packing preserves remaining networks
# **Validates: Requirements 2.4**

@testset "PBT Property 4: Re-packing preserves remaining networks" begin

    # Generator for random topologies (3-5 layers, sizes 1-10)
    hidden_gen_p4 = Data.Vectors(Data.Integers(1, 10); min_size=1, max_size=3)
    topo_gen_p4 = @composed function valid_topology_p4(
        input_size = Data.Integers(1, 10),
        hidden = hidden_gen_p4,
        output_size = Data.Integers(1, 10)
    )
        return vcat([input_size], hidden, [output_size])
    end

    seed_gen_p4 = Data.Integers(1, 10_000)
    n_networks_gen_p4 = Data.Integers(3, 15)

    @check max_examples=100 function prop_repacking_preserves_remaining(
        topo = topo_gen_p4,
        seed = seed_gen_p4,
        N = n_networks_gen_p4
    )
        # Generate N random networks with the same topology
        nube = [RedNeuronal(topo, MersenneTwister(seed + i)) for i in 1:N]

        # Pack the full cloud first
        W3ds, B3ds = empaquetar_pesos(nube, Float64)

        # Pick a random subset of indices (at least 1, at most N)
        rng_sub = MersenneTwister(seed + 99_999)
        subset_size = rand(rng_sub, 1:N)
        indices = sort(shuffle(rng_sub, collect(1:N))[1:subset_size])

        # Re-pack with the subset
        W3ds_new, B3ds_new = reempaquetar_pesos(nube, indices, W3ds, B3ds, Float64)

        n_capas = length(nube[1].pesos)

        # For each j in 1:length(indices), slice j of new tensors must match nube[indices[j]]
        for j in 1:length(indices)
            idx = indices[j]
            for l in 1:n_capas
                W3ds_new[l][:, :, j] == nube[idx].pesos[l] || return false
                B3ds_new[l][:, 1, j] == nube[idx].biases[l] || return false
            end
        end

        return true
    end
end
