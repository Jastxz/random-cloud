# lotes.jl — Cloud weight packing / unpacking utilities for batched evaluation

"""
    empaquetar_pesos(nube::Vector{RedNeuronal}, ::Type{T}=Float64) → (W3ds, B3ds)

Pack cloud weights into 3-D tensors for batched operations.

For each layer `l`:
- `W3ds[l]` has shape `(neurons_out, neurons_in, N)`
- `B3ds[l]` has shape `(neurons_out, 1, N)`

where `N = length(nube)`.
"""
function empaquetar_pesos(nube::Vector{RedNeuronal}, ::Type{T}=Float64) where T
    N = length(nube)
    n_capas = length(nube[1].pesos)

    W3ds = Vector{Array{T,3}}(undef, n_capas)
    B3ds = Vector{Array{T,3}}(undef, n_capas)

    for l in 1:n_capas
        neurons_out, neurons_in = size(nube[1].pesos[l])
        W = Array{T,3}(undef, neurons_out, neurons_in, N)
        B = Array{T,3}(undef, neurons_out, 1, N)
        for i in 1:N
            W[:, :, i] = T.(nube[i].pesos[l])
            B[:, 1, i] = T.(nube[i].biases[l])
        end
        W3ds[l] = W
        B3ds[l] = B
    end

    return (W3ds, B3ds)
end

"""
    reempaquetar_pesos(nube, indices, W3ds_old, B3ds_old, ::Type{T}=Float64) → (W3ds_new, B3ds_new)

Re-pack after topology reduction, keeping only networks at given `indices`.

Returns new 3-D tensors where slice `j` corresponds to `nube[indices[j]]`.
When the topology has changed (networks were reconstructed), weights are read
from the `nube` vector directly. When topology is unchanged, slices are copied
from the old tensors for efficiency.
"""
function reempaquetar_pesos(
    nube::Vector{RedNeuronal},
    indices::AbstractVector{<:Integer},
    W3ds_old::Vector{Array{T,3}},
    B3ds_old::Vector{Array{T,3}},
    ::Type{T}=Float64
) where T
    N_new = length(indices)
    n_capas = length(nube[indices[1]].pesos)

    W3ds_new = Vector{Array{T,3}}(undef, n_capas)
    B3ds_new = Vector{Array{T,3}}(undef, n_capas)

    # Check if topology changed by comparing layer dimensions
    topology_changed = n_capas != length(W3ds_old) ||
        any(l -> size(nube[indices[1]].pesos[l]) != (size(W3ds_old[l], 1), size(W3ds_old[l], 2)), 1:min(n_capas, length(W3ds_old)))

    for l in 1:n_capas
        neurons_out, neurons_in = size(nube[indices[1]].pesos[l])
        W = Array{T,3}(undef, neurons_out, neurons_in, N_new)
        B = Array{T,3}(undef, neurons_out, 1, N_new)

        if topology_changed
            # Topology changed — read from nube directly
            for (j, idx) in enumerate(indices)
                W[:, :, j] = T.(nube[idx].pesos[l])
                B[:, 1, j] = T.(nube[idx].biases[l])
            end
        else
            # Topology unchanged — copy slices from old tensors
            for (j, idx) in enumerate(indices)
                W[:, :, j] = W3ds_old[l][:, :, idx]
                B[:, 1, j] = B3ds_old[l][:, 1, idx]
            end
        end

        W3ds_new[l] = W
        B3ds_new[l] = B
    end

    return (W3ds_new, B3ds_new)
end
