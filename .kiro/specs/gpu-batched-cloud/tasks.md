# Implementation Plan: GPU-Batched Cloud

## Overview

Bottom-up implementation of batched matrix operations and GPU acceleration for RandomCloud.jl. Each task builds on the previous: broadcastable activations → batched feedforward → cloud packing → cloud evaluation → batched backprop → config/informe modifications → motor dispatch → GPU extension → tests → validation. All code is Julia, targeting `AbstractFloat` generics for CPU (`Float64`) and GPU (`Float32`) paths.

## Tasks

- [x] 1. Implement broadcastable activation functions
  - [x] 1.1 Add `aplicar_activacion_batch` and `aplicar_derivada_batch` to `src/activaciones.jl`
    - Implement `@inline` generic functions parameterized on `T<:AbstractFloat`
    - Support `:sigmoid`, `:relu`, and `:identidad` activations
    - Use `one(T)`, `zero(T)` for type stability with both Float32 and Float64
    - _Requirements: 1.1, 1.3_

  - [x] 1.2 Write property test for batched feedforward equivalence (Property 1)
    - **Property 1: Batched feedforward equivalence**
    - Generate random `RedNeuronal` with random topology and weights, random input matrix X, and random activation vector
    - Assert `feedforward_batch` output matches column-by-column `feedforward!` within ≤ 1e-10
    - Use Supposition.jl with min 100 iterations
    - **Validates: Requirements 1.1, 1.2, 1.3**

- [x] 2. Implement batched feedforward
  - [x] 2.1 Add `feedforward_batch` to `src/red_neuronal.jl`
    - Implement `feedforward_batch(pesos, biases, X, acts)` computing `W × X .+ b` per layer
    - Use `aplicar_activacion_batch` for element-wise activation via broadcasting
    - Accept `AbstractMatrix{T}` and `AbstractVector{T}` for CPU/GPU dispatch
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 2.2 Add default sigmoid behavior when `acts` is omitted
    - Add method signature `feedforward_batch(pesos, biases, X)` that defaults `acts` to `[:sigmoid, ...]`
    - _Requirements: 1.4_

  - [x] 2.3 Write unit tests for `feedforward_batch` edge cases
    - Test single-sample input (features × 1 matrix)
    - Test single-layer network
    - Test sigmoid default when `acts` omitted
    - Test identity activation produces linear output
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 3. Checkpoint — Verify batched feedforward
  - Ensure all existing 130 tests plus new activation/feedforward tests pass, ask the user if questions arise.

- [x] 4. Implement cloud packing utilities
  - [x] 4.1 Create `src/lotes.jl` with `empaquetar_pesos` and `reempaquetar_pesos`
    - `empaquetar_pesos(nube, T)` packs weights into `Array{T,3}` of shape `(neurons_out, neurons_in, N)` per layer and biases into `(neurons_out, 1, N)`
    - `reempaquetar_pesos(nube, indices, W3ds_old, B3ds_old, T)` re-packs a subset of networks after topology reduction
    - Include `lotes.jl` in `src/RandomCloud.jl`
    - _Requirements: 2.1, 2.4_

  - [x] 4.2 Write property test for weight packing round-trip (Property 2)
    - **Property 2: Weight packing round-trip**
    - Generate random cloud of N networks with same topology
    - Pack via `empaquetar_pesos`, extract slice `i`, assert bitwise equality with original weights
    - Use Supposition.jl with min 100 iterations
    - **Validates: Requirements 2.1**

  - [x] 4.3 Write property test for re-packing subset preservation (Property 4)
    - **Property 4: Re-packing preserves remaining networks**
    - Generate random cloud, pick random subset of indices, re-pack, assert slice `j` matches `nube[indices[j]]`
    - Use Supposition.jl with min 100 iterations
    - **Validates: Requirements 2.4**

- [x] 5. Implement batched cloud evaluation
  - [x] 5.1 Add `evaluar_nube_batch` to `src/evaluacion.jl`
    - Implement batched evaluation of all N networks using packed 3-D weight tensors
    - Use `feedforward_batch` per network (loop over N, batched over samples) as the default strategy
    - Compute per-network accuracy by comparing `argmax` of output vs target
    - Accept `AbstractMatrix{T}` for CPU/GPU compatibility
    - _Requirements: 2.2, 2.3_

  - [x] 5.2 Write property test for cloud evaluation accuracy equivalence (Property 3)
    - **Property 3: Cloud evaluation accuracy equivalence**
    - Generate random cloud, random data, compare `evaluar_nube_batch` per-network accuracy with individual `evaluar` calls
    - Assert absolute difference ≤ 1e-10
    - Use Supposition.jl with min 100 iterations
    - **Validates: Requirements 2.2, 2.3**

- [x] 6. Implement batched backpropagation
  - [x] 6.1 Add `entrenar_batch_matmul!` to `src/red_neuronal.jl`
    - Implement full-batch matrix backpropagation: forward pass storing activations, backward pass computing `δ`, `∇W`, `∇b` as matrix ops
    - Use `aplicar_derivada_batch` for derivative computation
    - Average gradients over batch dimension before weight update
    - Accept `AbstractMatrix{T}` for CPU/GPU dispatch
    - _Requirements: 4.1, 4.2, 4.3, 4.5_

  - [x] 6.2 Write property test for batched backprop weight-update equivalence (Property 6)
    - **Property 6: Batched backprop weight-update equivalence**
    - Generate random network, random mini-batch, random learning rate
    - Compare weights after `entrenar_batch_matmul!` vs sequential `entrenar!` per sample
    - Assert absolute difference ≤ 1e-8
    - Use Supposition.jl with min 100 iterations
    - **Validates: Requirements 4.2, 4.3**

- [x] 7. Checkpoint — Verify core batched operations
  - Ensure all existing 130 tests plus new batched operation tests pass, ask the user if questions arise.

- [x] 8. Modify ConfiguracionNube and InformeNube
  - [x] 8.1 Add `gpu::Bool` field to `ConfiguracionNube` in `src/configuracion.jl`
    - Add `gpu::Bool` field defaulting to `false`
    - Add module-level `const GPU_AVAILABLE = Ref(false)` in `src/RandomCloud.jl`
    - Validate: if `gpu=true` and `!GPU_AVAILABLE[]`, throw `ArgumentError("CUDA.jl must be added to use gpu=true. Run: ] add CUDA")`
    - Preserve all existing constructor signatures and defaults
    - _Requirements: 5.1, 5.2, 6.4_

  - [x] 8.2 Add `gpu_tiempo_ms` and `pico_vram_mb` fields to `InformeNube` in `src/informe.jl`
    - Add two `Float64` fields defaulting to `0.0`
    - Provide backward-compatible constructor that sets new fields to `0.0` when not specified
    - _Requirements: 7.3, 7.4_

  - [x] 8.3 Write property test for ConfiguracionNube backward compatibility (Property 7)
    - **Property 7: ConfiguracionNube backward compatibility**
    - Generate random valid keyword arguments without specifying `gpu`
    - Assert construction succeeds, `gpu == false`, all other fields match provided values
    - Use Supposition.jl with min 100 iterations
    - **Validates: Requirements 5.1, 5.2**

  - [x] 8.4 Write unit tests for ConfiguracionNube and InformeNube modifications
    - Test `gpu=true` without CUDA raises `ArgumentError`
    - Test `MotorNube(config, X, Y)` 2-arg constructor still works
    - Test `MotorNube(config, X, Y, fn)` 3-arg constructor still works
    - Test `InformeNube` backward-compatible constructor defaults new fields to 0.0
    - _Requirements: 5.3, 6.4_

- [x] 9. Modify ejecutar flow with dispatch logic
  - [x] 9.1 Refactor `ejecutar` in `src/motor.jl` to dispatch between legacy, batched CPU, and GPU paths
    - Extract existing logic into `_ejecutar_legacy(motor)`
    - Add `_ejecutar_batched(motor)` using `feedforward_batch`, `evaluar_nube_batch`, `entrenar_batch_matmul!`
    - Add `_ejecutar_gpu(motor)` stub that will be defined by the GPU extension
    - Dispatch: `gpu=true` → GPU path, `_usar_batched(config)` → batched CPU, else → legacy
    - Wire cloud packing (`empaquetar_pesos`, `reempaquetar_pesos`) into the batched path
    - _Requirements: 3.4, 5.4_

  - [x] 9.2 Write property test for end-to-end ejecutar equivalence (Property 8)
    - **Property 8: End-to-end ejecutar equivalence**
    - Generate valid `ConfiguracionNube` with `gpu=false` and fixed seed, random input/target matrices
    - Compare `InformeNube.precision` within ±0.01 and `exitoso` identical between batched and legacy paths
    - Use Supposition.jl with min 100 iterations
    - **Validates: Requirements 5.4**

- [x] 10. Checkpoint — Verify full CPU integration
  - Ensure all 130 existing tests plus all new tests pass with the refactored ejecutar flow, ask the user if questions arise.

- [x] 11. Implement GPU extension
  - [x] 11.1 Set up package extension in `Project.toml` and `src/RandomCloud.jl`
    - Add `[weakdeps]` section with `CUDA = "..."` in `Project.toml`
    - Add `[extensions]` section mapping `RandomCloudCUDAExt = "CUDA"` in `Project.toml`
    - Add extension hooks in `src/RandomCloud.jl` (export `GPU_AVAILABLE`, define stubs for `a_gpu`, `de_gpu`, `estimar_vram`, `verificar_gpu`)
    - _Requirements: 6.1, 6.2, 6.3_

  - [x] 11.2 Create `ext/RandomCloudCUDAExt/RandomCloudCUDAExt.jl` entry point
    - Import CUDA.jl and RandomCloud
    - Set `RandomCloud.GPU_AVAILABLE[] = true` on load
    - Include `gpu_backend.jl`
    - _Requirements: 6.1, 6.3_

  - [x] 11.3 Create `ext/RandomCloudCUDAExt/gpu_backend.jl` with GPU utilities
    - Implement `a_gpu(x)`: convert `Float64 → Float32`, transfer to `CuArray`
    - Implement `de_gpu(x)`: transfer `CuArray` to host, convert `Float32 → Float64`
    - Implement `estimar_vram(config, entradas)`: compute memory estimate with 20% overhead margin
    - Implement `verificar_gpu()`: check `CUDA.functional()`, throw informative error if not
    - _Requirements: 3.1, 3.5, 3.6_

  - [x] 11.4 Implement `_ejecutar_gpu` in the GPU extension
    - Call `verificar_gpu()` and `estimar_vram()` at start (fail fast)
    - Transfer data and packed weights to GPU via `a_gpu`
    - Run batched exploration and refinement on device
    - Transfer only final `InformeNube` results back via `de_gpu`
    - Record `gpu_tiempo_ms` and `pico_vram_mb` in `InformeNube`
    - Implement adaptive batching strategy (`_elegir_estrategia`) for VRAM management
    - _Requirements: 3.1, 3.2, 3.3, 3.5, 3.6, 4.4_

  - [x] 11.5 Write property test for VRAM estimation monotonicity (Property 5)
    - **Property 5: VRAM estimation monotonicity**
    - Generate two configs differing only in `tamano_nube` (c1 < c2), assert `estimar_vram(c2) > estimar_vram(c1)`
    - Generate two input matrices with different sample counts, assert larger produces higher estimate
    - Use Supposition.jl with min 100 iterations
    - **Validates: Requirements 3.6**

  - [x] 11.6 Write unit tests for GPU extension error handling
    - Test `estimar_vram` raises error for oversized config (> 3.5 GB)
    - Test `verificar_gpu` raises informative error when no device available
    - Test `a_gpu`/`de_gpu` round-trip preserves values (within Float32 precision)
    - _Requirements: 3.3, 3.6_

- [x] 12. Checkpoint — Verify GPU extension loads correctly
  - Ensure all 130 existing tests pass, GPU extension loads without errors when CUDA.jl is present, and CPU-only path works when CUDA.jl is absent. Ask the user if questions arise.

- [x] 13. GPU-specific integration tests
  - [x] 13.1 Write GPU integration tests gated behind `CUDA.functional()`
    - Test GPU feedforward matches CPU feedforward (Float32 tolerance: 1e-5)
    - Test GPU cloud evaluation matches CPU cloud evaluation
    - Test GPU backprop produces similar weight updates (Float32 tolerance: 1e-4)
    - Test VRAM estimation matches actual allocation within 20%
    - Test full `ejecutar` with `gpu=true` completes on small problems
    - Skip all tests with `@info` message when CUDA is not available
    - _Requirements: 3.1, 3.2, 4.4_

- [x] 14. Validation suite on 7 standard datasets
  - [x] 14.1 Create validation script comparing batched CPU vs legacy vs GPU
    - Run on Iris, Wine, Breast Cancer, Ionosphere, Sonar, Digits, Adult
    - Use same seed for all three modes
    - Assert batched CPU accuracy within ±1 percentage point of legacy per dataset
    - Assert GPU accuracy within ±1 percentage point of batched CPU per dataset (when CUDA available)
    - Report wall-clock time for each mode and dataset
    - Report peak GPU memory usage when running in GPU mode
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 15. Scalability testing on larger datasets
  - [x] 15.1 Create scalability harness for MNIST, Fashion-MNIST, and CIFAR-10
    - Test GPU implementation completes within 4 GB VRAM on MNIST and Fashion-MNIST
    - Test CIFAR-10 either completes or raises memory-limit error per Requirement 3.6
    - Record and report: accuracy, total wall-clock time, GPU time fraction, peak VRAM usage
    - Compare GPU vs CPU wall-clock time and report speedup factor
    - Gate behind `CUDA.functional()` with informative skip message
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 16. Final checkpoint — Full regression and integration
  - Ensure all 130 existing tests pass unchanged, all new property and unit tests pass, validation suite results are within tolerance. Ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key integration points
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples, edge cases, and error conditions
- GPU tests are gated behind `CUDA.functional()` and skip gracefully on CPU-only machines
- All code uses `T<:AbstractFloat` generics to support both Float64 (CPU) and Float32 (GPU)
