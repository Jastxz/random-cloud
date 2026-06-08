---
inclusion: always
---

# Architecture — RandomCloud

## Stack
- **Language:** Julia 1.9+
- **Build:** Pkg (Project.toml / Manifest.toml)
- **Key deps:** CairoMakie, DataFrames, MLDatasets, BenchmarkTools, SpecialFunctions
- **GPU:** Optional CUDA/AMDGPU via extensions

## Module Boundaries (from graphify: 374 nodes, 390 edges, 51 communities)
- God nodes: `correr_significancia()` (9 edges), `medir_tiempos()` (7), `normalizar!()` (7), `comparar_metodos()` (6), `_descargar_cache()` (6)
- `src/` — RandomCloud library (stochastic neural network with random cloud topology)
  - `normalizar!()` — In-place data normalization (core preprocessing)
  - `correr_significancia()` — Statistical significance testing
  - `medir_tiempos()` — Performance timing harness
  - `comparar_metodos()` — Method comparison framework
  - `_descargar_cache()` — Dataset download/caching
- `ext/` — GPU extensions (RandomCloudCUDAExt, RandomCloudAMDGPUExt)
- `test/` — Unit tests with Supposition (property-based)
- `examples/` — Usage examples and demos
- `docs/` — Documentation
- `paper/` — Related academic paper
- `figures/` — Generated plots and visualizations

## Dependency Rules
- Core `src/` uses only LinearAlgebra + Random + SpecialFunctions — no plotting
- CairoMakie, DataFrames, MLDatasets are for examples/scripts only
- GPU backends via Julia extension mechanism — weak deps in Project.toml
- BenchmarkTools for performance testing, not production code

## Design Principles
- Library designed for both CPU and GPU via KernelAbstractions pattern
- Extensions load lazily — `using CUDA` activates GPU path automatically
- Core algorithms must be type-stable and allocation-free in hot paths
