---
inclusion: always
---

# Coding Standards — RandomCloud (Julia)

## Julia
- Functions: lowercase_with_underscores; Types: PascalCase
- Multiple dispatch — define methods on concrete types
- Explicit type annotations on struct fields
- Avoid type piracy — don't extend other packages' methods on their types
- Minimize allocations in numerical kernels

## Performance
- Type stability is mandatory — check with `@code_warntype`
- GPU kernels via KernelAbstractions (portable across CUDA/AMDGPU)
- Use `@inbounds` only with proven bounds
- Benchmark regressions tracked in `test/` via BenchmarkTools

## Package
- `Project.toml`: define [compat] bounds for all dependencies
- Extensions in `ext/` — one file per backend
- `Manifest.toml` committed for reproducibility

## Testing
- `julia --project=. -e 'using Pkg; Pkg.test()'`
- Property-based testing via Supposition.jl
