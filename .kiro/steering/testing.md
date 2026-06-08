---
inclusion: fileMatch
fileMatchPattern: "test/**/*.jl"
---

# Testing — RandomCloud

## Framework
- Julia `Test` stdlib + Supposition.jl (property-based testing) + BenchmarkTools

## Run
- `julia --project=. -e 'using Pkg; Pkg.test()'`

## Conventions
- Test files in `test/` directory
- `runtests.jl` as entry point
- Property-based tests via Supposition.jl for numerical invariants
- Benchmark regressions via BenchmarkTools in test targets
- Use `@testset` blocks for grouping
- Seed RNG explicitly for reproducibility
