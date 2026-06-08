# Plan de inicialización: RandomCloud.jl

## Estructura del repositorio

```
RandomCloud.jl/
├── Project.toml              # Dependencias del paquete Julia
├── LICENSE                   # MIT o Apache 2.0
├── README.md                 # Descripción del método, instalación, uso
├── src/
│   ├── RandomCloud.jl        # Módulo principal (exports)
│   ├── configuracion.jl      # ConfiguracionNube (struct inmutable)
│   ├── red_neuronal.jl       # Red neuronal feedforward mínima
│   ├── politica.jl           # Interfaz de política + PoliticaSecuencial
│   ├── motor.jl              # MotorNube (orquestador)
│   ├── evaluacion.jl         # Funciones de evaluación (argmax, precisión)
│   └── informe.jl            # InformeNube (struct inmutable)
├── test/
│   ├── runtests.jl           # Entry point de tests
│   ├── test_configuracion.jl # Tests de ConfiguracionNube
│   ├── test_politica.jl      # Tests de políticas de eliminación
│   ├── test_motor.jl         # Tests del motor (XOR)
│   ├── test_integracion.jl   # Test de flujo completo
│   └── benchmark_escalabilidad.jl  # Benchmark comparativo
├── docs/
│   └── metodo.md             # Documento formal del método (para registro PI)
└── examples/
    └── xor.jl                # Ejemplo básico de uso
```

## Paso a paso para inicializar

### 1. Crear el paquete Julia

```bash
julia -e 'using Pkg; Pkg.generate("RandomCloud")'
cd RandomCloud
git init
```

### 2. Añadir dependencias en Project.toml

Las únicas dependencias necesarias son:

```toml
[deps]
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"

[targets]
test = ["Test", "BenchmarkTools"]
```

Random y LinearAlgebra son de la stdlib, no necesitan instalación.

### 3. Módulo principal — src/RandomCloud.jl

```julia
module RandomCloud

export ConfiguracionNube, configuracion_defecto
export PoliticaEliminacion, PoliticaSecuencial, siguiente_reduccion
export MotorNube, ejecutar
export InformeNube

include("configuracion.jl")
include("red_neuronal.jl")
include("politica.jl")
include("evaluacion.jl")
include("motor.jl")
include("informe.jl")

end
```

### 4. Componentes principales

#### src/configuracion.jl

```julia
"""
Configuración inmutable del Método de la Nube Aleatoria.
"""
struct ConfiguracionNube
    tamano_nube::Int
    topologia_inicial::Vector{Int}
    umbral_acierto::Float64
    neuronas_eliminar::Int
    epocas_refinamiento::Int
    tasa_aprendizaje::Float64
    semilla::Int

    function ConfiguracionNube(;
        tamano_nube::Int = 10,
        topologia_inicial::Vector{Int} = [2, 4, 1],
        umbral_acierto::Float64 = 0.5,
        neuronas_eliminar::Int = 1,
        epocas_refinamiento::Int = 1000,
        tasa_aprendizaje::Float64 = 0.1,
        semilla::Int = 42
    )
        tamano_nube < 1 && throw(ArgumentError(
            "El tamaño de la nube debe ser al menos 1 (valor: $tamano_nube)"))
        length(topologia_inicial) < 3 && throw(ArgumentError(
            "La topología debe tener al menos 3 capas (entrada, oculta, salida)"))
        !(0.0 <= umbral_acierto <= 1.0) && throw(ArgumentError(
            "El umbral de acierto debe estar en [0.0, 1.0] (valor: $umbral_acierto)"))
        neuronas_eliminar < 1 && throw(ArgumentError(
            "El número de neuronas a eliminar debe ser al menos 1 (valor: $neuronas_eliminar)"))
        for i in 2:length(topologia_inicial)-1
            topologia_inicial[i] < 1 && throw(ArgumentError(
                "Las capas ocultas deben tener al menos 1 neurona"))
        end
        new(tamano_nube, copy(topologia_inicial), umbral_acierto,
            neuronas_eliminar, epocas_refinamiento, tasa_aprendizaje, semilla)
    end
end
```

#### src/red_neuronal.jl

```julia
using Random, LinearAlgebra

"""
Red neuronal feedforward mínima.
"""
struct RedNeuronal
    topologia::Vector{Int}
    pesos::Vector{Matrix{Float64}}
    biases::Vector{Vector{Float64}}
end

function RedNeuronal(topologia::Vector{Int}, rng::AbstractRNG)
    pesos = [2.0 .* rand(rng, topologia[i+1], topologia[i]) .- 1.0
             for i in 1:length(topologia)-1]
    biases = [2.0 .* rand(rng, topologia[i+1]) .- 1.0
              for i in 1:length(topologia)-1]
    RedNeuronal(copy(topologia), pesos, biases)
end

function feedforward(red::RedNeuronal, entrada::Vector{Float64})
    x = entrada
    for i in eachindex(red.pesos)
        x = sigmoid.(red.pesos[i] * x .+ red.biases[i])
    end
    x
end

sigmoid(x) = 1.0 / (1.0 + exp(-x))
sigmoid_deriv(x) = x * (1.0 - x)

function entrenar!(red::RedNeuronal, entrada::Vector{Float64},
                   objetivo::Vector{Float64}, lr::Float64)
    # Forward pass guardando activaciones
    activaciones = Vector{Vector{Float64}}(undef, length(red.topologia))
    activaciones[1] = entrada
    for i in eachindex(red.pesos)
        activaciones[i+1] = sigmoid.(red.pesos[i] * activaciones[i] .+ red.biases[i])
    end

    # Backpropagation
    error = objetivo .- activaciones[end]
    for i in length(red.pesos):-1:1
        gradiente = error .* sigmoid_deriv.(activaciones[i+1]) .* lr
        red.pesos[i] .+= gradiente * activaciones[i]'
        red.biases[i] .+= gradiente
        if i > 1
            error = red.pesos[i]' * error
        end
    end
end
```

#### src/politica.jl

```julia
"""
Interfaz abstracta para políticas de eliminación.
"""
abstract type PoliticaEliminacion end

"""
Política secuencial: elimina de la última capa oculta hacia la primera.
"""
struct PoliticaSecuencial <: PoliticaEliminacion end

function siguiente_reduccion(::PoliticaSecuencial, topologia::Vector{Int}, n::Int)
    # Verificar si todas las capas ocultas están vacías
    all(topologia[i] == 0 for i in 2:length(topologia)-1) && return nothing

    nueva = copy(topologia)
    # Buscar última capa oculta con neuronas > 0
    for i in length(nueva)-1:-1:2
        if nueva[i] > 0
            nueva[i] = max(0, nueva[i] - n)
            break
        end
    end

    # Si todas quedaron en 0, retornar nothing
    all(nueva[i] == 0 for i in 2:length(nueva)-1) && return nothing
    nueva
end
```

#### src/evaluacion.jl

```julia
function evaluar(red::RedNeuronal, entradas::Matrix{Float64}, objetivos::Matrix{Float64})
    m = size(entradas, 2)
    correctas = 0
    for k in 1:m
        salida = feedforward(red, entradas[:, k])
        if argmax(salida) == argmax(objetivos[:, k])
            correctas += 1
        end
    end
    correctas / m
end
```

#### src/informe.jl

```julia
"""
Informe inmutable con los resultados del método.
"""
struct InformeNube
    mejor_red::Union{RedNeuronal, Nothing}
    precision::Float64
    topologia_final::Union{Vector{Int}, Nothing}
    total_redes_evaluadas::Int
    total_reducciones::Int
    tiempo_ejecucion_ms::Float64
    exitoso::Bool
end
```

### 5. Tests — test/runtests.jl

```julia
using Test
using RandomCloud

include("test_configuracion.jl")
include("test_politica.jl")
include("test_motor.jl")
include("test_integracion.jl")
```

### 6. Ejemplo de uso — examples/xor.jl

```julia
using RandomCloud

# Dataset XOR (columnas = muestras)
entradas = [0.0 0.0 1.0 1.0;
            0.0 1.0 0.0 1.0]
objetivos = [1.0 0.0 0.0 1.0;
             0.0 1.0 1.0 0.0]

config = ConfiguracionNube(
    tamano_nube = 10,
    topologia_inicial = [2, 8, 4, 2],
    umbral_acierto = 0.25,
    neuronas_eliminar = 1,
    epocas_refinamiento = 2000,
    tasa_aprendizaje = 0.5,
    semilla = 42
)

motor = MotorNube(config, entradas, objetivos)
informe = ejecutar(motor)

if informe.exitoso
    println("Topología encontrada: $(informe.topologia_final)")
    println("Precisión: $(round(informe.precision * 100, digits=2))%")
    println("Parámetros: reducidos de $(sum(prod, zip(config.topologia_inicial[1:end-1], config.topologia_inicial[2:end]))) a $(sum(prod, zip(informe.topologia_final[1:end-1], informe.topologia_final[2:end])))")
else
    println("No se encontró red viable")
end
```

### 7. README.md sugerido

```markdown
# RandomCloud.jl

Implementación en Julia del Método de la Nube Aleatoria: búsqueda automática
de arquitecturas de redes neuronales mediante evaluación sin entrenamiento
y reducción estructural progresiva.

## Instalación

    julia> using Pkg
    julia> Pkg.add(url="https://github.com/TU_USUARIO/RandomCloud.jl")

## Uso rápido

    using RandomCloud

    config = ConfiguracionNube(
        tamano_nube = 10,
        topologia_inicial = [2, 8, 4, 2],
        umbral_acierto = 0.25,
        semilla = 42
    )

    motor = MotorNube(config, entradas, objetivos)
    informe = ejecutar(motor)

## Documentación

Ver [docs/metodo.md](docs/metodo.md) para la descripción formal del método.

## Licencia

MIT
```

### 8. Orden de implementación recomendado

1. `configuracion.jl` + `test_configuracion.jl`
2. `red_neuronal.jl` (feedforward + entrenar!)
3. `politica.jl` + `test_politica.jl`
4. `evaluacion.jl`
5. `informe.jl`
6. `motor.jl` + `test_motor.jl`
7. `test_integracion.jl` (flujo completo XOR)
8. `benchmark_escalabilidad.jl`
9. Copiar `docs/metodo.md` desde este repo

## Notas sobre Julia vs Java

- En Julia los structs son inmutables por defecto (como los records de Java).
- No necesitas builder: los keyword arguments del constructor hacen lo mismo.
- Las matrices en Julia son column-major: las muestras van en columnas, no en filas.
- El multiple dispatch de Julia reemplaza la interfaz funcional: defines `siguiente_reduccion` para cada subtipo de `PoliticaEliminacion`.
- Julia tiene broadcasting nativo (`.+`, `.*`) que simplifica las operaciones matriciales.
- `BenchmarkTools.jl` es el equivalente a JMH para benchmarks serios.
