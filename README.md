# RandomCloud.jl

Implementación en Julia del **Método de la Nube Aleatoria**, un enfoque de búsqueda de arquitecturas de redes neuronales (NAS) que encuentra topologías mínimas mediante evaluación sin entrenamiento y reducción estructural progresiva.

## El método en una frase

Generar muchas redes con pesos aleatorios, evaluar cuál clasifica mejor *sin entrenarla*, reducir progresivamente su topología, y refinar la mejor con backpropagation al final.

## Instalación

```julia
using Pkg
Pkg.develop(path=".")
```

## Uso rápido

```julia
using RandomCloud

# Dataset XOR
entradas = [0.0 0.0 1.0 1.0; 0.0 1.0 0.0 1.0]
objetivos = [1.0 0.0 0.0 1.0; 0.0 1.0 1.0 0.0]

config = ConfiguracionNube(
    tamano_nube = 50,
    topologia_inicial = [2, 8, 4, 2],
    umbral_acierto = 0.5,
    epocas_refinamiento = 2000,
    tasa_aprendizaje = 0.5,
    semilla = 42
)

motor = MotorNube(config, entradas, objetivos)
informe = ejecutar(motor)

println("Exitoso: $(informe.exitoso)")
println("Precisión: $(informe.precision * 100)%")
println("Topología final: $(informe.topologia_final)")
```

## Resultados experimentales

### XOR

| Métrica | Clásico [2,8,4,2] | Nube Aleatoria |
|---|---|---|
| Topología final | [2,8,4,2] | [2,8,3,2] |
| Parámetros | 70 | 59 |
| Precisión | 100% | 100% |
| Reducción | — | 15.7% |

### MNIST (dígitos manuscritos 0-9)

Comparativa con la misma topología inicial, mismas épocas de entrenamiento (30), mismo learning rate (0.1) y misma semilla. La red clásica entrena 30 épocas con backpropagation desde pesos aleatorios. La nube usa esas 30 épocas solo para el refinamiento final. Test set fijo de 10,000 imágenes.

**Configuración de la nube:** 100 redes, umbral=0.15, eliminar=2 neuronas/paso.

| Muestras | Clásico (train) | Clásico (test) | Nube (train) | Nube (test) | Topología final | Reducción params | Tiempo clásico | Tiempo nube |
|----------|----------------|---------------|-------------|------------|----------------|-----------------|---------------|------------|
| 1,000 | 97.2% | 84.6% | 68.5% | 59.8% | [784,32,4,10] | -2.0% | 1.1s | 10.1s |
| 5,000 | 97.3% | 90.7% | 94.7% | 86.9% | [784,32,4,10] | -2.0% | 5.7s | 57.2s |
| 10,000 | 97.8% | 92.5% | 96.4% | 90.2% | [784,32,4,10] | -2.0% | 12.6s | 108.8s |
| 30,000 | 97.8% | 94.2% | 96.6% | 92.6% | [784,32,4,10] | -2.0% | 39.4s | 325.9s |
| 60,000 | 97.7% | 95.5% | 96.8% | 94.3% | [784,32,4,10] | -2.0% | 75.1s | 649.6s |

**Observaciones:**
- La brecha de precisión se cierra con más datos: de 24.8pp (1K) a 1.2pp (60K).
- La nube siempre reduce la segunda capa oculta de 16→4 neuronas (25,818→25,302 parámetros).
- La reducción de parámetros es modesta (2%) porque la capa 784→32 domina con 25,088 parámetros.
- El método es ~8-9x más lento por el overhead de exploración (2,400 evaluaciones sin entrenamiento).

### Iris (clasificación de flores, 3 clases)

150 muestras, 4 features, 3 clases. Split 80/20 estratificado (120 train, 30 test). 100 épocas, LR=0.1. Nube: 50 redes, umbral=0.4, eliminar=1. Hasta 3 reintentos con semillas distintas.

| Topología inicial | Params | Clásico (test) | Nube (test) | Topología final | Reducción params | Intentos |
|---|---|---|---|---|---|---|
| [4, 8, 3] | 67 | 100.0% | 96.7% | [4, 1, 3] | -83.6% | 1 |
| [4, 8, 4, 3] | 91 | 100.0% | 100.0% | [4, 8, 4, 3] | 0% | 2 |
| [4, 16, 8, 3] | 243 | 100.0% | 100.0% | [4, 16, 3, 3] | -41.2% | 1 |
| [4, 16, 8, 4, 3] | 267 | 100.0% | 100.0% | [4, 16, 8, 4, 3] | 0% | 2 |

**Observaciones:**
- Con [4, 16, 8, 3] la nube alcanza 100% test con 41% menos parámetros (243→143).
- Con [4, 8, 3] la nube reduce un 83.6% los parámetros (67→11): descubre que 1 neurona oculta basta.
- Las topologías profundas (3+ capas ocultas) necesitan un segundo intento con semilla distinta para encontrar una red viable sin entrenamiento.
- Cuando la nube no reduce (0%), significa que la topología original ya era cercana al mínimo para ese umbral.

## Estructura del proyecto

```
RandomCloud.jl/
├── Project.toml
├── src/
│   ├── RandomCloud.jl          # Módulo principal
│   ├── configuracion.jl        # ConfiguracionNube
│   ├── red_neuronal.jl         # RedNeuronal, feedforward, entrenar!, reconstruir
│   ├── politica.jl             # PoliticaEliminacion, PoliticaSecuencial
│   ├── evaluacion.jl           # evaluar
│   ├── motor.jl                # MotorNube, ejecutar
│   └── informe.jl              # InformeNube
├── test/
│   ├── runtests.jl
│   ├── test_configuracion.jl   # Unit + PBT
│   ├── test_red_neuronal.jl    # Unit + PBT
│   ├── test_politica.jl        # Unit + PBT
│   ├── test_evaluacion.jl      # Unit
│   ├── test_motor.jl           # Unit + PBT
│   ├── test_integracion.jl     # Integración
│   └── benchmark_escalabilidad.jl
├── examples/
│   ├── xor.jl                  # Ejemplo XOR
│   ├── mnist.jl                # Experimento MNIST
│   └── comparativa_mnist.jl    # Comparativa Nube vs Clásico
└── docs/
    └── metodo.md               # Descripción formal del método
```

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

117 tests: unitarios + 10 propiedades verificadas con property-based testing (Supposition.jl).

## Licencia

Ver [LICENSE](LICENSE).
