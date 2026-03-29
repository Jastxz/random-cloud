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
    semilla = 42,
    activacion = :sigmoid,       # :sigmoid (default), :relu, :identidad
    batch_size = 0               # 0 = SGD sample-by-sample (default)
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

### Wine (clasificación de vinos, 3 clases)

178 muestras, 13 features, 3 clases. Split 80/20 estratificado (142 train, 36 test). 100 épocas, LR=0.1. Nube: 50 redes, umbral=0.4, eliminar=1.

| Topología inicial | Params | Clásico (test) | Nube (test) | Topología final | Reducción params |
|---|---|---|---|---|---|
| [13, 8, 3] | 139 | 94.4% | 94.4% | [13, 5, 3] | -36.7% |
| [13, 16, 3] | 275 | 94.4% | 94.4% | [13, 7, 3] | -55.6% |
| [13, 16, 8, 3] | 387 | 94.4% | 94.4% | [13, 16, 5, 3] | -15.5% |
| [13, 32, 16, 3] | 1,027 | 91.7% | 94.4% | [13, 32, 13, 3] | -10.5% |

**Observaciones:**
- La nube iguala o supera al clásico en todos los casos, con menos parámetros.
- Con [13, 16, 3] la nube reduce un 55.6% los parámetros (275→122) sin perder precisión.
- Con [13, 32, 16, 3] la nube supera al clásico (94.4% vs 91.7%) con 10.5% menos parámetros.
- Todos los experimentos exitosos al primer intento (semilla=42).

### Fashion-MNIST (prendas de ropa, 10 clases)

Misma estructura que MNIST (784 entradas, 10 clases) pero más difícil. Topología [784, 64, 32, 10] (52,650 params). 30 épocas, LR=0.1. Nube: 50 redes, umbral=0.12, eliminar=2. Test set fijo de 10,000 imágenes.

| Muestras | Clásico (test) | Nube (test) | Topología final | Reducción params |
|----------|---------------|------------|----------------|-----------------|
| 1,000 | 77.9% | 77.0% | [784, 64, 24, 10] | -1.1% |
| 5,000 | 82.6% | 81.7% | [784, 64, 24, 10] | -1.1% |
| 10,000 | 83.2% | 83.1% | [784, 64, 24, 10] | -1.1% |
| 60,000 | 86.3% | 86.3% | [784, 64, 24, 10] | -1.1% |

**Observaciones:**
- La nube iguala al clásico con 60K muestras (86.3% ambos) con 1.1% menos parámetros.
- La brecha se cierra con más datos: de 0.9pp (1K) a 0pp (60K).
- La nube siempre reduce la segunda capa oculta de 32→24 neuronas.
- Comportamiento muy similar a MNIST pero con precisiones ~10pp menores (problema más difícil).

### CIFAR-10 (imágenes a color, 10 clases)

32×32×3 = 3,072 entradas, 10 clases. 30 épocas, LR=0.05. Nube: 50 redes, umbral=0.12, eliminar=2. 8 threads.

**5,000 muestras de train, 10,000 de test:**

| Topología inicial | Params | Clásico (test) | Nube (test) | Topología final | Reducción params |
|---|---|---|---|---|---|
| [3072, 64, 10] | 197,322 | 31.6% | 29.9% | [3072, 24, 10] | -62.5% |
| [3072, 64, 32, 10] | 199,082 | 32.5% | 34.0% | [3072, 64, 32, 10] | 0% |

**50,000 muestras de train (dataset completo), 10,000 de test:**

| Topología inicial | Params | Clásico (test) | Nube (test) | Topología final | Reducción params | Tiempo clásico | Tiempo nube |
|---|---|---|---|---|---|---|---|
| [3072, 64, 10] | 197,322 | 41.6% | 37.3% | [3072, 24, 10] | -62.5% | 8.2 min | 20.6 min |
| [3072, 64, 32, 10] | 199,082 | 42.5% | 41.8% | [3072, 64, 32, 10] | 0% | 8.1 min | 47.3 min |

**Observaciones:**
- CIFAR-10 es un problema difícil para redes feedforward sin convoluciones (~40% con estas topologías).
- Con [3072, 64, 10] la nube reduce un 62.5% los parámetros (197K→74K) con 4.3pp de pérdida.
- Con [3072, 64, 32, 10] la nube casi iguala al clásico (41.8% vs 42.5%) sin reducir topología.
- El overhead de exploración es significativo en problemas de alta dimensionalidad (~2.5x más lento que el clásico).

### Optical Digits (dígitos 8×8, 10 clases)

Imágenes 8×8 = 64 features, 10 clases. 3,823 train + 1,797 test (split predefinido UCI). 100 épocas, LR=0.1. Nube: 50 redes, umbral=0.15, eliminar=2.

| Topología inicial | Params | Clásico (test) | Nube (test) | Topología final | Reducción params |
|---|---|---|---|---|---|
| [64, 16, 10] | 1,210 | 95.8% | 95.5% | [64, 14, 10] | -12.4% |
| [64, 32, 10] | 2,410 | 96.3% | 95.9% | [64, 12, 10] | -62.2% |
| [64, 32, 16, 10] | 2,778 | 96.5% | 96.3% | [64, 32, 8, 10] | -12.4% |
| [64, 64, 32, 10] | 6,570 | 96.8% | 96.4% | [64, 64, 14, 10] | -20.5% |

**Observaciones:**
- La nube se acerca mucho al clásico (0.2-0.4pp de diferencia) con reducciones de 12-62%.
- Con [64, 32, 10] la nube reduce un 62.2% los parámetros (2,410→912) perdiendo solo 0.4pp.
- Puente entre datasets tabulares y MNIST: 64 features vs 784, con resultados >95% en ambos métodos.

### Sonar (minas vs rocas, 2 clases)

208 muestras, 60 features, 2 clases. Ratio features/muestras muy alto (60/208). Split 80/20 estratificado (167 train, 41 test). 200 épocas, LR=0.1. Nube: 50 redes, umbral=0.55, eliminar=1.

| Topología inicial | Params | Clásico (test) | Nube (test) | Topología final | Reducción params |
|---|---|---|---|---|---|
| [60, 8, 2] | 506 | 78.0% | 80.5% | [60, 1, 2] | -87.2% |
| [60, 16, 2] | 1,010 | 80.5% | 75.6% | [60, 10, 2] | -37.4% |
| [60, 32, 2] | 2,018 | 80.5% | 78.0% | [60, 16, 2] | -50.0% |
| [60, 16, 8, 2] | 1,130 | 78.0% | 75.6% | [60, 16, 1, 2] | -11.8% |
| [60, 32, 16, 2] | 2,514 | 78.0% | 75.6% | [60, 32, 13, 2] | -4.2% |

**Observaciones:**
- Con [60, 8, 2] la nube supera al clásico (80.5% vs 78.0%) con 87.2% menos parámetros: descubre que 1 sola neurona oculta basta.
- Sonar tiene el ratio features/muestras más alto de todos los datasets (60/208), lo que hace la búsqueda sin entrenamiento más difícil.
- En topologías más grandes la nube sacrifica 2-5pp a cambio de reducciones significativas (37-50%).

### Ionosphere (señales de radar, 2 clases)

351 muestras, 34 features, 2 clases (good/bad radar returns). Split 80/20 estratificado (281 train, 70 test). 200 épocas, LR=0.1. Nube: 50 redes, umbral=0.6, eliminar=1.

| Topología inicial | Params | Clásico (test) | Nube (test) | Topología final | Reducción params |
|---|---|---|---|---|---|
| [34, 8, 2] | 298 | 87.1% | 85.7% | [34, 3, 2] | -62.1% |
| [34, 16, 2] | 594 | 94.3% | 90.0% | [34, 3, 2] | -81.0% |
| [34, 16, 8, 2] | 714 | 91.4% | 90.0% | [34, 16, 4, 2] | -10.6% |
| [34, 32, 16, 2] | 1,682 | 90.0% | 88.6% | [34, 32, 5, 2] | -22.9% |
| [34, 32, 16, 8, 2] | 1,802 | 91.4% | 90.0% | [34, 32, 16, 8, 2] | 0% |

**Observaciones:**
- La nube sacrifica 1-4pp de precisión a cambio de reducciones masivas de parámetros (hasta 81%).
- Con [34, 16, 2] la nube descubre que solo 3 neuronas ocultas bastan (594→113 parámetros, -81%).
- Ionosphere tiene un ratio features/muestras alto (34/351), lo que dificulta la búsqueda sin entrenamiento.
- El trade-off precisión vs compresión es claro: la nube prioriza arquitecturas mínimas.

### Adult Income (predicción de ingresos, 2 clases)

Problema real: predecir si una persona gana >50K$/año. 30,162 train, 15,060 test, 104 features (6 numéricas + one-hot categóricas), 2 clases. Desbalanceo moderado (24.9% >50K). 30 épocas, LR=0.1. Nube: 50 redes, umbral=0.6, eliminar=2.

| Topología inicial | Params | Clásico (test) | Nube (test) | Topología final | Reducción params |
|---|---|---|---|---|---|
| [104, 16, 2] | 1,714 | 84.2% | 85.0% | [104, 8, 2] | -49.9% |
| [104, 32, 2] | 3,426 | 84.1% | 84.4% | [104, 16, 2] | -50.0% |
| [104, 32, 16, 2] | 3,922 | 84.0% | 84.3% | [104, 32, 16, 2] | 0% |
| [104, 64, 32, 2] | 8,866 | 83.7% | 83.8% | [104, 64, 10, 2] | -16.6% |

**Observaciones:**
- La nube supera al clásico en los 4 escenarios con un problema real de datos tabulares.
- Con [104, 16, 2] la nube reduce un 50% los parámetros (1,714→858) y mejora la precisión (+0.8pp).
- Con [104, 32, 2] la nube descubre que 16 neuronas bastan (reduce 50% de parámetros).
- Patrón consistente: la nube encuentra que la mitad de las neuronas ocultas son redundantes.

### Boston Housing (regresión, precio de vivienda)

506 muestras, 13 features, 1 salida (precio normalizado). Split 80/20 (405 train, 101 test). 500 épocas, LR=0.01. Nube: 100 redes, umbral=0.05, eliminar=1. Métrica: R² (coeficiente de determinación).

| Topología inicial | Params | Clásico (R² test) | Nube (R² test) | Topología final | Reducción params |
|---|---|---|---|---|---|
| [13, 8, 1] | 121 | 0.679 | 0.692 | [13, 8, 1] | 0% |
| [13, 16, 1] | 241 | 0.702 | 0.702 | [13, 13, 1] | -18.7% |
| [13, 16, 8, 1] | 369 | 0.685 | 0.696 | [13, 16, 4, 1] | -19.5% |
| [13, 32, 16, 1] | 993 | 0.693 | 0.705 | [13, 32, 16, 1] | 0% |
| [13, 64, 32, 1] | 3,009 | 0.744 | 0.725 | [13, 64, 29, 1] | -6.6% |
| [13, 64, 32, 16, 1] | 3,521 | 0.726 | 0.741 | [13, 64, 32, 7, 1] | -8.7% |
| [13, 128, 64, 1] | 10,113 | 0.748 | 0.723 | [13, 128, 22, 1] | -54.0% |
| [13, 128, 64, 32, 1] | 12,161 | 0.756 | 0.756 | [13, 128, 64, 16, 1] | -8.7% |

**Observaciones:**
- La nube iguala o supera al clásico en 6 de 8 escenarios.
- Con [13, 128, 64, 1] la nube reduce un 54% los parámetros (10,113→4,651) con solo 0.025 de pérdida en R².
- Con [13, 128, 64, 32, 1] la nube iguala al clásico (R²=0.756) con 8.7% menos parámetros.
- El R² máximo (~0.76) está limitado por la arquitectura feedforward con sigmoid, no por el método.
- Primer test exitoso del método en regresión usando `evaluar_regresion` (R²).

### Breast Cancer Wisconsin (diagnóstico, 2 clases)

569 muestras, 30 features, 2 clases (benigno/maligno). Split 80/20 estratificado (456 train, 113 test). 100 épocas, LR=0.1. Nube: 50 redes, umbral=0.7, eliminar=1.

| Topología inicial | Params | Clásico (test) | Nube (test) | Topología final | Reducción params |
|---|---|---|---|---|---|
| [30, 8, 2] | 266 | 97.3% | 97.3% | [30, 2, 2] | -74.4% |
| [30, 16, 2] | 530 | 97.3% | 97.3% | [30, 14, 2] | -12.5% |
| [30, 16, 8, 2] | 650 | 97.3% | 95.6% | [30, 16, 1, 2] | -20.5% |
| [30, 32, 16, 2] | 1,554 | 97.3% | 97.3% | [30, 32, 2, 2] | -31.5% |

**Observaciones:**
- Con [30, 8, 2] la nube reduce un 74.4% los parámetros (266→68): descubre que 2 neuronas ocultas bastan para clasificar cáncer de mama.
- En 3 de 4 escenarios la nube iguala al clásico con menos parámetros.
- Todos exitosos al primer intento.

### Two Moons (frontera no lineal, 2 clases)

Dataset sintético. 1,000 train, 500 test, 2 features, ruido=0.1. 200 épocas, LR=0.5. Nube: 50 redes, umbral=0.6, eliminar=1.

| Topología inicial | Params | Clásico (test) | Nube (test) | Topología final | Reducción params |
|---|---|---|---|---|---|
| [2, 4, 2] | 22 | 100.0% | 100.0% | [2, 3, 2] | -22.7% |
| [2, 8, 2] | 42 | 100.0% | 100.0% | [2, 6, 2] | -23.8% |
| [2, 8, 4, 2] | 70 | 100.0% | 100.0% | [2, 8, 3, 2] | -15.7% |
| [2, 16, 8, 2] | 202 | 99.8% | 99.8% | [2, 16, 7, 2] | -9.4% |
| [2, 16, 8, 4, 2] | 230 | 100.0% | 100.0% | [2, 16, 8, 3, 2] | -4.8% |

**Observaciones:**
- La nube iguala al clásico en todos los casos, con reducciones de 5-24% en parámetros.
- Todos exitosos al primer intento.
- El método descubre que [2,4,2] puede reducirse a [2,3,2] (3 neuronas ocultas bastan para Two Moons).

### Métricas adicionales: F1-Score y AUC-ROC

Accuracy puede ser engañosa en datasets desbalanceados. F1-score (macro-averaged) mide el balance entre precision y recall por clase. AUC-ROC mide la calidad de las probabilidades de salida independientemente del umbral de decisión.

| Dataset | Método | Accuracy | F1 | AUC | Topología | Reducción |
|---|---|---|---|---|---|---|
| Breast Cancer | Clásico | 97.3% | 0.971 | 0.993 | [30,8,2] | — |
| | Nube | 97.3% | 0.971 | 0.992 | [30,2,2] | -74.4% |
| Sonar | Clásico | 78.0% | 0.776 | 0.823 | [60,8,2] | — |
| | Nube | 80.5% | 0.799 | 0.809 | [60,1,2] | -87.2% |
| Ionosphere | Clásico | 94.3% | 0.935 | 0.918 | [34,16,2] | — |
| | Nube | 90.0% | 0.885 | 0.910 | [34,3,2] | -81.0% |
| Adult Income | Clásico | 84.2% | 0.758 | 0.901 | [104,16,2] | — |
| | Nube | 85.0% | 0.782 | 0.904 | [104,8,2] | -49.9% |
| Iris | Clásico | 100.0% | 1.000 | 1.000 | [4,16,8,3] | — |
| | Nube | 100.0% | 1.000 | 1.000 | [4,16,3,3] | -41.2% |
| Wine | Clásico | 94.4% | 0.944 | 0.996 | [13,16,3] | — |
| | Nube | 94.4% | 0.944 | 0.995 | [13,7,3] | -55.6% |
| Opt. Digits | Clásico | 96.3% | 0.963 | 0.997 | [64,32,10] | — |
| | Nube | 95.9% | 0.959 | 0.997 | [64,12,10] | -62.2% |

**Hallazgos:**
- En Adult Income (75% clase mayoritaria), la accuracy del clásico (84.2%) oculta un F1 bajo (0.758) — clasifica peor la clase minoritaria (>50K). La nube mejora tanto accuracy (+0.8pp) como F1 (+0.024) con la mitad de parámetros.
- En Breast Cancer, la nube mantiene F1 y AUC idénticos al clásico (0.971 / 0.992) con 74% menos parámetros. La reducción topológica no degrada la calidad de las probabilidades.
- En Sonar, la nube supera al clásico en accuracy (+2.5pp) y F1 (+0.023) con 87% menos parámetros, aunque el AUC baja ligeramente (-0.014). Con 1 sola neurona oculta, las probabilidades son menos calibradas pero la clasificación argmax es mejor.
- En Ionosphere, la nube sacrifica 4.3pp de accuracy y 0.050 de F1 a cambio de 81% de compresión, pero el AUC se mantiene cercano (0.910 vs 0.918) — las probabilidades siguen siendo discriminativas.
- En los datasets multiclase (Iris, Wine, Digits), F1 y AUC se mantienen prácticamente idénticos entre nube y clásico, confirmando que la reducción topológica no afecta la calidad por clase.
- El AUC es consistentemente alto (>0.9) en todos los datasets excepto Sonar, indicando que las redes producen probabilidades bien calibradas independientemente de la compresión.

### Comparativa con baselines de pruning

Para validar que el método aporta valor frente a técnicas de poda establecidas, comparamos con dos baselines que usan la misma topología objetivo que encontró la nube:

- Magnitude Pruning: entrenar red completa → eliminar neuronas con menor norma L2 de pesos entrantes → fine-tune.
- Random Pruning: entrenar red completa → eliminar neuronas al azar → fine-tune (promedio de 5 runs).

Todos los métodos usan las mismas épocas totales de entrenamiento para comparación justa.

| Dataset | Método | Acc | F1 | AUC | Topología | Reducción |
|---|---|---|---|---|---|---|
| Breast Cancer | Clásico | 97.3% | 0.971 | 0.993 | [30,8,2] | — |
| | Magnitude | 97.3% | 0.971 | 0.993 | [30,2,2] | -74.4% |
| | Random | 97.3% | 0.971 | 0.991 | [30,2,2] | -74.4% |
| | Nube | 97.3% | 0.971 | 0.992 | [30,2,2] | -74.4% |
| Sonar | Clásico | 78.0% | 0.776 | 0.823 | [60,8,2] | — |
| | Magnitude | 78.0% | 0.776 | 0.809 | [60,1,2] | -87.2% |
| | Random | 69.8% | 0.685 | 0.774 | [60,1,2] | -87.2% |
| | Nube | 80.5% | 0.799 | 0.809 | [60,1,2] | -87.2% |
| Ionosphere | Clásico | 94.3% | 0.935 | 0.918 | [34,16,2] | — |
| | Magnitude | 87.1% | 0.853 | 0.904 | [34,3,2] | -81.0% |
| | Random | 88.0% | 0.861 | 0.906 | [34,3,2] | -81.0% |
| | Nube | 90.0% | 0.885 | 0.910 | [34,3,2] | -81.0% |
| Adult Income | Clásico | 84.2% | 0.758 | 0.901 | [104,16,2] | — |
| | Magnitude | 84.4% | 0.762 | 0.901 | [104,8,2] | -49.9% |
| | Random | 84.4% | 0.764 | 0.902 | [104,8,2] | -49.9% |
| | Nube | 85.0% | 0.782 | 0.904 | [104,8,2] | -49.9% |
| Iris | Clásico | 100.0% | 1.000 | 1.000 | [4,16,8,3] | — |
| | Magnitude | 100.0% | 1.000 | 1.000 | [4,16,3,3] | -41.2% |
| | Random | 100.0% | 1.000 | 1.000 | [4,16,3,3] | -41.2% |
| | Nube | 100.0% | 1.000 | 1.000 | [4,16,3,3] | -41.2% |
| Wine | Clásico | 94.4% | 0.944 | 0.996 | [13,16,3] | — |
| | Magnitude | 94.4% | 0.944 | 0.996 | [13,7,3] | -55.6% |
| | Random | 94.4% | 0.944 | 0.997 | [13,7,3] | -55.6% |
| | Nube | 94.4% | 0.944 | 0.995 | [13,7,3] | -55.6% |
| Opt. Digits | Clásico | 96.3% | 0.963 | 0.997 | [64,32,10] | — |
| | Magnitude | 95.0% | 0.950 | 0.996 | [64,12,10] | -62.2% |
| | Random | 95.4% | 0.954 | 0.996 | [64,12,10] | -62.2% |
| | Nube | 95.9% | 0.959 | 0.997 | [64,12,10] | -62.2% |

**Hallazgos:**
- La nube iguala o supera a ambos baselines de pruning en 6 de 7 datasets (en Breast Cancer empatan los tres).
- En Sonar, la nube supera a magnitude pruning por +2.5pp accuracy y +0.023 F1, y a random pruning por +10.7pp accuracy y +0.114 F1. Con compresión del 87%, la diferencia es dramática.
- En Ionosphere, la nube supera a magnitude pruning por +2.9pp y a random pruning por +2.0pp, ambos con 81% de compresión.
- En Adult Income, la nube supera a ambos baselines por +0.5-0.6pp accuracy y +0.017-0.020 F1. En un dataset de 30K muestras con desbalanceo, la mejora en F1 es significativa.
- En datasets fáciles (Iris, Wine, Breast Cancer), los tres métodos empatan — la topología objetivo es suficientemente expresiva para que cualquier método de poda funcione.
- La ventaja clave de la nube: no necesita entrenar la red completa antes de podar. Magnitude y random pruning requieren entrenar primero (coste O(épocas × muestras × parámetros_completos)) y luego fine-tune. La nube evalúa sin entrenar (coste O(nube × muestras × parámetros)) y solo entrena la red reducida.

### Escalabilidad: baselines en MNIST (784 dimensiones)

Para evaluar cómo se comporta el método con dimensionalidad alta, comparamos en MNIST (784 entradas, 10 clases). Topología [784, 32, 16, 10] (25,818 params). 30 épocas, LR=0.1. Nube: 100 redes, umbral=0.15, eliminar=2. Test fijo: 10,000 imágenes.

| N muestras | Método | Acc (test) | F1 | AUC | Topología | Reducción |
|---|---|---|---|---|---|---|
| 1,000 | Clásico | 84.6% | 0.843 | 0.974 | [784,32,16,10] | — |
| | Magnitude | 77.0% | 0.764 | 0.940 | [784,32,4,10] | -2.0% |
| | Random (×5) | 77.3% | 0.761 | 0.943 | [784,32,4,10] | -2.0% |
| | Nube | 59.8% | 0.504 | 0.879 | [784,32,4,10] | -2.0% |
| 5,000 | Clásico | 90.7% | 0.906 | 0.986 | [784,32,16,10] | — |
| | Magnitude | 87.5% | 0.874 | 0.971 | [784,32,4,10] | -2.0% |
| | Random (×5) | 86.9% | 0.867 | 0.969 | [784,32,4,10] | -2.0% |
| | Nube | 86.9% | 0.864 | 0.971 | [784,32,4,10] | -2.0% |

**Hallazgos de escalabilidad:**
- Con 1K muestras y 784 dimensiones, la nube pierde claramente: -17pp vs magnitude pruning. Con tan pocos datos en dimensión alta, la evaluación sin entrenamiento no tiene suficiente señal para seleccionar buenas redes.
- Con 5K muestras, la brecha se cierra drásticamente: la nube empata con random pruning (86.9%) y queda a solo 0.6pp de magnitude pruning. El AUC es idéntico (0.971).
- La tendencia es clara: a más datos, la nube converge hacia los baselines. Con 10K+ muestras (extrapolando de los resultados previos del README), la nube alcanza 90.2% test vs 92.5% del clásico.
- La reducción de parámetros en MNIST es modesta (-2.0%) porque la capa 784→32 domina con 25,088 parámetros. El método brilla más cuando las capas ocultas son proporcionalmente grandes respecto a la entrada.
- El coste computacional de la nube (84s vs 27s del clásico con 5K) refleja el overhead de exploración (100 redes × reducciones). Magnitude pruning necesita 54s (train + prune + fine-tune = 2× épocas).

**Conclusión sobre escalabilidad:** El método de la nube es competitivo con los baselines de pruning cuando hay suficientes datos (≥5K muestras para 784 dimensiones). Con datos escasos en dimensión alta, los métodos post-training (que aprovechan información aprendida) tienen ventaja. El sweet spot del método son datasets pequeños/medianos con dimensionalidad moderada, donde la exploración sin entrenamiento tiene señal suficiente y el ahorro de no entrenar la red completa es significativo.

### Significancia estadística (10 semillas)

Para validar que los resultados no son artefactos de una semilla particular, cada método se ejecutó con 10 semillas distintas (42, 123, 256, 314, 501, 667, 789, 888, 1024, 1337). Se reporta media ± desviación estándar y Wilcoxon signed-rank test (p<0.05 = significativo).

| Dataset | Método | Acc (mean±std) | F1 (mean±std) | p-value vs Nube (Acc) |
|---|---|---|---|---|
| Breast Cancer | Clásico | 97.3±0.0% | 0.971±0.000 | — |
| (-74.4%) | Magnitude | 97.3±0.0% | 0.971±0.000 | p=1.000 |
| | Random | 97.3±0.0% | 0.971±0.000 | p=1.000 |
| | Nube | 97.3±0.0% | 0.971±0.000 | — |
| Sonar | Clásico | 78.3±4.7% | 0.779±0.047 | — |
| (-87.2%) | Magnitude | 72.4±3.5% | 0.718±0.036 | p=0.017 ✓ |
| | Random | 69.5±2.9% | 0.684±0.033 | p=0.005 ✓ |
| | Nube | 77.3±2.6% | 0.769±0.026 | — |
| Ionosphere | Clásico | 90.3±2.9% | 0.887±0.035 | — |
| (-81.0%) | Magnitude | 89.9±3.2% | 0.883±0.039 | p=0.721 |
| | Random | 88.9±3.1% | 0.871±0.037 | p=0.721 |
| | Nube | 88.9±3.4% | 0.870±0.042 | — |
| Adult Income | Clásico | 84.5±0.1% | 0.786±0.003 | — |
| (-49.9%) | Magnitude | 84.7±0.1% | 0.787±0.003 | p=0.028 ✓ |
| | Random | 84.7±0.3% | 0.788±0.003 | p=0.139 |
| | Nube | 84.5±0.2% | 0.786±0.003 | — |
| Wine | Clásico | 94.4±0.0% | 0.944±0.000 | — |
| (-55.6%) | Magnitude | 94.4±0.0% | 0.944±0.000 | p=0.317 |
| | Random | 94.7±0.9% | 0.947±0.009 | p=0.180 |
| | Nube | 94.2±0.9% | 0.941±0.009 | — |
| Opt. Digits | Clásico | 98.1±0.2% | 0.981±0.002 | — |
| (-62.2%) | Magnitude | 97.5±0.2% | 0.975±0.002 | p=0.203 |
| | Random | 97.2±0.4% | 0.972±0.004 | p=0.959 |
| | Nube | 96.6±2.3% | 0.966±0.023 | — |

**Hallazgos:**
- En Sonar, la nube supera significativamente a ambos baselines (p=0.017 vs magnitude, p=0.005 vs random). Con 87% de compresión, la nube obtiene +4.9pp sobre magnitude y +7.8pp sobre random de media, con menor varianza (std=2.6% vs 3.5% y 2.9%).
- En Breast Cancer, los 4 métodos obtienen exactamente el mismo resultado en las 10 semillas (97.3%). El problema es tan "fácil" con esta topología que la semilla no importa.
- En Ionosphere, la nube empata estadísticamente con ambos baselines (p>0.7). Con 81% de compresión, los tres métodos podados rinden similar (~89%), todos cerca del clásico (90.3%).
- En Adult Income, magnitude pruning supera ligeramente a la nube en accuracy (p=0.028), pero en F1 no hay diferencia significativa (p=0.508). Con 30K+ muestras, los métodos post-training tienen una ligera ventaja porque aprovechan información aprendida, pero la diferencia es de solo 0.2pp.
- En Wine, los 4 métodos empatan estadísticamente. La nube tiene una ligera caída en una semilla (91.7% en seed 888), pero la media (94.2%) es indistinguible de los baselines.
- En Optical Digits, la nube empata estadísticamente con random pruning (p=0.959) y con magnitude pruning (p=0.203). La nube tiene mayor varianza (std=2.3%) por un outlier en seed 123 (90.5%), pero en 9 de 10 seeds rinde >96%.
- La nube nunca pierde significativamente contra random pruning en ningún dataset. Contra magnitude pruning, solo pierde en Adult Income (y solo en accuracy, no en F1).

### Sensibilidad a hiperparámetros

Se varió un hiperparámetro a la vez (los demás fijos) en 3 datasets representativos, con 3 seeds por configuración.

**Tamaño de nube** (10, 25, 50, 100, 200):

| Dataset | Nube=10 | Nube=25 | Nube=50 | Nube=100 | Nube=200 |
|---|---|---|---|---|---|
| Sonar | 76.4% / 0% | 77.2% / 0% | 77.2% / -87% | 78.0% / -87% | 76.4% / -75% |
| Ionosphere | 86.2% / -87% | 90.5% / 0% | 91.4% / -81% | 89.5% / -62% | 90.0% / -62% |
| Wine | 87.0% / -68% | 94.4% / -68% | 94.4% / -56% | 94.4% / -56% | 88.0% / -56% |

Formato: Acc test / reducción de parámetros. El método es robusto a partir de nube≥25. Con nube=10 la exploración es insuficiente (menos candidatos). Con nube=200 no mejora y puede empeorar ligeramente (más ruido en la selección). El sweet spot es 50-100.

**Umbral de acierto** (0.3, 0.4, 0.5, 0.6, 0.7):

| Dataset | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|
| Sonar | 77.2% ✓ | 77.2% ✓ | 77.2% ✓ | 77.2% ✓ | FAIL |
| Ionosphere | 91.4% ✓ | 91.4% ✓ | 91.4% ✓ | 91.4% ✓ | 91.4% ✓ |
| Wine | 94.4% ✓ | 94.4% ✓ | 94.4% ✓ | 94.4% ✓ | 94.4% (1/3) |

El método es muy robusto al umbral en el rango 0.3-0.6. Solo falla cuando el umbral es demasiado alto para que alguna red aleatoria lo supere (Sonar con 0.7). Recomendación: usar umbral entre 0.4 y 0.6.

**Neuronas a eliminar** (1, 2, 4):

| Dataset | Elim=1 | Elim=2 | Elim=4 |
|---|---|---|---|
| Sonar | 77.2% / -87% | 75.6% / -50% | 75.6% / -50% |
| Ionosphere | 91.4% / -81% | 89.5% / -87% | 90.0% / 0% |
| Wine | 94.4% / -56% | 94.4% / -37% | 94.4% / -50% |

Elim=1 produce la mayor compresión en Sonar e Ionosphere porque explora topologías más granulares. Elim=2 y 4 saltan topologías intermedias que podrían ser óptimas. En Wine la accuracy es idéntica pero la compresión varía. Recomendación: usar elim=1 para máxima compresión, elim=2 para velocidad.

### Coste computacional

Tiempo total y desglose por fase para cada método (mediana de 3 runs, 8 threads).

| Dataset | Clásico | Magnitude | Random | Nube | Ratio Nube/Clásico |
|---|---|---|---|---|---|
| Sonar (167 train) | 234ms | 351ms (1.5×) | 349ms (1.5×) | 158ms (0.67×) | 0.67× |
| Ionosphere (281 train) | 421ms | 676ms (1.6×) | 660ms (1.6×) | 341ms (0.81×) | 0.81× |
| Breast Cancer (456 train) | 241ms | 398ms (1.7×) | 425ms (1.8×) | 227ms (0.94×) | 0.94× |
| Opt. Digits (3823 train) | 7.9s | 11.5s (1.5×) | 11.5s (1.5×) | 6.3s (0.79×) | 0.79× |
| Adult Income (30162 train) | 13.1s | 21.7s (1.7×) | 21.7s (1.7×) | 15.4s (1.17×) | 1.17× |

Desglose de la nube (exploración + refinamiento):

| Dataset | Exploración | Refinamiento | % exploración |
|---|---|---|---|
| Sonar | 43ms | 115ms | 27% |
| Ionosphere | 86ms | 255ms | 25% |
| Breast Cancer | 57ms | 170ms | 25% |
| Opt. Digits | 2.4s | 3.8s | 39% |
| Adult Income | 6.7s | 8.7s | 44% |

**Hallazgos:**
- La nube es más rápida que el clásico en 4 de 5 datasets (0.67×-0.94×). Solo en Adult Income (30K muestras) es ligeramente más lenta (1.17×), porque la exploración con 50 redes × muchas reducciones sobre 30K muestras tiene overhead.
- Magnitude y random pruning son siempre 1.5-1.8× más lentos que el clásico porque necesitan entrenar la red completa Y luego fine-tune la red podada (doble entrenamiento).
- La nube es siempre más rápida que ambos baselines de pruning (0.67× vs 1.5× en Sonar, 0.79× vs 1.5× en Digits).
- La exploración (feedforward sin backprop) consume solo 25-44% del tiempo total. El refinamiento (backprop sobre la red reducida) domina. A medida que el dataset crece, la exploración pesa más (27% en Sonar → 44% en Adult).
- El coste de pruning en sí (seleccionar neuronas) es despreciable (<1ms) en todos los casos. El coste real está en el entrenamiento.

## Rendimiento

El motor paraleliza automáticamente la fase de exploración cuando Julia se lanza con múltiples threads:

```bash
# Lanzar con 8 threads (recomendado)
julia --project=. -t 8 examples/mnist.jl

# O usar auto-detección de cores
julia --project=. -t auto examples/mnist.jl
```

| Benchmark (MNIST 5K, nube=50) | 1 thread | 8 threads | Speedup |
|---|---|---|---|
| Exploración + refinamiento | 28.5s | 11.4s | 2.5x |

La fase de exploración (evaluar redes sin entrenar) se paraleliza con `Threads.@threads`. El refinamiento final con backpropagation es secuencial. BLAS (LinearAlgebra) puede usar threads adicionales internamente.

No se usa GPU — las matrices son demasiado pequeñas para que el overhead de transferencia CPU↔GPU compense. Para redes feedforward con estas topologías, multi-threading en CPU es más eficiente.

## Activaciones configurables y mini-batches

### Activaciones

La configuración soporta tres funciones de activación via el parámetro `activacion`:

- `:sigmoid` (por defecto) — Capas ocultas y salida usan sigmoid. Comportamiento original.
- `:relu` — Capas ocultas usan ReLU, capa de salida usa sigmoid. Requiere LR más bajo (~10x).
- `:identidad` — Capas ocultas usan identidad, capa de salida usa identidad. Para regresión lineal.

```julia
config = ConfiguracionNube(
    topologia_inicial = [13, 16, 3],
    activacion = :relu,          # ReLU en capas ocultas
    tasa_aprendizaje = 0.01,     # LR reducido para ReLU
    # ... otros parámetros
)
```

### Mini-batches

El parámetro `batch_size` controla el tamaño de mini-batch en el refinamiento:

- `0` (por defecto) — SGD sample-by-sample (comportamiento original).
- `> 0` — Acumula gradientes sobre el batch y actualiza una vez. Divide el LR por el tamaño del batch automáticamente.

```julia
config = ConfiguracionNube(
    batch_size = 32,             # Mini-batches de 32 muestras
    # ... otros parámetros
)
```

### Resultados experimentales: Sigmoid vs ReLU × SGD vs Mini-batch

Comparativa de las 4 combinaciones en 6 datasets representativos. ReLU usa LR=0.01 (clasificación) o LR=0.001 (regresión); sigmoid usa LR=0.1 (clasificación) o LR=0.01 (regresión).

| Dataset | Sigmoid+SGD | Sigmoid+MB(32) | ReLU+SGD | ReLU+MB(32) | Observación |
|---|---|---|---|---|---|
| XOR | 75.0% / -15.7% | 75.0% / -15.7% | 100.0% / 0% | 100.0% / 0% | ReLU resuelve XOR al 100% con LR bajo |
| Iris | 100.0% / -41.2% | 100.0% / -41.2% | 100.0% / -8.2% | 73.3% / -8.2% | Sigmoid comprime más; ReLU+MB subconverge |
| Wine | 94.4% / -55.6% | 94.4% / -55.6% | 94.4% / -68.0% | 66.7% / -68.0% | ReLU+SGD comprime más; MB degrada |
| Breast Cancer | 97.3% / -74.4% | 97.3% / -74.4% | 97.3% / 0% | 93.8% / 0% | Sigmoid comprime mejor |
| Boston (R²) | 0.725 / -6.6% | 0.725 / -6.6% | 0.590 / -50.4% | 0.316 / -50.4% | Sigmoid gana en regresión |
| Adult Income | 85.0% / -49.9% | 84.5% / -49.9% | 83.4% / 0% | 82.7% / 0% | Sigmoid gana en datos tabulares grandes |
| Opt. Digits | 95.9% / -62.2% | 95.6% / -62.2% | 60.9% / -93.4% | 21.9% / -93.4% | Sigmoid domina en visión |

Formato: precisión test / reducción de parámetros.

**Hallazgos:**

- **Sigmoid + SGD sigue siendo la mejor configuración general** para el método de la nube. La evaluación sin entrenamiento (fase de exploración) funciona mejor con sigmoid porque sus salidas están acotadas en (0,1), lo que produce clasificaciones más estables con pesos aleatorios.
- **ReLU mejora en XOR** (100% vs 75%) y **comprime más en Wine** (-68% vs -55.6%), pero degrada en la mayoría de datasets porque la fase de exploración (sin entrenamiento) es menos efectiva con activaciones no acotadas.
- **Mini-batches no aportan ventaja** en datasets pequeños/medianos (<1000 muestras) donde el batch es casi todo el dataset. En datasets grandes (Adult Income, 30K muestras), la diferencia es marginal (-0.5pp) porque el refinamiento ya usa pocas épocas (30).
- **ReLU + Mini-batch es la peor combinación**: el LR reducido necesario para ReLU, dividido además por el batch_size, produce convergencia insuficiente en las pocas épocas de refinamiento.

**Recomendación:** Usar `activacion=:sigmoid, batch_size=0` (valores por defecto) para la mayoría de problemas. ReLU puede ser útil para problemas específicos donde se necesita mayor compresión y se puede compensar con más épocas de refinamiento.

## Estructura del proyecto

```
RandomCloud.jl/
├── Project.toml
├── src/
│   ├── RandomCloud.jl          # Módulo principal
│   ├── configuracion.jl        # ConfiguracionNube (activacion, batch_size)
│   ├── activaciones.jl         # sigmoid, relu, identidad + despacho por símbolo
│   ├── red_neuronal.jl         # RedNeuronal, feedforward, entrenar!, entrenar_batch!, reconstruir
│   ├── politica.jl             # PoliticaEliminacion, PoliticaSecuencial
│   ├── evaluacion.jl           # evaluar, evaluar_regresion
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
│   ├── comparativa_*.jl        # Comparativas Nube vs Clásico
│   └── comparativa_relu_*.jl   # Comparativas activaciones y mini-batches
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
