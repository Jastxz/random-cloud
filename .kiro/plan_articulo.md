# Plan del Artículo: Método de la Nube Aleatoria

## Estrategia de publicación

**Fase 1:** Workshop paper (4-6 páginas) → NeurIPS/ICML Workshop on Efficient Neural Network Design
**Fase 2:** Paper completo (8-10 páginas) → AutoML Conference o ECML-PKDD

---

## FASE 1: Workshop Paper (4-6 páginas)

### Título propuesto

"Random Cloud: Training-Free Neural Architecture Search via Stochastic Topology Exploration and Progressive Structural Reduction"

Alternativa más corta: "Finding Minimal Neural Architectures Without Training"

### Estructura

#### 1. Introduction (0.75 páginas)

Problema: encontrar la topología mínima de una red neuronal es costoso porque los métodos existentes (NAS, pruning) requieren entrenar la red completa antes de podarla o evaluarla.

Contribución: un método que encuentra topologías mínimas evaluando redes con pesos aleatorios (sin backpropagation), reduciendo progresivamente la arquitectura, y refinando solo la mejor candidata al final.

Claim principal: "El método iguala o supera a magnitude pruning y random pruning en 6 de 7 datasets evaluados, con menor coste computacional, sin requerir entrenamiento previo de la red completa."

Key results en un mini-table: Sonar (+4.9pp vs magnitude, p=0.017), Adult Income (+0.8pp accuracy, +0.024 F1), Breast Cancer (mismo resultado con 74% menos parámetros).

#### 2. Method (1 página)

Algoritmo en 4 pasos:
1. Generar N redes con pesos aleatorios y topología T₀
2. Para cada red: evaluar accuracy sin entrenar → reducir topología → repetir hasta no poder reducir
3. Seleccionar la red con mejor accuracy sin entrenamiento que supere el umbral θ
4. Refinar la red seleccionada con backpropagation (épocas fijas)

Pseudocódigo formal (Algorithm 1).

Diferencia clave vs pruning: la nube opera pre-training (evalúa sin entrenar), los baselines operan post-training (entrenan → podan → re-entrenan).

Política de reducción: secuencial (última capa oculta primero, luego anteriores). Eliminar n neuronas por paso.

#### 3. Experiments (1.5-2 páginas)

Setup experimental:
- 7 datasets: Breast Cancer, Sonar, Ionosphere, Adult Income, Iris, Wine, Optical Digits
- 3 métricas: Accuracy, F1-score (macro), AUC-ROC
- 4 métodos: Clásico (full training), Magnitude Pruning, Random Pruning, Nube Aleatoria
- 10 semillas, Wilcoxon signed-rank test

Tabla principal (Table 1): Accuracy/F1/Reducción por dataset × método (la tabla de baselines del README).

Tabla de significancia (Table 2): mean±std y p-values (datos de significancia_estadistica.jl).

Tabla de coste (Table 3): Tiempo total y ratio vs clásico (datos de coste_computacional.jl).

#### 4. Analysis (0.5 páginas)

- Sweet spot: datasets tabulares con dimensionalidad moderada (30-104 features)
- Limitación: dimensionalidad alta (784 en MNIST) con pocos datos degrada la señal de evaluación sin entrenamiento
- Robustez: insensible al umbral (0.3-0.6), robusto al tamaño de nube (≥25)
- Coste: 0.67-0.94× del clásico en 4/5 datasets, siempre más rápido que los baselines de pruning

#### 5. Related Work (0.5 páginas)

Posicionar vs:
- Pruning post-training: Han et al. 2015 (magnitude), Li et al. 2017 (filter pruning)
- Lottery Ticket: Frankle & Carlin 2019 (train-prune-reset cycle)
- NAS: Zoph & Le 2017 (RL-based), Liu et al. 2019 (DARTS, gradient-based)
- Training-free NAS: Mellor et al. 2021 (score-based), Abdelfattah et al. 2021 (zero-cost proxies)

Diferenciación: la nube es el único método que combina evaluación sin entrenamiento con reducción topológica progresiva en un solo paso.

#### 6. Conclusion (0.25 páginas)

Resumen de contribución + limitaciones honestas + trabajo futuro (más políticas de eliminación, escalabilidad a CNNs).

### Figuras para el workshop paper (3 figuras)

**Fig 1:** Diagrama del método (flowchart: generar nube → evaluar → reducir → seleccionar → refinar)

**Fig 2:** Barras agrupadas: Accuracy por método × dataset (7 datasets, 4 barras cada uno)

**Fig 3:** Scatter: Accuracy vs Reducción de parámetros (cada punto = un dataset, color = método)

---

## FASE 2: Paper Completo (8-10 páginas, AutoML/ECML)

### Secciones adicionales respecto al workshop

#### Extended Experiments
- MNIST y Fashion-MNIST (escalabilidad a 784 dimensiones)
- CIFAR-10 (límites del método en visión)
- Boston Housing (regresión con R²)
- Two Moons y XOR (toy problems para intuición)

#### Ablation Study
- Sensibilidad al tamaño de nube (10-200)
- Sensibilidad al umbral (0.3-0.7)
- Sensibilidad a neuronas_eliminar (1, 2, 4)
- ReLU vs Sigmoid (resultados negativos honestos)
- Mini-batches vs SGD (resultados negativos honestos)

#### Computational Cost Analysis
- Desglose exploración vs refinamiento
- FLOPs teóricos
- Escalabilidad con número de muestras (MNIST 1K-60K)

#### Theoretical Analysis
- Por qué funciona: la evaluación sin entrenamiento con sigmoid produce señal suficiente porque las salidas están acotadas en (0,1)
- Por qué falla en dimensión alta: con 784 entradas y pesos aleatorios, las pre-activaciones tienden a valores extremos → sigmoid satura → todas las redes producen salidas similares → la señal se pierde
- Conexión con lottery ticket hypothesis: la nube busca "winning topologies" en lugar de "winning tickets"

### Figuras adicionales para el paper completo (3 más)

**Fig 4:** Líneas: Accuracy vs N muestras en MNIST (4 métodos, mostrando convergencia)

**Fig 5:** Heatmap: Sensibilidad (tamaño_nube × umbral → accuracy) para Sonar

**Fig 6:** Barras apiladas: Desglose de tiempo (exploración + refinamiento) por dataset

---

## Datos disponibles para cada figura/tabla

| Figura/Tabla | Script fuente | Datos listos |
|---|---|---|
| Table 1: Baselines | comparativa_baselines.jl | ✓ |
| Table 2: Significancia | significancia_estadistica.jl | ✓ |
| Table 3: Coste | coste_computacional.jl | ✓ |
| Table 4: Sensibilidad | sensibilidad_hiperparametros.jl | ✓ |
| Table 5: F1/AUC | metricas_f1_auc.jl | ✓ |
| Table 6: MNIST baselines | baselines_mnist.jl | ✓ (1K, 5K) |
| Fig 1: Diagrama método | — | Diseñar |
| Fig 2: Barras accuracy | Datos de Table 1 | Generar con CairoMakie |
| Fig 3: Scatter acc vs red | Datos de Table 1 | Generar con CairoMakie |
| Fig 4: MNIST escalabilidad | comparativa_mnist.jl | ✓ |
| Fig 5: Heatmap sensibilidad | sensibilidad_hiperparametros.jl | Parcial (falta grid completo) |
| Fig 6: Desglose tiempo | coste_computacional.jl | ✓ |

## Referencias clave a citar

1. Han, S., Pool, J., Tung, J., & Dally, W. J. (2015). Learning both weights and connections for efficient neural networks. NeurIPS.
2. Frankle, J., & Carlin, M. (2019). The lottery ticket hypothesis. ICLR.
3. Zoph, B., & Le, Q. V. (2017). Neural architecture search with reinforcement learning. ICLR.
4. Liu, H., Simonyan, K., & Yang, Y. (2019). DARTS: Differentiable architecture search. ICLR.
5. Mellor, J., Turner, J., Sherwood, A., & Sherwood, P. (2021). Neural architecture search without training. ICML.
6. Abdelfattah, M. S., et al. (2021). Zero-cost proxies for lightweight NAS. ICLR.
7. Li, H., Kadav, A., Durdanovic, I., Samet, H., & Graf, H. P. (2017). Pruning filters for efficient convnets. ICLR.

## Próximos pasos

1. [ ] Generar las 3 figuras del workshop paper con CairoMakie
2. [ ] Escribir el borrador del workshop paper (4-6 páginas)
3. [ ] Revisar y pulir
4. [ ] Identificar deadline del workshop target
5. [ ] Expandir a paper completo si el workshop es aceptado
