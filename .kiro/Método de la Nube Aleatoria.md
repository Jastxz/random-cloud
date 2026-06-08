# El Método de la Nube Aleatoria

## Un nuevo método de búsqueda de arquitectura de redes neuronales mediante evaluación sin entrenamiento y reducción estructural progresiva

**Autor:** Javier Gil Blázquez

**Fecha:** Marzo de 2026

**Versión:** 1.0

---

## Resumen

Este documento describe el Método de la Nube Aleatoria, un procedimiento original para la búsqueda automática de arquitecturas de redes neuronales artificiales. El método se basa en tres principios: (1) la generación de un conjunto diverso de redes neuronales con pesos inicializados aleatoriamente, denominado Nube Aleatoria; (2) la evaluación de cada red sin entrenamiento previo, utilizando únicamente propagación hacia adelante contra un umbral de acierto mínimo; y (3) la reducción estructural progresiva de neuronas en las capas ocultas para encontrar la arquitectura mínima viable. La red resultante se refina mediante entrenamiento clásico por retropropagación. El método permite descubrir arquitecturas más compactas que las diseñadas manualmente, manteniendo un rendimiento equivalente, y se diferencia de los métodos existentes de poda y búsqueda de arquitectura en que opera sobre múltiples redes simultáneamente, elimina neuronas completas (no conexiones individuales), y no requiere entrenamiento previo para la fase de búsqueda.

---

## 1. Introducción

El diseño de la arquitectura de una red neuronal artificial es un problema abierto en el campo del aprendizaje automático. Tradicionalmente, el número de capas y neuronas se elige de forma heurística o mediante búsqueda exhaustiva, lo que resulta en redes sobredimensionadas que consumen más recursos de los necesarios.

Este documento presenta un método alternativo que invierte el enfoque habitual: en lugar de partir de una arquitectura pequeña y crecer, se parte de una arquitectura grande y se reduce sistemáticamente hasta encontrar la estructura mínima que satisface un criterio de rendimiento. La originalidad del método reside en que esta búsqueda se realiza sobre un conjunto de redes con pesos aleatorios, sin necesidad de entrenar ninguna de ellas durante la fase de búsqueda.

---

## 2. Definiciones formales

### 2.1. Red neuronal feedforward

Sea una red neuronal feedforward definida por su topología **T** = (t₀, t₁, ..., tₗ), donde:
- t₀ es el número de neuronas de la capa de entrada
- tₗ es el número de neuronas de la capa de salida
- t₁, ..., tₗ₋₁ son los números de neuronas de las capas ocultas
- L = l + 1 es el número total de capas

La red tiene asociadas matrices de pesos **W** = {W₁, W₂, ..., Wₗ} donde Wᵢ ∈ ℝ^(tᵢ × tᵢ₋₁), y vectores de sesgo **b** = {b₁, b₂, ..., bₗ} donde bᵢ ∈ ℝ^(tᵢ).

El número total de parámetros entrenables de la red es:

P(T) = Σᵢ₌₁ˡ (tᵢ₋₁ · tᵢ + tᵢ)

### 2.2. Nube Aleatoria

Sea n ∈ ℕ con n ≥ 1. Una Nube Aleatoria **C** de tamaño n y topología **T** es un conjunto:

**C** = {R₁, R₂, ..., Rₙ}

donde cada Rⱼ es una red neuronal feedforward con topología **T** cuyos pesos y sesgos han sido inicializados de forma aleatoria e independiente. Formalmente, para cada red Rⱼ y cada capa i:

Wᵢ⁽ʲ⁾ ~ U(-1, 1)^(tᵢ × tᵢ₋₁)

bᵢ⁽ʲ⁾ ~ U(-1, 1)^(tᵢ)

donde U(-1, 1) denota la distribución uniforme en el intervalo [-1, 1].

### 2.3. Función de evaluación

Sea D = {(x₁, y₁), (x₂, y₂), ..., (xₘ, yₘ)} un conjunto de datos de evaluación con m muestras, donde xₖ ∈ ℝ^(t₀) e yₖ ∈ ℝ^(tₗ).

La función de evaluación mide qué proporción de las muestras clasifica correctamente una red R sobre el conjunto D. Se calcula así:

1. Para cada muestra k, se propaga la entrada xₖ a través de la red y se obtiene un vector de salida R(xₖ).
2. Se compara la posición del valor más alto del vector de salida con la posición del valor más alto del vector objetivo yₖ. Si coinciden, la predicción es correcta.
3. La precisión es el número de predicciones correctas dividido entre el total de muestras.

Formalmente:

eval(R, D) = aciertos / m

donde:
- aciertos = número de muestras k en las que posición_del_máximo(R(xₖ)) = posición_del_máximo(yₖ)
- m = número total de muestras

Por ejemplo, si la red produce la salida [0.1, 0.8, 0.3] y el objetivo es [0, 1, 0], ambos tienen su valor máximo en la segunda posición, por lo que la predicción se considera correcta.

En notación matemática formal, esto se expresa como:

eval(R, D) = (1/m) · Σₖ₌₁ᵐ 𝟙[argmax(R(xₖ)) = argmax(yₖ)]

donde argmax devuelve el índice del valor máximo de un vector, y 𝟙[·] es la función indicadora que vale 1 si la condición entre corchetes es verdadera y 0 en caso contrario.

### 2.4. Umbral de acierto

Sea θ ∈ [0, 1] un valor real denominado umbral de acierto. Una red R se considera viable si:

eval(R, D) > θ

### 2.5. Política de eliminación

Una política de eliminación es una función:

π: ℕᴸ × ℕ → ℕᴸ ∪ {∅}

que recibe la topología actual de la red y un número de neuronas a eliminar, y devuelve una nueva topología con neuronas eliminadas de alguna capa oculta, o el conjunto vacío ∅ si no es posible realizar más reducciones (todas las capas ocultas tienen 0 neuronas).

La política debe satisfacer las siguientes restricciones:
- La capa de entrada y la capa de salida no se modifican: π(T, x)₀ = t₀ y π(T, x)ₗ = tₗ
- El número total de neuronas ocultas se reduce: Σᵢ₌₁ˡ⁻¹ π(T, x)ᵢ < Σᵢ₌₁ˡ⁻¹ tᵢ

Se define la política de eliminación secuencial πₛ como aquella que elimina x neuronas de la última capa oculta con neuronas disponibles, avanzando hacia capas anteriores cuando una capa se agota.

---

## 3. Descripción del método

El Método de la Nube Aleatoria consta de cuatro fases secuenciales.

### 3.1. Fase 1: Generación de la Nube Aleatoria

**Entrada:** Topología T, tamaño de nube n, semilla de reproducibilidad s.

**Proceso:** Se generan n redes neuronales feedforward con topología T y pesos aleatorios, utilizando la semilla s para garantizar reproducibilidad. Cada red se inicializa de forma independiente mediante sub-semillas derivadas de s.

**Salida:** Nube Aleatoria C = {R₁, R₂, ..., Rₙ}.

### 3.2. Fase 2: Selección del umbral de acierto

**Entrada:** Valor θ ∈ [0, 1] elegido por el usuario.

El umbral puede seleccionarse de forma arbitraria o estocástica. Un umbral bajo (ej: 0.15-0.25) aumenta la probabilidad de encontrar redes viables pero puede resultar en arquitecturas menos optimizadas. Un umbral alto exige mayor rendimiento de las redes aleatorias, lo que puede requerir nubes más grandes.

### 3.3. Fase 3: Proceso de reducción

Esta es la fase central del método. Para cada red Rⱼ de la nube, se aplica iterativamente el siguiente ciclo. La clave es que la política de eliminación genera una secuencia de topologías decrecientes, y el método evalúa la red en **cada uno de esos estados intermedios**, guardando siempre la mejor configuración encontrada. Así, para cada red de la nube se exploran todas las subredes posibles que la política puede generar a partir de ella.

```
ALGORITMO: Proceso de Reducción
─────────────────────────────────────────────────
Entrada: Nube C, datos D, umbral θ, política π, neuronas a eliminar x
Salida: Mejor red R* y su precisión p*

1.  R* ← ∅
2.  p* ← 0
3.  Para cada red Rⱼ ∈ C:
4.      R_actual ← Rⱼ
5.      T_actual ← topología(Rⱼ)
6.      // Explorar todas las subredes generadas por la política
7.      // partiendo de Rⱼ con su topología completa hasta agotar
8.      // todas las posibilidades de reducción
9.      Repetir:
10.         p ← eval(R_actual, D)
11.         Si p > θ  Y  p > p*:
12.             R* ← R_actual
13.             p* ← p
14.         T_nueva ← π(T_actual, x)
15.         Si T_nueva = ∅:
16.             // No quedan más subredes que explorar para esta red
17.             Salir del bucle
18.         R_actual ← reconstruir(R_actual, T_nueva)
19.         T_actual ← T_nueva
20. Retornar (R*, p*)
```

El bucle de las líneas 9-19 recorre la secuencia completa de subredes que la política puede generar a partir de Rⱼ: desde la red con su topología original hasta la red mínima posible. En cada paso se evalúa la red actual y se actualiza la mejor configuración global si procede. Cuando la política devuelve ∅ (no hay más reducciones posibles), se pasa a la siguiente red de la nube. Al finalizar el bucle externo (línea 3), se habrán explorado todas las subredes de todas las redes de la nube.

La operación **reconstruir(R, T')** crea una nueva red con topología T' preservando los pesos de las neuronas no eliminadas. Específicamente, para cada transición de capa i:
- La matriz de pesos se recorta a la submatriz superior-izquierda de dimensiones t'ᵢ × t'ᵢ₋₁
- El vector de sesgos se recorta a las primeras t'ᵢ componentes
- Si una capa oculta queda con 0 neuronas, se elimina de la topología y se colapsan las conexiones de las capas adyacentes

### 3.4. Fase 4: Refinamiento

Si el proceso de reducción ha encontrado una red viable R* (es decir, R* ≠ ∅), se procede a entrenarla mediante retropropagación clásica (backpropagation) con los datos de entrenamiento D, durante un número de épocas E y con una tasa de aprendizaje α configurables.

Si ninguna red de la nube ha superado el umbral (R* = ∅), el método indica que no se encontró una arquitectura viable con los parámetros dados, y se recomienda reiniciar el proceso con una nube más grande o un umbral más bajo.

---

## 4. Análisis de complejidad

### 4.1. Complejidad temporal

Sea:
- n = tamaño de la nube
- H = número total de neuronas ocultas en la topología inicial (H = Σᵢ₌₁ˡ⁻¹ tᵢ)
- x = neuronas eliminadas por iteración
- m = número de muestras de evaluación
- P = número de parámetros de la red

El número máximo de iteraciones de reducción por red es ⌈H/x⌉.

La complejidad de una evaluación (feedforward sobre m muestras) es O(m · P).

Por tanto, la complejidad total del proceso de reducción es:

O(n · ⌈H/x⌉ · m · P)

La complejidad del refinamiento final es O(E · m · P'), donde E es el número de épocas de entrenamiento (configurado por el usuario) y P' ≤ P es el número de parámetros de la red reducida.

### 4.2. Complejidad espacial

El método requiere almacenar n redes simultáneamente durante la generación, pero solo una red activa durante el proceso de reducción. La complejidad espacial es O(n · P) durante la generación y O(P) durante la reducción.

---

## 5. Propiedades del método

### 5.1. Reproducibilidad

Dado que todos los generadores de números aleatorios se inicializan a partir de una semilla determinista, el método produce resultados idénticos para la misma configuración y los mismos datos de entrada.

### 5.2. Independencia de las redes

Cada red de la nube se genera de forma independiente, sin estado compartido. Esto permite la paralelización trivial del proceso de reducción.

### 5.3. Preservación de pesos

Durante la reducción, los pesos de las neuronas no eliminadas se preservan exactamente. Esto garantiza que la evaluación de la red reducida refleja fielmente el comportamiento de la subred correspondiente dentro de la red original.

### 5.4. Extensibilidad de la política de eliminación

La política de eliminación se define como una interfaz abstracta, lo que permite implementar estrategias alternativas (eliminación por magnitud de pesos, eliminación aleatoria, eliminación basada en sensibilidad, etc.) sin modificar el resto del método.

---

## 6. Resultados experimentales

Se presentan los resultados obtenidos con la implementación de referencia del método, comparando con el entrenamiento clásico por retropropagación.

### 6.1. Problema XOR

| Métrica | Clásico [2,8,4,2] | Nube Aleatoria |
|---|---|---|
| Topología final | [2,8,4,2] | [2,3,2] |
| Parámetros | 70 | 17 |
| Precisión | 100% | 100% |
| Reducción de parámetros | — | 75.7% |

### 6.2. Juego de 3 en Raya (4520 estados)

| Métrica | Clásico [10,30,9] | Nube Aleatoria |
|---|---|---|
| Topología final | [10,30,9] | [10,20,9] |
| Parámetros | 609 | 409 |
| Precisión sobre datos | 62.43% | 53.30% |
| Victorias vs aleatorio | 97.0% | 96.0% |
| Enfrentamiento directo | 100% empates (400 partidas) |
| Reducción de parámetros | — | 32.8% |

### 6.3. Escalabilidad

Resultados de escalado combinado (ancho + profundidad crecientes):

| Topología inicial | Params. inicial | Topología Nube | Params. Nube | Reducción |
|---|---|---|---|---|
| [2,4,2] | 22 | [2,2,2] | 12 | 45.5% |
| [2,8,4,2] | 70 | [2,3,2] | 17 | 75.7% |
| [2,16,8,2] | 202 | [2,16,1,2] | 69 | 65.8% |
| [2,32,16,8,2] | 778 | [2,32,10,2] | 448 | 42.4% |
| [2,64,32,16,2] | 2834 | [2,64,2,2] | 328 | 88.4% |
| [2,64,32,16,8,2] | 2954 | [2,56,2] | 282 | 90.5% |

En todos los casos, la red reducida por el Método de la Nube Aleatoria mantuvo una precisión del 100% en el problema XOR, con reducciones de parámetros de entre el 42% y el 90%.

---

## 7. Relación con trabajos existentes

El Método de la Nube Aleatoria se sitúa en la intersección de tres áreas de investigación:

### 7.1. Búsqueda de Arquitectura Neural (NAS)

Los métodos de NAS buscan automáticamente la arquitectura óptima de una red neuronal. A diferencia de los enfoques basados en aprendizaje por refuerzo o algoritmos evolutivos, el Método de la Nube Aleatoria no requiere entrenamiento durante la fase de búsqueda, lo que reduce significativamente el coste computacional.

### 7.2. Hipótesis del Billete de Lotería (Lottery Ticket Hypothesis)

La Lottery Ticket Hypothesis (Frankle y Carlin, 2019) postula que dentro de una red aleatoria existe una subred que, entrenada aisladamente, alcanza el rendimiento de la red completa. El Método de la Nube Aleatoria comparte la intuición de que las redes aleatorias contienen estructura útil, pero se diferencia en que: (a) opera sobre múltiples redes simultáneamente; (b) elimina neuronas completas, no conexiones individuales; (c) no requiere un ciclo de entrenamiento-poda-reinicio.

### 7.3. Poda en la inicialización (Pruning at Initialization)

Los métodos de PaI podan redes antes de entrenarlas. El Método de la Nube Aleatoria se diferencia en que realiza poda estructural (eliminación de neuronas enteras que modifica la topología) en lugar de poda no estructural (eliminación de conexiones individuales que mantiene la topología pero introduce dispersión).

---

## 8. Limitaciones conocidas

1. El método evalúa redes sin entrenar, por lo que el umbral de acierto debe ser bajo (las redes aleatorias tienen rendimiento cercano al azar). Esto limita la capacidad de discriminación en la fase de búsqueda.
2. La complejidad del proceso de reducción crece linealmente con el tamaño de la nube y el número de neuronas ocultas.
3. El método ha sido validado en problemas de clasificación de complejidad baja a media. Su comportamiento en problemas de alta dimensionalidad (imágenes, texto) requiere investigación adicional.
4. La política de eliminación secuencial es la más simple posible. Políticas más sofisticadas podrían mejorar significativamente los resultados.

---

## 9. Conclusiones

El Método de la Nube Aleatoria es un procedimiento original para la búsqueda automática de arquitecturas de redes neuronales que combina la generación de múltiples redes aleatorias, la evaluación sin entrenamiento y la reducción estructural progresiva. Los resultados experimentales demuestran que el método es capaz de encontrar arquitecturas significativamente más compactas (hasta un 90% menos de parámetros) que las diseñadas manualmente, manteniendo un rendimiento equivalente tras el refinamiento por retropropagación.

---

*Documento elaborado como descripción formal del Método de la Nube Aleatoria para su registro como obra científica original.*
