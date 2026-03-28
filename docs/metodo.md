# Método de la Nube Aleatoria

## 1. Introducción

El Método de la Nube Aleatoria es un enfoque de búsqueda de arquitecturas de redes neuronales
(*Neural Architecture Search*, NAS) que encuentra topologías mínimas de redes neuronales
feedforward capaces de resolver un problema de clasificación dado. A diferencia de los métodos
NAS convencionales que entrenan cada arquitectura candidata, este método evalúa las redes
**sin entrenamiento previo** — utilizando únicamente propagación hacia adelante (*feedforward*)
con pesos aleatorios — y aplica **reducción estructural progresiva** para descubrir la
arquitectura más compacta que supere un umbral de precisión definido por el usuario.

La hipótesis central es que, dada una nube suficientemente grande de redes con pesos aleatorios,
al menos una de ellas (o una de sus sub-topologías) exhibirá una precisión aceptable sin
necesidad de entrenamiento. La red seleccionada se refina posteriormente con retropropagación
para alcanzar su rendimiento final.

## 2. Definición formal del algoritmo

### 2.1 Notación

| Símbolo | Significado |
|---------|-------------|
| $C$ | Configuración de hiperparámetros (`ConfiguracionNube`) |
| $D = (X, Y)$ | Dataset con entradas $X$ y objetivos $Y$ en formato column-major |
| $N$ | Tamaño de la nube (`tamano_nube`) |
| $T_0$ | Topología inicial (`topologia_inicial`), vector $[n_1, n_2, \ldots, n_L]$ |
| $\theta$ | Umbral de acierto mínimo (`umbral_acierto`) |
| $x$ | Neuronas a eliminar por paso de reducción (`neuronas_eliminar`) |
| $E$ | Épocas de refinamiento (`epocas_refinamiento`) |
| $\eta$ | Tasa de aprendizaje (`tasa_aprendizaje`) |
| $\pi$ | Política de eliminación (`PoliticaSecuencial`) |
| $R^*$ | Mejor red encontrada globalmente |
| $p^*$ | Mejor precisión encontrada globalmente |
| $\sigma(x)$ | Función sigmoide: $\sigma(x) = \frac{1}{1 + e^{-x}}$ |

### 2.2 Algoritmo principal

**Entrada:** Configuración $C$, dataset $D = (X, Y)$

**Salida:** `InformeNube` con la mejor red encontrada, precisión, topología final y metadatos


**Procedimiento:**

1. Inicializar generador de números aleatorios local: $\text{rng} \leftarrow \text{MersenneTwister}(C.\text{semilla})$
2. Inicializar $R^* \leftarrow \emptyset$, $p^* \leftarrow 0$
3. Inicializar contadores: $\text{eval} \leftarrow 0$, $\text{reduc} \leftarrow 0$
4. Registrar tiempo de inicio $t_0$
5. **Generar nube:** Crear $N$ redes neuronales $\{R_1, R_2, \ldots, R_N\}$ con topología $T_0$ y pesos aleatorios en $[-1, 1]$ usando $\text{rng}$
6. **Para cada** red $R_j$ en la nube ($j = 1, \ldots, N$):
   - $R_{\text{actual}} \leftarrow R_j$
   - $T_{\text{actual}} \leftarrow T_0$
   - **Repetir** (exploración de sub-topologías):
     1. $p \leftarrow \text{evaluar}(R_{\text{actual}}, X, Y)$; $\text{eval} \leftarrow \text{eval} + 1$
     2. **Si** $p > \theta$ **y** $p > p^*$ **entonces:** $R^* \leftarrow R_{\text{actual}}$, $p^* \leftarrow p$
     3. $T_{\text{nueva}} \leftarrow \pi(T_{\text{actual}}, x)$
     4. **Si** $T_{\text{nueva}} = \emptyset$ **entonces:** salir del bucle (pasar a $R_{j+1}$)
     5. $R_{\text{actual}} \leftarrow \text{reconstruir}(R_{\text{actual}}, T_{\text{nueva}})$; $\text{reduc} \leftarrow \text{reduc} + 1$
     6. $T_{\text{actual}} \leftarrow T_{\text{nueva}}$
7. **Si** $R^* \neq \emptyset$ (se encontró red viable):
   - **Refinar** $R^*$ con retropropagación durante $E$ épocas con tasa $\eta$
   - $p^* \leftarrow \text{evaluar}(R^*, X, Y)$
   - **Retornar** `InformeNube(exitoso=true, mejor_red=R*, precision=p*, ...)`
8. **Si** $R^* = \emptyset$ (ninguna red superó $\theta$):
   - **Retornar** `InformeNube(exitoso=false, mejor_red=nothing, ...)`

### 2.3 Subprocedimientos

#### Evaluación sin entrenamiento

La función `evaluar` calcula la proporción de muestras correctamente clasificadas:

$$p = \frac{|\{k : \arg\max(\text{feedforward}(R, X_k)) = \arg\max(Y_k)\}|}{m}$$

donde $m$ es el número total de muestras y $X_k$, $Y_k$ son la $k$-ésima columna de $X$ e $Y$ respectivamente.

#### Propagación hacia adelante (feedforward)

Para una red con $L$ capas y matrices de pesos $W_i$, vectores de biases $b_i$:

$$a^{(0)} = \text{entrada}$$
$$a^{(i)} = \sigma(W_i \cdot a^{(i-1)} + b_i), \quad i = 1, \ldots, L-1$$

La salida es $a^{(L-1)}$, con cada componente en el intervalo abierto $(0, 1)$.

#### Política de eliminación secuencial

La función $\pi(T, x)$ opera sobre la topología $T = [n_1, \ldots, n_L]$:

1. Si todas las capas ocultas $n_2, \ldots, n_{L-1}$ tienen 0 neuronas → retornar $\emptyset$
2. Encontrar el mayor índice $i \in \{2, \ldots, L-1\}$ tal que $n_i > 0$
3. $n_i' \leftarrow \max(0, n_i - x)$
4. Si tras la reducción todas las capas ocultas quedan en 0 → retornar $\emptyset$
5. Retornar $T' = [n_1, \ldots, n_i', \ldots, n_L]$

#### Reconstrucción de red

La función `reconstruir` crea una nueva red con topología reducida $T'$ preservando los pesos existentes:

1. Para cada transición de capa $i$: recortar $W_i$ a la submatriz superior-izquierda de dimensiones $n'_{i+1} \times n'_i$
2. Recortar cada $b_i$ a las primeras $n'_{i+1}$ componentes
3. Si alguna capa oculta tiene 0 neuronas: eliminarla y colapsar las conexiones adyacentes

## 3. Hiperparámetros

| Símbolo | Parámetro | Tipo | Defecto | Restricción | Efecto |
|---------|-----------|------|---------|-------------|--------|
| $N$ | `tamano_nube` | `Int` | 10 | $\geq 1$ | Número de redes generadas. Valores mayores incrementan la probabilidad de encontrar una red viable a costa de mayor tiempo de cómputo. Controla la amplitud de la exploración del espacio de pesos aleatorios. |
| $T_0$ | `topologia_inicial` | `Vector{Int}` | `[2, 4, 1]` | $\geq 3$ elementos; capas ocultas $\geq 1$ | Arquitectura de partida. Define el espacio de búsqueda: todas las sub-topologías alcanzables por reducción progresiva. Topologías iniciales más grandes permiten explorar arquitecturas más diversas pero incrementan el costo computacional. |
| $\theta$ | `umbral_acierto` | `Float64` | 0.5 | $\in [0, 1]$ | Precisión mínima para considerar una red viable. Valores bajos facilitan encontrar candidatos pero pueden seleccionar redes de baja calidad. Valores altos son más selectivos pero pueden resultar en búsquedas fallidas. |
| $x$ | `neuronas_eliminar` | `Int` | 1 | $\geq 1$ | Neuronas removidas por paso de reducción. Valores pequeños (1) producen una exploración granular del espacio de topologías. Valores grandes saltan sub-topologías intermedias, reduciendo el costo pero potencialmente omitiendo arquitecturas óptimas. |
| $E$ | `epocas_refinamiento` | `Int` | 1000 | — | Épocas de retropropagación aplicadas a la mejor red al final del proceso. Más épocas permiten mayor convergencia pero incrementan el tiempo de ejecución. El refinamiento ocurre una sola vez sobre la red seleccionada. |
| $\eta$ | `tasa_aprendizaje` | `Float64` | 0.1 | — | Tasa de aprendizaje para el descenso de gradiente durante el refinamiento. Valores altos aceleran la convergencia pero pueden causar oscilaciones. Valores bajos son más estables pero requieren más épocas. |
| — | `semilla` | `Int` | 42 | — | Semilla para el generador de números pseudoaleatorios (MersenneTwister). Garantiza reproducibilidad: la misma semilla con los mismos datos produce resultados idénticos. Semillas diferentes generan nubes de redes distintas. |


## 4. Análisis de complejidad computacional

Sea $N$ el tamaño de la nube, $L$ el número de capas de la topología inicial, $n_{\max}$ el
máximo de neuronas por capa, $M$ el número de muestras del dataset, y $E$ las épocas de
refinamiento.

### 4.1 Generación de la nube

Generar una red requiere inicializar $L - 1$ matrices de pesos y vectores de biases. El costo
de inicializar los pesos de una red es proporcional al número total de conexiones:

$$O\left(\sum_{i=1}^{L-1} n_{i+1} \cdot n_i\right)$$

Para $N$ redes:

$$O\left(N \cdot \sum_{i=1}^{L-1} n_{i+1} \cdot n_i\right)$$

### 4.2 Evaluación de una red

Una evaluación consiste en ejecutar feedforward sobre las $M$ muestras. El costo de un
feedforward es la suma de las multiplicaciones matriciales por capa:

$$O\left(M \cdot \sum_{i=1}^{L-1} n_{i+1} \cdot n_i\right)$$

### 4.3 Exploración por red

Para cada red, el número de pasos de reducción $R_j$ depende de la topología y del parámetro
$x$. En el peor caso, con $x = 1$, el total de reducciones por red es:

$$R_j = \sum_{i=2}^{L-1} n_i$$

es decir, la suma de neuronas en todas las capas ocultas. Con $x > 1$, el número de pasos se
reduce proporcionalmente: $R_j \approx \frac{1}{x}\sum_{i=2}^{L-1} n_i$.

Cada paso de reducción implica una evaluación y una reconstrucción. El costo de reconstrucción
es $O(\sum n_{i+1} \cdot n_i)$ (copiar submatrices), dominado por el costo de evaluación.

El costo total de exploración para una red es:

$$O\left(R_j \cdot M \cdot \sum_{i=1}^{L-1} n_{i+1} \cdot n_i\right)$$

### 4.4 Costo total de exploración (todas las redes)

$$O\left(N \cdot R \cdot M \cdot \sum_{i=1}^{L-1} n_{i+1} \cdot n_i\right)$$

donde $R = \max_j R_j$ es el número máximo de reducciones por red. Nótese que a medida que
la topología se reduce, el costo de cada evaluación individual disminuye, por lo que esta
cota es conservadora.

### 4.5 Refinamiento

El refinamiento aplica retropropagación durante $E$ épocas sobre las $M$ muestras. Cada
iteración de entrenamiento tiene costo similar al feedforward más la retropropagación
(aproximadamente $2\times$ el costo del feedforward):

$$O\left(E \cdot M \cdot \sum_{i=1}^{L'-1} n'_{i+1} \cdot n'_i\right)$$

donde $T' = [n'_1, \ldots, n'_{L'}]$ es la topología de la red seleccionada (potencialmente
mucho menor que $T_0$).

### 4.6 Complejidad total

$$O\left(N \cdot R \cdot M \cdot \sum n_{i+1} \cdot n_i + E \cdot M \cdot \sum n'_{i+1} \cdot n'_i\right)$$

El término dominante depende de la relación entre los parámetros:

- **Para $E$ grande** (típico: $E = 1000$): el refinamiento domina, especialmente si la red
  seleccionada no es significativamente más pequeña que la original.
- **Para $N$ grande con $E$ pequeño**: la exploración domina.
- **En la práctica**, la reducción progresiva disminuye el tamaño de las redes evaluadas en
  cada paso, lo que hace que el costo real de exploración sea menor que la cota teórica.

### 4.7 Complejidad espacial

$$O\left(N \cdot \sum_{i=1}^{L-1} n_{i+1} \cdot n_i\right)$$

Dominada por el almacenamiento de las $N$ redes de la nube. Cada red almacena $L - 1$ matrices
de pesos y vectores de biases. Durante la exploración, solo se mantiene una red activa
adicional ($R_{\text{actual}}$) y la mejor red global ($R^*$).
