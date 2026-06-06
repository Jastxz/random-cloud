# Fase 2: Exploración Estructural de la Nube Aleatoria

## Resumen

La Fase 1 (método actual) encuentra la anchura óptima por capa mediante poda secuencial
de neuronas. La Fase 2 extiende la exploración al espacio de **profundidad y estructura**:
eliminar capas enteras y redistribuir capas anchas en capas más estrechas y profundas.

El objetivo es encontrar sub-arquitecturas más eficientes que la poda de ancho sola
no puede descubrir, sin añadir nunca más neuronas que las originales.

---

## Operaciones estructurales

La Fase 2 define dos operaciones complementarias sobre la topología resultante de la Fase 1
(o directamente sobre la topología original):

### 1. Colapso de capas (reducción vertical)

**Qué hace:** elimina una capa oculta intermedia, componiendo las transformaciones
lineales adyacentes para preservar los pesos existentes.

**Ejemplo:**
```
Original:    [784, 128, 64, 32, 10]
             W1(128×784), W2(64×128), W3(32×64), W4(10×32)

Colapso(capa 128):  [784, 64, 32, 10]
                    W_nueva = W2 × W1 (64×784), W3, W4
                    → Compone las dos transformaciones en una sola matriz

Colapso(capa 64):   [784, 128, 32, 10]
                    W1, W_nueva = W3 × W2 (32×128), W4
```

**Reglas:**
- Solo se eliminan capas ocultas (nunca entrada ni salida).
- Los pesos de la nueva conexión se obtienen componiendo las matrices adyacentes:
  si se elimina la capa i, la nueva conexión es `W[i+1] × W[i]` (producto matricial).
  Esto preserva la transformación lineal que ya existía, sin generar pesos nuevos.
- Los biases de la capa eliminada se absorben: `b_nuevo = W[i+1] × b[i] + b[i+1]`.
- Se evalúa la nueva topología sin entrenamiento (forward pass), igual que en Fase 1.

**Nota:** la activación no lineal intermedia se pierde, lo cual cambia el comportamiento
de la red. Eso es precisamente lo que exploramos: si esa no-linealidad era necesaria
o era un cuello de botella.

### 2. Redistribución de capas (split)

**Qué hace:** divide una capa ancha en 2 o 3 sub-capas más estrechas, reutilizando
los pesos existentes mediante partición por filas y conexiones identidad.

**Mecanismo de partición de pesos:**

Dada una capa con peso W(128×n_in) y bias b(128), al dividirla en dos sub-capas de 64:

```
Original:        W(128×n_in), b(128)

Sub-capa 1:      W1 = W[1:64, :]      (primeras 64 filas)
                 b1 = b[1:64]

Conexión:        I(64×64)             (matriz identidad — mínima distorsión)

Sub-capa 2:      W2 = W[65:128, :]... → se convierte en W2(64×64) = I
                 El peso de salida de la sub-capa 2 toma las filas 65:128
                 de la matriz que conectaba con la capa siguiente.
```

**Esquema detallado para split de capa i (ancho W) en dos sub-capas (A, B):**

```
Antes:   capa[i-1] --W_i(W×n_in)--> capa[i] --W_{i+1}(n_out×W)--> capa[i+1]

Después: capa[i-1] --W_i[1:A, :](A×n_in)--> sub1 --I(B×A)--> sub2 --W_{i+1}[:, :](n_out×B)--> capa[i+1]

Donde:
  - sub1 recibe los pesos de las primeras A filas de W_i, bias b_i[1:A]
  - La conexión sub1→sub2 es la identidad (si A == B) o una submatriz identidad
  - sub2→capa[i+1] usa las columnas correspondientes de W_{i+1}
```

**Variantes que se exploran para cada split:**
- Orden normal: filas [1:A] → sub1, filas [A+1:W] → sub2
- Orden invertido: filas [A+1:W] → sub1, filas [1:A] → sub2
- Divisiones asimétricas: (W/2, W/2), (W×2/3, W/3), (W×3/4, W/4)

**Ejemplo (split profundidad 1 → 2 capas):**
```
Original:        [784, 128, 10]
                 W1(128×784), W2(10×128)

Split(capa 2):   [784, 64, 64, 10]
                 W1_new = W1[1:64, :]    (64×784)  ← primeras 64 filas
                 W_inter = I(64×64)       ← identidad
                 W2_new = W2[:, 65:128]   (10×64)  ← columnas 65:128 de W2

Split invertido: [784, 64, 64, 10]
                 W1_new = W1[65:128, :]  (64×784)  ← últimas 64 filas
                 W_inter = I(64×64)       ← identidad
                 W2_new = W2[:, 1:64]    (10×64)  ← columnas 1:64 de W2
```

**Ejemplo (split profundidad 2 → 3 capas):**
```
Original:        [784, 128, 10]

Split(capa 2):   [784, 48, 48, 32, 10]
                 W1_new = W1[1:48, :]     (48×784)
                 W_inter1 = I(48×48)       ← identidad
                 W_inter2 = I(32×48)       ← submatriz identidad (padding zeros)
                 W2_new = W2[:, 97:128]   (10×32)
```

**Reglas:**
- Los pesos se reutilizan siempre — nunca se generan pesos aleatorios nuevos.
- Las conexiones entre sub-capas son matrices identidad (mínima distorsión).
- El ancho de cada sub-capa generada es ≥ ancho_minimo (default: 4 neuronas).
- La profundidad de split es máximo 2 (una capa → máximo 3 sub-capas).

---

## Restricción de pesos

**Principio fundamental:** nunca se generan pesos aleatorios nuevos en la Fase 2.
Todos los pesos proceden de la red original:

- **Colapso:** composición matricial `W[i+1] × W[i]` + absorción de biases.
- **Redistribución:** partición por filas de la matriz original + identidad como conexión.

Esto mantiene la propiedad core del método: la Fase 1 ya encontró la mejor
combinación de pesos aleatorios. La Fase 2 busca si reorganizar esos mismos pesos
en una estructura distinta produce un resultado aún mejor.

---

## Generación de candidatos

Dada una topología base T (resultado de Fase 1, o la topología original), se generan
candidatos estructurales combinando:

### Colapso:
- Para cada capa oculta c_i en T: generar T sin c_i (pesos compuestos W[i+1]×W[i]).
- Si hay L capas ocultas, hay L candidatos de colapso simple.

### Redistribución (split profundidad 1):
- Para cada capa oculta c_i con ancho W:
  - Generar divisiones en 2 sub-capas: (W/2, W/2), (W×2/3, W/3), (W×3/4, W/4)...
  - Para cada división: variante orden normal + variante orden invertido.
  - Filtrar: ambas sub-capas ≥ ancho_minimo.

### Redistribución (split profundidad 2):
- Para cada capa oculta c_i con ancho W:
  - Generar divisiones en 3 sub-capas con combinaciones de anchos.
  - Filtrar: todas las sub-capas ≥ ancho_minimo.

### Combinaciones:
- Opcionalmente, combinar colapso + redistribución en la misma topología candidata
  (colapsar una capa y redistribuir otra). Esto multiplica el espacio pero es acotado.

---

## Algoritmo de la Fase 2

```
Entrada: red ganadora de Fase 1 (con sus pesos), topología original T_orig, configuración
Salida:  mejor topología encontrada + informe

1. Tomar la red ganadora de Fase 1 (la mejor inicialización encontrada con sus pesos).

2. Generar candidatos estructurales desde esa red:
   - Colapsos: componer matrices adyacentes para cada capa eliminable.
   - Splits: particionar filas + identidad para cada capa divisible.
   - Combinaciones (opcional): colapso + split en la misma red.

3. Para cada candidato:
   a. Construir la red candidata reutilizando los pesos de la red Fase 1
      (composición, partición, identidad — sin aleatorios).
   b. Evaluar sin entrenamiento (forward pass sobre datos).
   c. Registrar precisión.

4. Seleccionar las K mejores topologías candidatas.

5. Para cada top-K candidata:
   a. Aplicar refinamiento (entrenamiento con backprop).
   b. Opcionalmente, ejecutar Fase 1 (poda de ancho) sobre la candidata.
   c. Registrar precisión final.

6. Comparar con el resultado de Fase 1 pura.
   Retornar la mejor red global.
```

---

## Hiperparámetros nuevos (Fase 2)

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `explorar_estructura` | Bool | false | Activar Fase 2 |
| `max_profundidad_split` | Int | 2 | Máxima profundidad de redistribución (1 o 2) |
| `ancho_minimo_split` | Int | 4 | Ancho mínimo de sub-capas generadas |
| `n_candidatos_estructura` | Int | 10 | Top-K candidatos a refinar |

---

## Flujo completo del método extendido

```
┌─────────────────────────────────────────────────┐
│            RED ORIGINAL (grande)                 │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │   FASE 1 (Nube)        │
          │   Poda de ancho        │
          │   → mejor_red (pesos)  │
          └────────────┬───────────┘
                       │
                       ▼
          ┌────────────────────────────────────┐
          │   FASE 2 (Estructura)              │
          │   Toma los PESOS de mejor_red      │
          │   Genera candidatos:               │
          │     • Colapso (W[i+1]×W[i])        │
          │     • Split (filas + identidad)    │
          │   Evalúa sin entrenamiento         │
          │   Top-K → refina con backprop      │
          └────────────┬───────────────────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  COMPARAR RESULTADOS   │
          │  Fase 1 vs Fase 2      │
          └────────────┬───────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  RED FINAL ÓPTIMA      │
          └────────────────────────┘
```

---

## Ejemplo concreto

Red original: `[784, 128, 128, 64, 10]` (MNIST)
Pesos: W1(128×784), W2(128×128), W3(64×128), W4(10×64)

Supongamos que la Fase 1 la poda a `[784, 96, 80, 48, 10]` con pesos P1, P2, P3, P4.

**Candidatos de colapso (sobre la red Fase 1):**
- Eliminar capa 96: `[784, 80, 48, 10]`, peso nuevo = P2 × P1 (80×784)
- Eliminar capa 80: `[784, 96, 48, 10]`, peso nuevo = P3 × P2 (48×96)
- Eliminar capa 48: `[784, 96, 80, 10]`, peso nuevo = P4 × P3 (10×80)

**Candidatos de redistribución (split prof. 1 de la capa 96):**
- `[784, 48, 48, 80, 48, 10]`:
  - P1_new = P1[1:48, :] (48×784), I(48×48), P1_rest afecta a P2...
  - Se ajustan las conexiones siguientes con las columnas correspondientes.
- Variante invertida: P1[49:96, :] primero.

**Candidatos de redistribución (split prof. 2 de la capa 96):**
- `[784, 32, 32, 32, 80, 48, 10]`:
  - P1[1:32, :], I(32×32), I(32×32), ...

Se evalúan todos con forward pass, se refinan los K mejores, se elige el ganador.

---

## Justificación teórica

1. **La Lottery Ticket Hypothesis** dice que redes grandes contienen sub-redes ganadoras.
   Pero solo busca sub-redes con la misma profundidad — nunca explora si
   redistribuir la estructura produce mejores "tickets".

2. **El método de la Nube (Fase 1)** busca sub-redes en ancho pero no en profundidad.
   La Fase 2 completa la exploración.

3. **La reutilización de pesos** es clave: la Fase 1 ya determinó que esta
   inicialización concreta es la mejor entre N aleatorias. La Fase 2 pregunta:
   "¿hay una reorganización geométrica de estos mismos pesos que funcione aún mejor?"

4. **La redistribución** (split) explora una hipótesis concreta: que una transformación
   profunda y estrecha puede ser equivalente o superior a una ancha y superficial,
   usando los mismos valores numéricos en los pesos.

5. **El colapso** con composición matricial (`W2×W1`) es lossless en el componente
   lineal — solo pierde la no-linealidad intermedia. Si esa no-linealidad no aportaba,
   la red colapsada funcionará igual o mejor con menos cómputo.

6. **Todo se evalúa sin entrenamiento previo** (en la pre-selección), manteniendo
   la propiedad core del método: nunca entrenas lo que no necesitas.

---

## Relación con la implementación actual

- `ConfiguracionNube` se extiende con los nuevos hiperparámetros.
- Se crea un nuevo módulo/archivo `src/fase2_estructura.jl`.
- `ejecutar()` en `motor.jl` invoca la Fase 2 si `explorar_estructura == true`.
- El `InformeNube` se extiende para reportar candidatos explorados, topología
  original vs topología final, y el tipo de operación que la encontró.
- Los tests PBT verifican que la Fase 2 nunca produce redes con más parámetros
  que la original.

---

## Estado

**Diseño**: completo (este documento)
**Implementación**: pendiente
**Paper**: se incorporará como extensión tras validación experimental
