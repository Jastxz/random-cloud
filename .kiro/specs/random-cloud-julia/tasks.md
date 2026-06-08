# Plan de Implementación: RandomCloud.jl

## Visión General

Implementación incremental del paquete RandomCloud.jl siguiendo el orden de dependencias entre componentes. Cada tarea construye sobre las anteriores, comenzando por la estructura del proyecto y los tipos base, avanzando hacia la lógica del motor, y finalizando con integración, ejemplo, documentación y benchmarks.

## Tareas

- [x] 1. Configurar estructura del proyecto y módulo principal
  - [x] 1.1 Crear Project.toml con dependencias y extras de test
    - Definir `[deps]` con Random y LinearAlgebra (UUIDs correctos)
    - Definir `[extras]` con Test, BenchmarkTools y Supposition
    - Definir `[targets]` test
    - _Requisitos: 9.1, 9.2, 9.3_

  - [x] 1.2 Crear src/RandomCloud.jl como esqueleto del módulo principal
    - Declarar `module RandomCloud` con todos los exports
    - Incluir los archivos fuente con `include()`
    - Crear archivos fuente vacíos (configuracion.jl, red_neuronal.jl, politica.jl, evaluacion.jl, motor.jl, informe.jl) para que el módulo cargue sin errores
    - _Requisitos: 7.1, 7.2, 7.3_

  - [x] 1.3 Crear test/runtests.jl como entry point de tests
    - Incluir `using RandomCloud` y `using Test`
    - Incluir los archivos de test con `include()`
    - Crear archivos de test vacíos para que `Pkg.test()` pase sin errores
    - _Requisitos: 10.6_

- [x] 2. Implementar ConfiguracionNube
  - [x] 2.1 Implementar struct ConfiguracionNube en src/configuracion.jl
    - Definir struct inmutable con los 7 campos
    - Implementar constructor interno con keyword arguments y valores por defecto
    - Implementar todas las validaciones (tamano_nube, topologia, umbral, neuronas_eliminar, capas ocultas)
    - Realizar copia defensiva de topologia_inicial
    - _Requisitos: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8_

  - [x] 2.2 Escribir tests unitarios para ConfiguracionNube en test/test_configuracion.jl
    - Verificar valores por defecto exactos
    - Verificar que cada validación lanza ArgumentError con el mensaje correcto
    - Verificar copia defensiva de topología
    - _Requisitos: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 10.1_

  - [ ]* 2.3 Escribir test PBT para ConfiguracionNube — Propiedad 1
    - **Propiedad 1: Validación rechaza entradas inválidas**
    - Generar parámetros aleatorios donde al menos uno viola su restricción y verificar que se lanza ArgumentError
    - Mínimo 100 iteraciones con Supposition.jl
    - `# Feature: random-cloud-julia, Property 1: Validación rechaza entradas inválidas`
    - **Valida: Requisitos 1.3, 1.4, 1.5, 1.6, 1.7**

  - [ ]* 2.4 Escribir test PBT para ConfiguracionNube — Propiedad 2
    - **Propiedad 2: Almacena valores correctamente**
    - Generar parámetros válidos aleatorios, construir ConfiguracionNube y verificar que cada campo retorna el valor proporcionado
    - Mínimo 100 iteraciones con Supposition.jl
    - `# Feature: random-cloud-julia, Property 2: Almacena valores correctamente`
    - **Valida: Requisitos 1.1**

  - [x] 2.5 Escribir test PBT para ConfiguracionNube — Propiedad 3
    - **Propiedad 3: Copia defensiva de topología**
    - Generar topología válida, construir ConfiguracionNube, mutar el vector original y verificar que la topología almacenada no cambió
    - Mínimo 100 iteraciones con Supposition.jl
    - `# Feature: random-cloud-julia, Property 3: Copia defensiva de topología`
    - **Valida: Requisitos 1.8, 2.3**

- [x] 3. Checkpoint — Verificar estructura base
  - Asegurar que `Pkg.test()` pasa sin errores, preguntar al usuario si hay dudas.

- [x] 4. Implementar RedNeuronal
  - [x] 4.1 Implementar struct RedNeuronal y constructor en src/red_neuronal.jl
    - Definir struct inmutable con topologia, pesos y biases
    - Implementar constructor `RedNeuronal(topologia, rng)` con pesos en [-1.0, 1.0]
    - Implementar copia defensiva de topología
    - Implementar función `sigmoid(x)` y `sigmoid_deriv(x)`
    - _Requisitos: 2.1, 2.2, 2.3, 2.5_

  - [x] 4.2 Implementar feedforward en src/red_neuronal.jl
    - Propagar entrada capa por capa aplicando `sigmoid.(W * x .+ b)`
    - Retornar vector de salida con dimensión igual a la última capa
    - _Requisitos: 2.4, 2.7, 2.8_

  - [x] 4.3 Implementar entrenar! en src/red_neuronal.jl
    - Forward pass almacenando activaciones por capa
    - Retropropagar error multiplicando por sigmoid_deriv
    - Actualizar pesos y biases con tasa de aprendizaje
    - _Requisitos: 2.6_

  - [x] 4.4 Implementar reconstruir en src/red_neuronal.jl
    - Recortar cada matriz de pesos Wᵢ a submatriz superior-izquierda de dimensiones t'ᵢ × t'ᵢ₋₁
    - Recortar cada vector de biases bᵢ a las primeras t'ᵢ componentes
    - Manejar capas ocultas con 0 neuronas: eliminar capa y colapsar conexiones
    - Retornar nueva RedNeuronal sin modificar la original
    - _Requisitos: 2.9, 2.10, 2.11, 2.12_

  - [x] 4.5 Escribir tests unitarios para RedNeuronal en test/test_red_neuronal.jl
    - Verificar dimensiones de pesos y biases tras construcción
    - Verificar dimensión de salida de feedforward
    - Verificar que entrenar! modifica pesos (caso concreto)
    - Verificar reconstruir con topología reducida (caso concreto)
    - Verificar reconstruir con capa oculta en 0 neuronas (edge case 2.12)
    - _Requisitos: 2.1, 2.2, 2.4, 2.6, 2.9, 2.10, 2.11, 2.12_

  - [ ]* 4.6 Escribir test PBT para RedNeuronal — Propiedad 4
    - **Propiedad 4: Pesos y biases en rango [-1.0, 1.0]**
    - Generar topologías válidas aleatorias, construir RedNeuronal y verificar que todos los valores están en [-1.0, 1.0]
    - Mínimo 100 iteraciones con Supposition.jl
    - `# Feature: random-cloud-julia, Property 4: Pesos y biases en rango [-1.0, 1.0]`
    - **Valida: Requisitos 2.2**

  - [ ]* 4.7 Escribir test PBT para RedNeuronal — Propiedad 5
    - **Propiedad 5: Dimensión y rango de salida de feedforward**
    - Generar RedNeuronal y entrada aleatorias, verificar que la salida tiene dimensión correcta y cada componente está en (0.0, 1.0)
    - Mínimo 100 iteraciones con Supposition.jl
    - `# Feature: random-cloud-julia, Property 5: Dimensión y rango de salida de feedforward`
    - **Valida: Requisitos 2.7, 2.8**

  - [ ]* 4.8 Escribir test PBT para RedNeuronal — Propiedad 6
    - **Propiedad 6: Corrección de la función sigmoide**
    - Generar valores Float64 aleatorios, verificar que sigmoid(x) == 1.0 / (1.0 + exp(-x)) y resultado en (0.0, 1.0)
    - Mínimo 100 iteraciones con Supposition.jl
    - `# Feature: random-cloud-julia, Property 6: Corrección de sigmoid`
    - **Valida: Requisitos 2.5**

  - [x] 4.9 Escribir test PBT para RedNeuronal — Propiedad 7
    - **Propiedad 7: Entrenamiento modifica pesos**
    - Generar RedNeuronal, entrada y objetivo aleatorios, ejecutar entrenar! y verificar que al menos un peso cambió
    - Mínimo 100 iteraciones con Supposition.jl
    - `# Feature: random-cloud-julia, Property 7: Entrenamiento modifica pesos`
    - **Valida: Requisitos 2.6**

  - [x] 4.10 Escribir test PBT para RedNeuronal — Propiedad 8
    - **Propiedad 8: Reconstruir preserva pesos mediante recorte de submatrices**
    - Generar RedNeuronal y topología reducida válida, verificar que los pesos recortados coinciden con la submatriz superior-izquierda original
    - Mínimo 100 iteraciones con Supposition.jl
    - `# Feature: random-cloud-julia, Property 8: Reconstruir preserva pesos`
    - **Valida: Requisitos 2.9, 2.10, 2.11**

  - [x] 4.11 Escribir test PBT para RedNeuronal — Propiedad 13
    - **Propiedad 13: Semillas diferentes producen redes diferentes**
    - Generar pares de semillas distintas con la misma topología, verificar que los pesos generados son diferentes
    - Mínimo 100 iteraciones con Supposition.jl
    - `# Feature: random-cloud-julia, Property 13: Semillas diferentes → redes diferentes`
    - **Valida: Requisitos 8.2**

- [x] 5. Implementar PoliticaEliminacion
  - [x] 5.1 Implementar tipos y función siguiente_reduccion en src/politica.jl
    - Definir tipo abstracto PoliticaEliminacion
    - Definir struct PoliticaSecuencial como subtipo
    - Implementar siguiente_reduccion: buscar última capa oculta con neuronas > 0, restar n, retornar nothing si no es posible
    - Retornar copia nueva de la topología sin modificar la original
    - _Requisitos: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

  - [x] 5.2 Escribir tests unitarios para PoliticaSecuencial en test/test_politica.jl
    - Verificar que PoliticaSecuencial es subtipo de PoliticaEliminacion
    - Verificar reducción normal (caso concreto)
    - Verificar retorno de nothing cuando no hay reducción posible
    - Verificar que la topología original no se modifica
    - Verificar que capa de entrada y salida permanecen iguales
    - _Requisitos: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 10.2_

  - [x] 5.3 Escribir test PBT para PoliticaSecuencial — Propiedad 9
    - **Propiedad 9: Propiedades de reducción válida**
    - Generar topologías con capas ocultas > 0, verificar que la reducción preserva entrada/salida, reduce exactamente una capa, y no modifica la original
    - Mínimo 100 iteraciones con Supposition.jl
    - `# Feature: random-cloud-julia, Property 9: Propiedades de reducción válida`
    - **Valida: Requisitos 3.3, 3.6, 3.7, 3.8**

  - [x] 5.4 Escribir test PBT para PoliticaSecuencial — Propiedad 10
    - **Propiedad 10: siguiente_reduccion retorna nothing cuando no hay reducción posible**
    - Generar topologías donde todas las capas ocultas tienen 0 neuronas o la reducción las dejaría todas en 0, verificar retorno de nothing
    - Mínimo 100 iteraciones con Supposition.jl
    - `# Feature: random-cloud-julia, Property 10: Reducción imposible → nothing`
    - **Valida: Requisitos 3.4, 3.5**

- [x] 6. Implementar Evaluacion
  - [x] 6.1 Implementar función evaluar en src/evaluacion.jl
    - Para cada columna de entradas, ejecutar feedforward
    - Comparar argmax(salida) con argmax(objetivo)
    - Retornar aciertos / total_muestras como Float64 en [0.0, 1.0]
    - _Requisitos: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [x] 6.2 Escribir tests unitarios para evaluar en test/test_evaluacion.jl
    - Verificar con red y datos conocidos que la proporción es correcta
    - Verificar caso de 100% acierto y 0% acierto
    - _Requisitos: 4.1, 4.2, 4.5_

  - [ ]* 6.3 Escribir test PBT para evaluar — Propiedad 11
    - **Propiedad 11: Evaluación retorna proporción correcta**
    - Generar RedNeuronal, entradas y objetivos aleatorios, verificar que el resultado está en [0.0, 1.0] y es igual a k/m calculado manualmente
    - Mínimo 100 iteraciones con Supposition.jl
    - `# Feature: random-cloud-julia, Property 11: Evaluación = k/m`
    - **Valida: Requisitos 4.1, 4.5**

- [x] 7. Checkpoint — Verificar componentes individuales
  - Asegurar que todos los tests pasan, preguntar al usuario si hay dudas.

- [x] 8. Implementar InformeNube
  - [x] 8.1 Implementar struct InformeNube en src/informe.jl
    - Definir struct inmutable con los 7 campos: mejor_red, precision, topologia_final, total_redes_evaluadas, total_reducciones, tiempo_ejecucion_ms, exitoso
    - _Requisitos: 5.1, 5.2_

- [x] 9. Implementar MotorNube
  - [x] 9.1 Implementar struct MotorNube y función ejecutar en src/motor.jl
    - Definir mutable struct MotorNube con config, entradas y objetivos
    - Implementar función ejecutar con el algoritmo completo:
      - Inicializar RNG local con MersenneTwister(semilla)
      - Generar nube de N redes con topología inicial
      - Para cada red, explorar todas las sub-topologías via reducción progresiva
      - Evaluar cada red/sub-topología sin entrenamiento, rastrear mejor red global
      - Usar reconstruir para preservar pesos al reducir topología
      - Si se encontró red viable: refinar UNA SOLA VEZ con backpropagation al final
      - Registrar tiempo, contadores de evaluaciones y reducciones
      - Retornar InformeNube
    - _Requisitos: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10, 6.11, 6.12, 8.1, 8.2, 8.3_

  - [x] 9.2 Escribir tests unitarios para MotorNube en test/test_motor.jl
    - Test XOR: ejecutar con topología [2, 8, 4, 2] y verificar exitoso == true
    - Verificar que el informe contiene campos válidos (precision > 0, total_redes_evaluadas > 0, etc.)
    - Verificar caso de fallo: umbral_acierto = 1.0 con topología mínima para forzar exitoso == false
    - _Requisitos: 6.1, 6.2, 6.5, 6.9, 10.3_

  - [x] 9.3 Escribir test PBT para MotorNube — Propiedad 12
    - **Propiedad 12: Ejecución determinista**
    - Generar ConfiguracionNube con semillas aleatorias y datos XOR, ejecutar dos veces y verificar que topologia_final, precision y exitoso son idénticos
    - Mínimo 100 iteraciones con Supposition.jl
    - `# Feature: random-cloud-julia, Property 12: Ejecución determinista`
    - **Valida: Requisitos 8.1, 6.3**

  - [x] 9.4 Escribir test PBT para MotorNube — Propiedad 14
    - **Propiedad 14: Ejecución exitosa implica precisión ≥ umbral**
    - Generar configuraciones con umbrales bajos y datos XOR, ejecutar y verificar que si exitoso == true entonces precision ≥ umbral_acierto
    - Mínimo 100 iteraciones con Supposition.jl
    - `# Feature: random-cloud-julia, Property 14: Exitoso → precisión ≥ umbral`
    - **Valida: Requisitos 6.5**

  - [x] 9.5 Escribir test PBT para MotorNube — Propiedad 15
    - **Propiedad 15: Ejecución fallida implica campos nothing**
    - Generar configuraciones con umbrales altos (cercanos a 1.0) y topologías mínimas, ejecutar y verificar que si exitoso == false entonces mejor_red == nothing y topologia_final == nothing
    - Mínimo 100 iteraciones con Supposition.jl
    - `# Feature: random-cloud-julia, Property 15: Fallido → campos nothing`
    - **Valida: Requisitos 5.4, 6.9**

  - [x] 9.6 Escribir test PBT para MotorNube — Propiedad 16
    - **Propiedad 16: Metadatos del informe son consistentes**
    - Generar configuraciones aleatorias y datos XOR, ejecutar y verificar que tiempo_ejecucion_ms > 0 y total_redes_evaluadas ≥ tamano_nube
    - Mínimo 100 iteraciones con Supposition.jl
    - `# Feature: random-cloud-julia, Property 16: Metadatos consistentes`
    - **Valida: Requisitos 6.10, 6.11**

- [x] 10. Checkpoint — Verificar motor y flujo completo
  - Asegurar que todos los tests pasan, preguntar al usuario si hay dudas.

- [x] 11. Implementar test de integración
  - [x] 11.1 Escribir test de integración completo en test/test_integracion.jl
    - Ejecutar flujo completo: ConfiguracionNube → MotorNube → ejecutar → InformeNube
    - Verificar todos los campos del InformeNube (exitoso, mejor_red, topologia_final, precision, contadores, tiempo)
    - Verificar caso exitoso y caso fallido
    - Verificar inmutabilidad de InformeNube
    - _Requisitos: 5.1, 5.2, 5.3, 5.4, 10.4_

- [x] 12. Crear ejemplo de uso
  - [x] 12.1 Crear examples/xor.jl con ejemplo funcional del método
    - Definir dataset XOR en formato column-major
    - Crear ConfiguracionNube con parámetros adecuados para XOR
    - Crear MotorNube, ejecutar e imprimir resultados
    - Imprimir topología final, precisión y reducción de parámetros si exitoso
    - _Requisitos: 11.1, 11.2, 11.3, 11.4, 11.5_

- [x] 13. Crear documentación del método
  - [x] 13.1 Crear docs/metodo.md con descripción formal del método
    - Definición formal del algoritmo con pasos numerados
    - Descripción de hiperparámetros y su efecto
    - Análisis de complejidad computacional
    - _Requisitos: 12.1, 12.2, 12.3, 12.4_

- [x] 14. Implementar benchmark de escalabilidad
  - [x] 14.1 Crear test/benchmark_escalabilidad.jl
    - Comparar rendimiento con distintos tamaños de nube usando BenchmarkTools.jl
    - Medir tiempo de ejecución para tamaños crecientes
    - _Requisitos: 10.5_

- [x] 15. Checkpoint final — Verificar paquete completo
  - Ejecutar `Pkg.test()` y asegurar que todos los tests pasan sin errores. Preguntar al usuario si hay dudas.

## Notas

- Las tareas marcadas con `*` son opcionales y pueden omitirse para un MVP más rápido
- Cada tarea referencia los requisitos específicos que valida
- Los checkpoints permiten validación incremental
- Los tests PBT validan propiedades universales de corrección (mínimo 100 iteraciones cada uno)
- Los tests unitarios validan ejemplos concretos y casos borde
- Se usa Supposition.jl como biblioteca de property-based testing
