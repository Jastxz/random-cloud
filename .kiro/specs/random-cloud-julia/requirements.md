# Documento de Requisitos — RandomCloud.jl

## Introducción

RandomCloud.jl es un paquete Julia que implementa el Método de la Nube Aleatoria, un enfoque original de búsqueda de arquitecturas de redes neuronales (Neural Architecture Search). El método encuentra arquitecturas mínimas de redes neuronales mediante la evaluación de múltiples redes sin entrenamiento y la reducción estructural progresiva. Este paquete sirve como implementación de referencia para una publicación académica.

## Glosario

- **ConfiguracionNube**: Estructura inmutable que contiene todos los hiperparámetros del método.
- **RedNeuronal**: Red neuronal feedforward con pesos, biases y topología definida.
- **Topologia**: Vector de enteros que define el número de neuronas por capa (entrada, ocultas, salida).
- **Nube**: Conjunto de N redes neuronales generadas con pesos aleatorios a partir de una misma topología.
- **PoliticaEliminacion**: Tipo abstracto que define la estrategia para reducir la topología de la red.
- **PoliticaSecuencial**: Política concreta que elimina neuronas desde la última capa oculta hacia la primera.
- **MotorNube**: Orquestador principal que ejecuta el ciclo completo del método.
- **InformeNube**: Estructura inmutable que contiene los resultados de una ejecución del método.
- **Evaluacion**: Módulo de funciones que calcula la precisión de una RedNeuronal sobre un dataset.
- **Feedforward**: Propagación hacia adelante de una entrada a través de las capas de la RedNeuronal.
- **Backpropagation**: Algoritmo de entrenamiento que ajusta pesos y biases mediante retropropagación del error.
- **Umbral_Acierto**: Valor en [0.0, 1.0] que define la precisión mínima aceptable para considerar una red viable.
- **Reduccion**: Operación de eliminar neuronas de una capa oculta según la PoliticaEliminacion activa.
- **Reconstruir**: Operación que crea una nueva red con topología reducida preservando los pesos de las neuronas no eliminadas (recorte de matrices y vectores).
- **Refinamiento**: Fase de entrenamiento (backpropagation) aplicada únicamente a la mejor red encontrada al final del proceso completo de búsqueda.

## Requisitos

### Requisito 1: Configuración del método

**Historia de usuario:** Como investigador, quiero definir los hiperparámetros del método en una estructura inmutable con valores por defecto y validación, para que la configuración sea reproducible y libre de errores.

#### Criterios de aceptación

1. THE ConfiguracionNube SHALL almacenar los campos: tamano_nube (Int), topologia_inicial (Vector{Int}), umbral_acierto (Float64), neuronas_eliminar (Int), epocas_refinamiento (Int), tasa_aprendizaje (Float64) y semilla (Int).
2. THE ConfiguracionNube SHALL proporcionar valores por defecto: tamano_nube=10, topologia_inicial=[2,4,1], umbral_acierto=0.5, neuronas_eliminar=1, epocas_refinamiento=1000, tasa_aprendizaje=0.1, semilla=42.
3. WHEN tamano_nube es menor que 1, THE ConfiguracionNube SHALL lanzar un ArgumentError con un mensaje que incluya el valor proporcionado.
4. WHEN topologia_inicial tiene menos de 3 elementos, THE ConfiguracionNube SHALL lanzar un ArgumentError indicando que se requieren al menos 3 capas.
5. WHEN umbral_acierto está fuera del rango [0.0, 1.0], THE ConfiguracionNube SHALL lanzar un ArgumentError con un mensaje que incluya el valor proporcionado.
6. WHEN neuronas_eliminar es menor que 1, THE ConfiguracionNube SHALL lanzar un ArgumentError con un mensaje que incluya el valor proporcionado.
7. WHEN alguna capa oculta de topologia_inicial tiene menos de 1 neurona, THE ConfiguracionNube SHALL lanzar un ArgumentError indicando que las capas ocultas requieren al menos 1 neurona.
8. THE ConfiguracionNube SHALL crear una copia defensiva de topologia_inicial para evitar mutaciones externas.


### Requisito 2: Red neuronal feedforward

**Historia de usuario:** Como investigador, quiero una red neuronal feedforward mínima con propagación hacia adelante, reconstrucción con pesos preservados y entrenamiento por backpropagation, para que el método pueda evaluar, reducir y refinar redes de distintas topologías.

#### Criterios de aceptación

1. THE RedNeuronal SHALL almacenar la topología, una lista de matrices de pesos y una lista de vectores de biases.
2. WHEN se construye una RedNeuronal a partir de una topología y un generador aleatorio (AbstractRNG), THE RedNeuronal SHALL inicializar los pesos y biases con valores aleatorios en el rango [-1.0, 1.0].
3. WHEN se construye una RedNeuronal, THE RedNeuronal SHALL crear una copia defensiva de la topología proporcionada.
4. WHEN se invoca feedforward con un vector de entrada, THE RedNeuronal SHALL propagar la entrada a través de todas las capas aplicando la función sigmoide como activación y retornar el vector de salida.
5. THE RedNeuronal SHALL implementar la función sigmoide como σ(x) = 1.0 / (1.0 + exp(-x)).
6. WHEN se invoca entrenar! con una entrada, un objetivo y una tasa de aprendizaje, THE RedNeuronal SHALL actualizar los pesos y biases mediante backpropagation con descenso de gradiente.
7. FOR ALL vectores de entrada válidos, la función feedforward SHALL retornar un vector cuya dimensión sea igual al número de neuronas de la última capa de la topología.
8. FOR ALL valores de salida de feedforward, cada componente SHALL estar en el rango (0.0, 1.0) dado que la función sigmoide produce valores en ese rango abierto.
9. WHEN se invoca reconstruir con una RedNeuronal y una nueva topología reducida, THE RedNeuronal SHALL crear una nueva RedNeuronal con la topología reducida preservando los pesos de las neuronas no eliminadas.
10. WHEN se reconstruye una red, THE reconstruir SHALL recortar cada matriz de pesos Wᵢ a la submatriz superior-izquierda de dimensiones t'ᵢ × t'ᵢ₋₁ de la nueva topología.
11. WHEN se reconstruye una red, THE reconstruir SHALL recortar cada vector de biases bᵢ a las primeras t'ᵢ componentes de la nueva topología.
12. WHEN una capa oculta de la nueva topología tiene 0 neuronas, THE reconstruir SHALL eliminar esa capa de la topología y colapsar las conexiones de las capas adyacentes.

### Requisito 3: Política de eliminación

**Historia de usuario:** Como investigador, quiero un sistema extensible de políticas de eliminación de neuronas, para que el método pueda probar distintas estrategias de reducción topológica.

#### Criterios de aceptación

1. THE PoliticaEliminacion SHALL ser un tipo abstracto que permita definir subtipos concretos mediante el sistema de despacho múltiple de Julia.
2. THE PoliticaSecuencial SHALL ser un subtipo concreto de PoliticaEliminacion.
3. WHEN se invoca siguiente_reduccion con una PoliticaSecuencial, una topología y un número n de neuronas a eliminar, THE PoliticaSecuencial SHALL retornar una nueva topología con n neuronas menos en la última capa oculta que tenga neuronas disponibles (mayor que 0).
4. WHEN todas las capas ocultas de la topología tienen 0 neuronas, THE PoliticaSecuencial SHALL retornar nothing.
5. WHEN la reducción resultaría en que todas las capas ocultas queden con 0 neuronas, THE PoliticaSecuencial SHALL retornar nothing.
6. THE PoliticaSecuencial SHALL buscar capas ocultas desde la última hacia la primera para determinar cuál reducir.
7. FOR ALL invocaciones de siguiente_reduccion, THE PoliticaSecuencial SHALL retornar una copia nueva de la topología sin modificar la topología original.
8. FOR ALL reducciones válidas, la capa de entrada y la capa de salida de la topología retornada SHALL permanecer iguales a las de la topología original.


### Requisito 4: Evaluación de redes

**Historia de usuario:** Como investigador, quiero evaluar la precisión de una red neuronal sobre un dataset, para que el método pueda seleccionar la mejor red de la nube y verificar si cumple el umbral de acierto.

#### Criterios de aceptación

1. WHEN se invoca evaluar con una RedNeuronal, una matriz de entradas y una matriz de objetivos, THE Evaluacion SHALL retornar la proporción de muestras clasificadas correctamente como un Float64 en [0.0, 1.0].
2. THE Evaluacion SHALL clasificar una muestra como correcta cuando el índice del valor máximo (argmax) de la salida de la RedNeuronal coincida con el índice del valor máximo del objetivo correspondiente.
3. THE Evaluacion SHALL tratar las columnas de la matriz de entradas como muestras individuales (formato column-major de Julia).
4. THE Evaluacion SHALL tratar las columnas de la matriz de objetivos como vectores objetivo individuales correspondientes a cada muestra.
5. WHEN el dataset tiene m muestras y k son clasificadas correctamente, THE Evaluacion SHALL retornar k/m.

### Requisito 5: Informe de resultados

**Historia de usuario:** Como investigador, quiero recibir un informe inmutable con los resultados de la ejecución del método, para poder analizar y comparar ejecuciones de forma reproducible.

#### Criterios de aceptación

1. THE InformeNube SHALL almacenar los campos: mejor_red (Union{RedNeuronal, Nothing}), precision (Float64), topologia_final (Union{Vector{Int}, Nothing}), total_redes_evaluadas (Int), total_reducciones (Int), tiempo_ejecucion_ms (Float64) y exitoso (Bool).
2. THE InformeNube SHALL ser una estructura inmutable.
3. WHEN la ejecución del método encuentra una red viable, THE InformeNube SHALL contener la mejor_red encontrada, la topologia_final reducida, la precision alcanzada y exitoso igual a true.
4. WHEN la ejecución del método no encuentra una red viable, THE InformeNube SHALL contener mejor_red igual a nothing, topologia_final igual a nothing y exitoso igual a false.

### Requisito 6: Motor del método (orquestador)

**Historia de usuario:** Como investigador, quiero un orquestador que ejecute el ciclo completo del Método de la Nube Aleatoria, para poder encontrar la arquitectura mínima de red neuronal que resuelva un problema dado.

#### Criterios de aceptación

1. THE MotorNube SHALL aceptar una ConfiguracionNube, una matriz de entradas y una matriz de objetivos para su construcción.
2. WHEN se invoca ejecutar sobre un MotorNube, THE MotorNube SHALL generar UNA SOLA nube de N redes neuronales aleatorias con la topología inicial, donde N es el tamano_nube de la ConfiguracionNube.
3. WHEN se genera la nube, THE MotorNube SHALL utilizar la semilla de la ConfiguracionNube para inicializar el generador de números aleatorios, garantizando reproducibilidad.
4. WHEN se ejecuta el proceso de reducción, THE MotorNube SHALL iterar sobre CADA red de la nube y explorar TODAS sus sub-topologías posibles mediante reducción progresiva.
5. WHEN se evalúa una red en una sub-topología, THE MotorNube SHALL evaluar mediante feedforward sin entrenamiento previo y actualizar la mejor red global si la precisión supera el umbral Y supera la mejor precisión encontrada hasta el momento.
6. WHEN se reduce la topología de una red, THE MotorNube SHALL usar la función reconstruir para crear una nueva red con la topología reducida preservando los pesos existentes de las neuronas no eliminadas.
7. WHEN la PoliticaEliminacion retorna nothing (no hay más reducciones posibles para una red), THE MotorNube SHALL pasar a la siguiente red de la nube.
8. WHEN se han explorado todas las redes y todas sus sub-topologías, Y se ha encontrado una red viable (R* ≠ ∅), THE MotorNube SHALL refinar la mejor red encontrada mediante entrenamiento (backpropagation) durante epocas_refinamiento épocas. El refinamiento ocurre UNA SOLA VEZ al final del proceso.
9. WHEN ninguna red en ninguna sub-topología supera el umbral_acierto durante todo el proceso, THE MotorNube SHALL retornar un InformeNube con exitoso igual a false y mejor_red igual a nothing.
10. THE MotorNube SHALL registrar el tiempo total de ejecución en milisegundos en el InformeNube.
11. THE MotorNube SHALL registrar el total de redes evaluadas (contando cada evaluación en cada sub-topología) y el total de reducciones realizadas en el InformeNube.
12. THE MotorNube SHALL rastrear la mejor red y su precisión de forma GLOBAL a través de todas las redes de la nube y todas sus sub-topologías.


### Requisito 7: Módulo principal y exports

**Historia de usuario:** Como usuario del paquete, quiero importar RandomCloud.jl y tener acceso a todos los tipos y funciones públicas, para poder usar el método sin conocer la estructura interna del paquete.

#### Criterios de aceptación

1. THE RandomCloud SHALL exportar los tipos: ConfiguracionNube, PoliticaEliminacion, PoliticaSecuencial, MotorNube e InformeNube.
2. THE RandomCloud SHALL exportar las funciones: siguiente_reduccion, ejecutar y reconstruir.
3. WHEN un usuario ejecuta `using RandomCloud`, THE RandomCloud SHALL hacer disponibles todos los tipos y funciones exportados sin necesidad de prefijo de módulo.

### Requisito 8: Reproducibilidad y determinismo

**Historia de usuario:** Como investigador, quiero que el método produzca resultados idénticos dada la misma configuración y datos, para que los experimentos sean reproducibles en publicaciones académicas.

#### Criterios de aceptación

1. WHEN se ejecuta el método dos veces con la misma ConfiguracionNube y los mismos datos de entrada, THE MotorNube SHALL producir resultados idénticos (misma topología final, misma precisión).
2. WHEN se cambia únicamente la semilla en la ConfiguracionNube, THE MotorNube SHALL generar nubes de redes diferentes.
3. THE MotorNube SHALL utilizar un generador de números aleatorios local (no el global) para evitar interferencias con otros procesos.

### Requisito 9: Gestión de dependencias

**Historia de usuario:** Como desarrollador Julia, quiero que el paquete dependa únicamente de la biblioteca estándar de Julia, para facilitar la instalación y evitar conflictos de dependencias.

#### Criterios de aceptación

1. THE RandomCloud SHALL depender únicamente de los módulos de la biblioteca estándar de Julia: Random y LinearAlgebra.
2. THE RandomCloud SHALL declarar Test y BenchmarkTools como dependencias de test (extras) en Project.toml.
3. THE RandomCloud SHALL definir las dependencias en un archivo Project.toml válido con los UUIDs correctos de cada paquete.

### Requisito 10: Tests y benchmarks

**Historia de usuario:** Como investigador, quiero una suite de tests que valide cada componente del paquete y benchmarks que midan el rendimiento, para garantizar la corrección y poder reportar métricas en la publicación.

#### Criterios de aceptación

1. THE RandomCloud SHALL incluir tests unitarios para ConfiguracionNube que verifiquen valores por defecto y validaciones de error.
2. THE RandomCloud SHALL incluir tests unitarios para PoliticaSecuencial que verifiquen la reducción progresiva y los casos límite.
3. THE RandomCloud SHALL incluir un test del MotorNube que resuelva el problema XOR con una topología inicial de al menos 2 capas ocultas.
4. THE RandomCloud SHALL incluir un test de integración que ejecute el flujo completo del método desde la configuración hasta el InformeNube.
5. THE RandomCloud SHALL incluir un benchmark de escalabilidad que compare el rendimiento con distintos tamaños de nube.
6. WHEN se ejecutan todos los tests con `Pkg.test()`, THE RandomCloud SHALL pasar todos los tests sin errores.

### Requisito 11: Ejemplo de uso

**Historia de usuario:** Como usuario nuevo del paquete, quiero un ejemplo funcional que demuestre cómo usar el método con el problema XOR, para poder entender rápidamente la API del paquete.

#### Criterios de aceptación

1. THE RandomCloud SHALL incluir un archivo de ejemplo en examples/xor.jl que demuestre el uso completo del método.
2. THE ejemplo SHALL definir el dataset XOR con entradas y objetivos en formato column-major (muestras en columnas).
3. THE ejemplo SHALL crear una ConfiguracionNube con parámetros adecuados para resolver XOR.
4. THE ejemplo SHALL crear un MotorNube, ejecutar el método e imprimir los resultados del InformeNube.
5. WHEN el método encuentra una red viable, THE ejemplo SHALL imprimir la topología final, la precisión y la reducción de parámetros.

### Requisito 12: Documentación del método

**Historia de usuario:** Como investigador que prepara una publicación, quiero un documento formal que describa el método con rigor académico, para usarlo como base del paper y como registro de propiedad intelectual.

#### Criterios de aceptación

1. THE RandomCloud SHALL incluir un documento en docs/metodo.md que describa formalmente el Método de la Nube Aleatoria.
2. THE documento SHALL incluir la definición formal del algoritmo con sus pasos numerados.
3. THE documento SHALL incluir la descripción de los hiperparámetros y su efecto en el comportamiento del método.
4. THE documento SHALL incluir el análisis de complejidad computacional del método.
