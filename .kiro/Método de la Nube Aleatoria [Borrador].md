title:Nuevo método de red neuronal
subtitle:El método de la Nube Aleatoria
title1:Introducción del método
A continuación paso a definir, lo más formalmente que sé, una nueva forma de diseño y búsqueda de redes neuronales.
title1:Definición del método
Se definirá la metodología en dirección global a local, por lo que primero se distinguirán sus partes.
title2:Partes del método
El método consta de tres partes o pasos:
    1. Primero es necesario un conjunto de al menos n redes neuronales (con n perteneciente a los Naturales) de un tamaño predefinido y con sus pesos o neuronas inicializadas aleatoriamente. A esto se le llama Nube Aleatoria.
    2. Elegimos estocástica o arbitrariamente un umbral o porcentaje de acierto mínimo.
    3. Luego se comienza el Proceso de reducción.
        1. Se recorre cada red neuronal perteneciente a nuestra nube.
        2. En cada nube se realizan predicciones con los valores actuales de sus neuronas y sus correspondientes tamaños. Si cualquier nube supera el umbral se guarda su configuración actual como la mejor de su recorrido en caso de hacerlo en paralelo, se guardará simplemente como la mejor al recorrer las redes secuencialmente.
        3. En cualquier caso, eliminamos x neuronas (con x perteneciente a los Naturales y mayor que 0) de alguna de sus capas ocultas.
            1. La política de eliminación queda a discreción del autor.
            2. En este punto es cuando el método comienza a consumir recursos, ya que en su forma básica hay que recorrer todas las posibles estructuras de la red neuronal al ir eliminando neuronas. Una posible mejora inicial es usar algoritmos de búsqueda de adversario como Minimax.
        4. Se reducen los conjuntos de neuronas y se va guardando siempre la mejor configuración hasta haber revisado todas las posibilidades de cada red.
        5. Si ninguna de las redes de la Nube Aleatoria ha pasado el umbra mínimo, reiniciar el proceso o ajustar hiperparámetros de inicialización.
    4. Por último se realiza el Refinamiento de la red.
        1. Una vez que se ha encontrado una red que satisfaga el mínimo del umbral, se entrena como una red neuronal clásica hasta obtener el resultado esperado.
