# ConfiguracionNube — Hiperparámetros del Método de la Nube Aleatoria

const ACTIVACIONES_VALIDAS = (:sigmoid, :relu, :identidad)

struct ConfiguracionNube
    tamano_nube::Int
    topologia_inicial::Vector{Int}
    umbral_acierto::Float64
    neuronas_eliminar::Int
    epocas_refinamiento::Int
    tasa_aprendizaje::Float64
    semilla::Int
    activacion::Symbol
    batch_size::Int
    gpu::Bool

    function ConfiguracionNube(;
        tamano_nube::Int = 10,
        topologia_inicial::Vector{Int} = [2, 4, 1],
        umbral_acierto::Float64 = 0.5,
        neuronas_eliminar::Int = 1,
        epocas_refinamiento::Int = 1000,
        tasa_aprendizaje::Float64 = 0.1,
        semilla::Int = 42,
        activacion::Symbol = :sigmoid,
        batch_size::Int = 0,
        gpu::Bool = false
    )
        tamano_nube < 1 && throw(ArgumentError(
            "tamano_nube debe ser ≥ 1, se recibió: $tamano_nube"))

        length(topologia_inicial) < 3 && throw(ArgumentError(
            "topologia_inicial requiere al menos 3 capas (entrada, oculta(s), salida)"))

        (umbral_acierto < 0.0 || umbral_acierto > 1.0) && throw(ArgumentError(
            "umbral_acierto debe estar en [0.0, 1.0], se recibió: $umbral_acierto"))

        neuronas_eliminar < 1 && throw(ArgumentError(
            "neuronas_eliminar debe ser ≥ 1, se recibió: $neuronas_eliminar"))

        for i in 2:(length(topologia_inicial) - 1)
            topologia_inicial[i] < 1 && throw(ArgumentError(
                "las capas ocultas requieren al menos 1 neurona"))
        end

        activacion in ACTIVACIONES_VALIDAS || throw(ArgumentError(
            "activacion debe ser una de $ACTIVACIONES_VALIDAS, se recibió: :$activacion"))

        batch_size < 0 && throw(ArgumentError(
            "batch_size debe ser ≥ 0 (0 = sample-by-sample), se recibió: $batch_size"))

        gpu && !GPU_AVAILABLE[] && throw(ArgumentError(
            "CUDA.jl must be added to use gpu=true. Run: ] add CUDA"))

        new(tamano_nube, copy(topologia_inicial), umbral_acierto,
            neuronas_eliminar, epocas_refinamiento, tasa_aprendizaje, semilla,
            activacion, batch_size, gpu)
    end
end
