module RandomCloudCUDAExt

using CUDA
using RandomCloud
using RandomCloud: RedNeuronal, MotorNube, ConfiguracionNube, InformeNube,
                   feedforward_batch, aplicar_activacion_batch, aplicar_derivada_batch,
                   activaciones_por_capa, evaluar_nube_batch, entrenar_batch_matmul!,
                   empaquetar_pesos, reempaquetar_pesos, _explorar_red, _ejecutar_gpu,
                   ResultadoExploracion, PoliticaSecuencial, siguiente_reduccion, reconstruir,
                   EntrenarBuffers, entrenar!, evaluar

using Random: MersenneTwister, shuffle!

# Enable GPU support
RandomCloud.GPU_AVAILABLE[] = true

include("gpu_backend.jl")

end
