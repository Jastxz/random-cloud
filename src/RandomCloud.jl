module RandomCloud

using Random
using LinearAlgebra

export ConfiguracionNube
export PoliticaEliminacion, PoliticaSecuencial, siguiente_reduccion
export MotorNube, ejecutar
export InformeNube
export reconstruir
export EntrenarBuffers
export evaluar, evaluar_regresion, evaluar_f1, evaluar_auc
export activaciones_por_capa
export aplicar_activacion_batch, aplicar_derivada_batch
export feedforward_batch
export empaquetar_pesos, reempaquetar_pesos
export evaluar_nube_batch
export entrenar_batch_matmul!
export GPU_AVAILABLE
export a_gpu, de_gpu, estimar_vram, verificar_gpu

const GPU_AVAILABLE = Ref(false)

include("configuracion.jl")
include("activaciones.jl")
include("red_neuronal.jl")
include("lotes.jl")
include("politica.jl")
include("evaluacion.jl")
include("motor.jl")
include("informe.jl")

# GPU extension stubs — overridden by ext/RandomCloudCUDAExt when CUDA.jl is loaded
function a_gpu end
function de_gpu end
function estimar_vram end
function verificar_gpu end

end
