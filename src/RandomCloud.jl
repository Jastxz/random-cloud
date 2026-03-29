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

include("configuracion.jl")
include("activaciones.jl")
include("red_neuronal.jl")
include("politica.jl")
include("evaluacion.jl")
include("motor.jl")
include("informe.jl")

end
