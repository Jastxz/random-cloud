module RandomCloud

using Random
using LinearAlgebra

export ConfiguracionNube
export PoliticaEliminacion, PoliticaSecuencial, siguiente_reduccion
export MotorNube, ejecutar
export InformeNube
export reconstruir
export EntrenarBuffers

include("configuracion.jl")
include("red_neuronal.jl")
include("politica.jl")
include("evaluacion.jl")
include("motor.jl")
include("informe.jl")

end
