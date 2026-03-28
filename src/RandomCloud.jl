module RandomCloud

using Random

export ConfiguracionNube
export PoliticaEliminacion, PoliticaSecuencial, siguiente_reduccion
export MotorNube, ejecutar
export InformeNube
export reconstruir

include("configuracion.jl")
include("red_neuronal.jl")
include("politica.jl")
include("evaluacion.jl")
include("motor.jl")
include("informe.jl")

end
