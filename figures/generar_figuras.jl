# =============================================================================
# Generación de figuras para el workshop paper
# =============================================================================
#
# Ejecutar con:
#   julia --project=. figures/generar_figuras.jl
#
# Genera 3 figuras en PDF (calidad publicación):
#   1. fig1_method_barplot.pdf — Barras agrupadas: Accuracy por método × dataset
#   2. fig2_acc_vs_compression.pdf — Scatter: Accuracy vs Reducción de parámetros
#   3. fig3_time_comparison.pdf — Barras: Tiempo por método × dataset
# =============================================================================

using CairoMakie

CairoMakie.activate!(type="pdf")
set_theme!(theme_minimal(), fontsize=11)

outdir = joinpath(@__DIR__)

# ─── Datos de la comparativa de baselines (single-seed, semilla=42) ───

datasets = ["Breast\nCancer", "Sonar", "Iono-\nsphere", "Adult\nIncome", "Iris", "Wine", "Opt.\nDigits"]

# Accuracy test por método (Clásico, Magnitude, Random, Nube)
acc_clasico   = [97.3, 78.0, 94.3, 84.2, 100.0, 94.4, 96.3]
acc_magnitude = [97.3, 78.0, 87.1, 84.4, 100.0, 94.4, 95.0]
acc_random    = [97.3, 69.8, 88.0, 84.4, 100.0, 94.4, 95.4]
acc_nube      = [97.3, 80.5, 90.0, 85.0, 100.0, 94.4, 95.9]

# Reducción de parámetros (%)
red_nube = [74.4, 87.2, 81.0, 49.9, 41.2, 55.6, 62.2]

# Tiempos (ms) — mediana de 3 runs
t_clasico   = [234, 234, 421, 13100, 0, 0, 7900]  # Iris/Wine no medidos, usamos 0
t_magnitude = [351, 351, 676, 21700, 0, 0, 11500]
t_random    = [349, 349, 660, 21700, 0, 0, 11500]
t_nube      = [158, 158, 341, 15400, 0, 0, 6300]

# Datasets con tiempos medidos
ds_tiempo = ["Sonar", "Ionosphere", "Breast\nCancer", "Opt. Digits", "Adult\nIncome"]
t_c = [234, 421, 241, 7900, 13100]
t_m = [351, 676, 398, 11500, 21700]
t_r = [349, 660, 425, 11500, 21700]
t_n = [158, 341, 227, 6300, 15400]

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURA 1: Barras agrupadas — Accuracy por método × dataset
# ═══════════════════════════════════════════════════════════════════════════════

println("Generando Fig 1: Barras de accuracy...")

fig1 = Figure(size=(700, 350))
ax1 = Axis(fig1[1, 1],
    ylabel="Test Accuracy (%)",
    xticks=(1:7, datasets),
    yticks=60:5:100,
    limits=(nothing, (60, 102)),
    xticklabelrotation=0,
    title="Accuracy by Method and Dataset"
)

n = length(datasets)
w = 0.18  # ancho de cada barra
offsets = [-1.5w, -0.5w, 0.5w, 1.5w]

colors = [:gray70, :steelblue3, :lightsalmon, :seagreen]
labels = ["Full Training", "Magnitude Pruning", "Random Pruning", "Random Cloud"]

for (i, (acc, col, lab)) in enumerate(zip(
    [acc_clasico, acc_magnitude, acc_random, acc_nube],
    colors, labels))
    barplot!(ax1, (1:n) .+ offsets[i], acc, width=w, color=col, label=lab)
end

Legend(fig1[2, 1], ax1, orientation=:horizontal, framevisible=false, nbanks=1, labelsize=9)

save(joinpath(outdir, "fig1_method_barplot.pdf"), fig1)
println("  → fig1_method_barplot.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURA 2: Scatter — Accuracy vs Reducción de parámetros
# ═══════════════════════════════════════════════════════════════════════════════

println("Generando Fig 2: Accuracy vs Compresión...")

fig2 = Figure(size=(500, 400))
ax2 = Axis(fig2[1, 1],
    xlabel="Parameter Reduction (%)",
    ylabel="Test Accuracy (%)",
    title="Accuracy vs Compression (Random Cloud)",
    limits=((30, 95), (75, 102))
)

# Colores por tipo de dataset
ds_labels_short = ["BC", "Sonar", "Iono", "Adult", "Iris", "Wine", "Digits"]
ds_colors = [:crimson, :darkorange, :royalblue, :purple, :forestgreen, :goldenrod, :teal]

# Línea horizontal: accuracy del clásico (referencia)
for i in 1:n
    # Punto de la nube
    scatter!(ax2, [red_nube[i]], [acc_nube[i]], color=ds_colors[i], markersize=14, marker=:circle)
    # Línea horizontal punteada: accuracy del clásico
    hlines!(ax2, [acc_clasico[i]], color=ds_colors[i], linestyle=:dash, linewidth=0.5)
    # Label
    text!(ax2, red_nube[i] + 1.5, acc_nube[i] + 0.3, text=ds_labels_short[i], fontsize=9, color=ds_colors[i])
end

# Leyenda manual
elem_cloud = MarkerElement(color=:black, marker=:circle, markersize=10)
elem_classic = LineElement(color=:gray50, linestyle=:dash, linewidth=1)
Legend(fig2[2, 1],
    [elem_cloud, elem_classic],
    ["Random Cloud accuracy", "Full training accuracy (reference)"],
    orientation=:horizontal, framevisible=false, labelsize=9)

save(joinpath(outdir, "fig2_acc_vs_compression.pdf"), fig2)
println("  → fig2_acc_vs_compression.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURA 3: Barras — Tiempo por método (ratio vs clásico)
# ═══════════════════════════════════════════════════════════════════════════════

println("Generando Fig 3: Coste computacional...")

fig3 = Figure(size=(600, 350))
ax3 = Axis(fig3[1, 1],
    ylabel="Time Ratio vs Full Training",
    xticks=(1:5, ds_tiempo),
    title="Computational Cost (ratio to full training baseline)",
    limits=(nothing, (0, 2.2))
)

# Ratios vs clásico
r_m = t_m ./ t_c
r_r = t_r ./ t_c
r_n = t_n ./ t_c

w3 = 0.22
offsets3 = [-w3, 0, w3]

barplot!(ax3, (1:5) .- w3, r_m, width=w3, color=:steelblue3, label="Magnitude Pruning")
barplot!(ax3, (1:5), r_r, width=w3, color=:lightsalmon, label="Random Pruning")
barplot!(ax3, (1:5) .+ w3, r_n, width=w3, color=:seagreen, label="Random Cloud")

# Línea de referencia: ratio = 1.0 (clásico)
hlines!(ax3, [1.0], color=:gray40, linestyle=:dash, linewidth=1.5, label="Full Training (1.0×)")

Legend(fig3[2, 1], ax3, orientation=:horizontal, framevisible=false, nbanks=1, labelsize=9)

save(joinpath(outdir, "fig3_time_comparison.pdf"), fig3)
println("  → fig3_time_comparison.pdf")

println()
println("Todas las figuras generadas en $(outdir)/")
