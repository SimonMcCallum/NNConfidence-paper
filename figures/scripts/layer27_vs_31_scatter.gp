# layer27_vs_31_scatter.gp — Per-example scatter: Layer 27 vs Layer 31 norm-shift
# Source data: data/results/llama8b_norm_shift_signals.json
# Checkpoint:  data/checkpoints/llama3.1-8b_norm_shift/best_norm_shift_combined.pt
#
# Each point is one MCQ example. Correct answers cluster higher on both axes
# but the vertical (L31) separation is 2.5x wider than horizontal (L27).
#
# Generates: docs/figures/output/layer27_vs_31_scatter.tex  (epslatex terminal)
#
# Usage:  gnuplot docs/figures/scripts/layer27_vs_31_scatter.gp

set terminal epslatex size 5.5in,4.5in color colortext standalone \
    header '\usepackage{amsmath}\usepackage{lmodern}'
set output 'docs/figures/output/layer27_vs_31_scatter.tex'

set title '\textbf{Confidence Head: Layer 27 vs Layer 31 Norm-Shift Separation}' font ",12"
set xlabel 'Norm-shift $s_{27} = 1 - \mathrm{std}(\mathbf{h}^{[27]})$' font ",11"
set ylabel 'Norm-shift $s_{31} = 1 - \mathrm{std}(\mathbf{h}^{[31]})$' font ",11"

set xrange [0.595:0.680]
set yrange [0.31:0.47]
set xtics 0.01 font ",9"
set ytics 0.02 font ",9"
set mxtics 2
set mytics 2
set grid ytics lt 0 lw 0.5 lc rgb "#cccccc"
set grid xtics lt 0 lw 0.5 lc rgb "#cccccc"

set key bottom right font ",9" spacing 1.2 box lw 0.5

# Mean crosshairs for correct (green)
set arrow 1 from 0.6512,0.31 to 0.6512,0.47 nohead lc rgb "#228B22" lw 1.0 dt 3
set arrow 2 from 0.595,0.4145 to 0.680,0.4145 nohead lc rgb "#228B22" lw 1.0 dt 3

# Mean crosshairs for incorrect (red)
set arrow 3 from 0.6322,0.31 to 0.6322,0.47 nohead lc rgb "#B22222" lw 1.0 dt 3
set arrow 4 from 0.595,0.3678 to 0.680,0.3678 nohead lc rgb "#B22222" lw 1.0 dt 3

# Annotations for the gaps
set label 1 at 0.642,0.465 \
    '\footnotesize L27 gap: $\Delta\mu = 0.019$' tc rgb "#555555"
set label 2 at 0.642,0.455 \
    '\footnotesize L31 gap: $\Delta\mu = 0.047$ ($2.5\times$)' tc rgb "#555555"

# Shaded region labels for cluster centers
set label 3 'Correct' at 0.655,0.435 center font ",9" tc rgb "#228B22"
set label 4 'Incorrect' at 0.625,0.355 center font ",9" tc rgb "#B22222"

# Plot
plot 'docs/figures/data/layer27_vs_31_scatter.dat' \
        using ($4==1 ? $2 : 1/0):($4==1 ? $3 : 1/0) with points \
        pt 7 ps 1.2 lc rgb "#228B22" title 'Correct ($n$=60)', \
     '' using ($4==0 ? $2 : 1/0):($4==0 ? $3 : 1/0) with points \
        pt 5 ps 1.2 lc rgb "#B22222" title 'Incorrect ($n$=30)'

set output
