# norm_shift_avg.gp — Average norm-shift signal: correct vs incorrect predictions
# Source data: data/results/llama8b_norm_shift_signals.json
# Checkpoint:  data/checkpoints/llama3.1-8b_norm_shift/best_norm_shift_combined.pt
#
# Generates: docs/figures/output/norm_shift_avg.tex  (epslatex terminal)
#
# Usage:  gnuplot docs/figures/scripts/norm_shift_avg.gp

set terminal epslatex size 5in,3in color colortext standalone \
    header '\usepackage{amsmath}'
set output 'docs/figures/output/norm_shift_avg.tex'

set object 1 rect from screen 0,0 to screen 1,1 behind fillcolor rgb "white" fillstyle solid noborder

set title 'Norm-Shift Signal by Layer --- Correct vs Incorrect Predictions' font ",12"
set xlabel 'Transformer Layer' font ",11"
set ylabel '$s_i = 1 - \mathrm{std}(\mathbf{h}^{[i]})$' font ",11"

set xrange [1:32]
set yrange [-1.8:1.2]
set xtics 4
set grid ytics lt 0 lw 0.5 lc rgb "#cccccc"
set grid xtics lt 0 lw 0.5 lc rgb "#cccccc"

set key top right font ",9" spacing 1.2

set style fill transparent solid 0.15

# Plot with error bands
plot 'docs/figures/data/norm_shift_avg.dat' \
        using 1:($2-$3):($2+$3) with filledcurves lc rgb "#228B22" notitle, \
     '' using 1:2 with linespoints lc rgb "#228B22" lw 2 pt 7 ps 0.6 \
        title 'Correct predictions (mean $\pm$ 1$\sigma$)', \
     '' using 1:($4-$5):($4+$5) with filledcurves lc rgb "#B22222" notitle, \
     '' using 1:4 with linespoints lc rgb "#B22222" lw 2 pt 5 ps 0.6 \
        title 'Incorrect predictions (mean $\pm$ 1$\sigma$)', \
     0 with lines lc rgb "#888888" lw 0.5 dt 2 notitle

set output
