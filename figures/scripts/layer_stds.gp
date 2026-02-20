# layer_stds.gp — Hidden state standard deviation by layer
# Source data: data/results/llama8b_norm_shift_signals.json
# This shows std(h[i], dim=-1) — the raw activation spread that RMSNorm must correct.
#
# Generates: docs/figures/output/layer_stds.tex
#
# Usage:  gnuplot docs/figures/scripts/layer_stds.gp

set terminal epslatex size 5in,3in color colortext standalone \
    header '\usepackage{amsmath}'
set output 'docs/figures/output/layer_stds.tex'

set object 1 rect from screen 0,0 to screen 1,1 behind fillcolor rgb "white" fillstyle solid noborder

set title 'Hidden State $\mathrm{std}(\mathbf{h}^{[i]})$ by Layer --- Llama-3.1-8B' font ",12"
set xlabel 'Transformer Layer' font ",11"
set ylabel '$\mathrm{std}(\mathbf{h}^{[i]}, \mathrm{dim}=-1)$' font ",11"

set xrange [1:32]
set yrange [0:2.8]
set xtics 4
set grid ytics lt 0 lw 0.5 lc rgb "#cccccc"
set grid xtics lt 0 lw 0.5 lc rgb "#cccccc"

set key top left font ",9" spacing 1.2

plot 'docs/figures/data/layer_stds.dat' \
        using 1:2 with linespoints lc rgb "#228B22" lw 2 pt 7 ps 0.6 \
        title 'Correct predictions', \
     '' using 1:3 with linespoints lc rgb "#B22222" lw 2 pt 5 ps 0.6 \
        title 'Incorrect predictions'

set output
