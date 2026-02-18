# example_signals.gp — Three individual MCQ examples at different confidence levels
# Source data: data/results/llama8b_norm_shift_signals.json
# Shows actual per-layer norm-shift profiles for high/medium/low confidence predictions.
#
# Generates: docs/figures/output/example_signals.tex
#
# Usage:  gnuplot docs/figures/scripts/example_signals.gp

set terminal epslatex size 5in,3in color colortext standalone \
    header '\usepackage{amsmath}'
set output 'docs/figures/output/example_signals.tex'

set title 'Norm-Shift Profiles: High vs Medium vs Low Confidence' font ",12"
set xlabel 'Transformer Layer' font ",11"
set ylabel '$s_i = 1 - \mathrm{std}(\mathbf{h}^{[i]})$' font ",11"

set xrange [1:32]
set yrange [-1.8:1.2]
set xtics 4
set grid ytics lt 0 lw 0.5 lc rgb "#cccccc"
set grid xtics lt 0 lw 0.5 lc rgb "#cccccc"

set key top right font ",9" spacing 1.2

plot 'docs/figures/data/example_signals.dat' \
        using 1:2 with linespoints lc rgb "#228B22" lw 2 pt 7 ps 0.5 \
        title 'High confidence ($c = 0.90$)', \
     '' using 1:3 with linespoints lc rgb "#DAA520" lw 2 pt 9 ps 0.5 \
        title 'Medium confidence ($c = 0.23$)', \
     '' using 1:4 with linespoints lc rgb "#B22222" lw 2 pt 5 ps 0.5 \
        title 'Low confidence ($c = 0.01$)', \
     0 with lines lc rgb "#888888" lw 0.5 dt 2 notitle

set output
