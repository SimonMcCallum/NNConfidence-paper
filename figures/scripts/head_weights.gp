# head_weights.gp — Trained confidence head: per-layer weight importance
# Source: data/checkpoints/llama3.1-8b_norm_shift/best_norm_shift_combined.pt
# Shows sum of absolute weights in norm_shift_proj.0.weight per input layer.
# This reveals which transformer layers the head pays most attention to.
#
# Generates: docs/figures/output/head_weights.tex
#
# Usage:  gnuplot docs/figures/scripts/head_weights.gp

set terminal epslatex size 5in,3in color colortext standalone \
    header '\usepackage{amsmath}'
set output 'docs/figures/output/head_weights.tex'

set title 'Trained Head: Layer Attention Weights ($\sum |w_{j,i}|$ per layer $i$)' font ",12"
set xlabel 'Transformer Layer' font ",11"
set ylabel 'Weight Importance $\sum_j |w_{j,i}|$' font ",11"

set xrange [0:33]
set yrange [0:8]
set xtics 4
set grid ytics lt 0 lw 0.5 lc rgb "#cccccc"

set key off

set style fill solid 0.7 border -1
set boxwidth 0.8

# Color bars by layer group
plot 'docs/figures/data/head_weights.dat' using 1:2:(($1<=8)? 0x555555 : \
     ($1<=16)? 0xDAA520 : ($1<=24)? 0xB22222 : 0x4a4a8a) \
     with boxes lc rgbcolor variable notitle

set output
