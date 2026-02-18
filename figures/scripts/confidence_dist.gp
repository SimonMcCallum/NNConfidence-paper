# confidence_dist.gp — Confidence score distribution: correct vs incorrect
# Source data: data/results/llama8b_norm_shift_signals.json
#
# Generates: docs/figures/output/confidence_dist.tex
#
# Usage:  gnuplot docs/figures/scripts/confidence_dist.gp

set terminal epslatex size 5in,3in color colortext standalone \
    header '\usepackage{amsmath}'
set output 'docs/figures/output/confidence_dist.tex'

set title 'Confidence Distribution: Correct vs Incorrect (Norm-Shift Head)' font ",12"
set xlabel 'Confidence Score' font ",11"
set ylabel 'Count' font ",11"

set xrange [0:1]
set xtics 0.1
set grid ytics lt 0 lw 0.5 lc rgb "#cccccc"

set key top right font ",9" spacing 1.2

set style data histograms
set style histogram clustered gap 1
set style fill solid 0.6 border -1
set boxwidth 0.04

plot 'docs/figures/data/confidence_dist.dat' \
        using 2:xtic(1) with boxes lc rgb "#228B22" title 'Correct', \
     '' using 3 with boxes lc rgb "#B22222" title 'Incorrect'

set output
