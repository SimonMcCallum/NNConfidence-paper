# layer27_vs_31.gp — Subtle differences in norm-shift signals at layer 27 vs 31
# Source data: data/results/llama8b_norm_shift_signals.json
# Checkpoint:  data/checkpoints/llama3.1-8b_norm_shift/best_norm_shift_combined.pt
#
# Shows how the confidence head leverages the widening separation between
# correct and incorrect predictions from layer 27 (gap=0.019) to layer 31 (gap=0.047)
#
# Generates: docs/figures/output/layer27_vs_31.tex  (epslatex terminal)
#
# Usage:  gnuplot docs/figures/scripts/layer27_vs_31.gp

set terminal epslatex size 7in,9in color colortext standalone \
    header '\usepackage{amsmath}\usepackage{lmodern}'
set output 'docs/figures/output/layer27_vs_31.tex'

set multiplot layout 3,1 margins 0.12,0.95,0.06,0.96 spacing 0.0,0.08

# ── Panel 1: Density histograms (top) ────────────────────────────────────────

set title '\textbf{Norm-Shift Distribution: Layer 27 vs Layer 31}' font ",13"
set xlabel '' font ",11"
set ylabel 'Count' font ",11"
set xrange [0.30:0.70]
set yrange [0:20]
set xtics 0.05 font ",9"
set ytics 5 font ",9"
set grid ytics lt 0 lw 0.5 lc rgb "#cccccc"
set grid xtics lt 0 lw 0.5 lc rgb "#cccccc"

set key top left font ",8" spacing 1.1 box lw 0.5

set style fill transparent solid 0.50

# Vertical reference lines for group means
set arrow 1 from 0.6512,0 to 0.6512,18 nohead lc rgb "#228B22" lw 1.5 dt 3
set arrow 2 from 0.6322,0 to 0.6322,18 nohead lc rgb "#B22222" lw 1.5 dt 3
set arrow 3 from 0.4145,0 to 0.4145,18 nohead lc rgb "#1E90FF" lw 1.5 dt 3
set arrow 4 from 0.3678,0 to 0.3678,18 nohead lc rgb "#DC143C" lw 1.5 dt 3

# Labels for the gaps
set label 1 '$\Delta = 0.019$' at 0.640,17 font ",8" tc rgb "#555555"
set label 2 '$\Delta = 0.047$' at 0.385,17 font ",8" tc rgb "#555555"

# Annotating arrows between means
set arrow 5 from 0.6322,16.5 to 0.6512,16.5 heads lc rgb "#555555" lw 1.0
set arrow 6 from 0.3678,16.5 to 0.4145,16.5 heads lc rgb "#555555" lw 1.0

boxw = 0.016

plot 'docs/figures/data/layer27_vs_31_density.dat' \
        using ($1-boxw/4):2 with boxes lc rgb "#228B22" lw 1.0 \
        title 'L27 correct ($\mu$=0.651)', \
     '' using ($1+boxw/4):3 with boxes lc rgb "#B22222" lw 1.0 \
        title 'L27 incorrect ($\mu$=0.632)', \
     '' using ($1-boxw/4):4 with boxes lc rgb "#1E90FF" lw 1.0 \
        title 'L31 correct ($\mu$=0.415)', \
     '' using ($1+boxw/4):5 with boxes lc rgb "#DC143C" lw 1.0 \
        title 'L31 incorrect ($\mu$=0.368)'

unset arrow 1; unset arrow 2; unset arrow 3; unset arrow 4
unset arrow 5; unset arrow 6
unset label 1; unset label 2

# ── Panel 2: Scatter plot L27 vs L31 per example (middle) ───────────────────

set title '\textbf{Per-Example Norm-Shift: Layer 27 vs Layer 31}' font ",13"
set xlabel 'Norm-shift $s_{27}$ (Layer 27)' font ",11"
set ylabel 'Norm-shift $s_{31}$ (Layer 31)' font ",11"
set xrange [0.595:0.680]
set yrange [0.31:0.47]
set xtics 0.01 font ",9"
set ytics 0.02 font ",9"
set grid ytics lt 0 lw 0.5 lc rgb "#cccccc"
set grid xtics lt 0 lw 0.5 lc rgb "#cccccc"

set key top left font ",8" spacing 1.1 box lw 0.5

# Diagonal reference: equal-rate decline
set arrow 7 from 0.595,0.33 to 0.680,0.415 nohead lc rgb "#aaaaaa" lw 0.8 dt 4

# Mean crosshairs
set arrow 8 from 0.6512,0.31 to 0.6512,0.47 nohead lc rgb "#228B22" lw 0.8 dt 3
set arrow 9 from 0.595,0.4145 to 0.680,0.4145 nohead lc rgb "#228B22" lw 0.8 dt 3
set arrow 10 from 0.6322,0.31 to 0.6322,0.47 nohead lc rgb "#B22222" lw 0.8 dt 3
set arrow 11 from 0.595,0.3678 to 0.680,0.3678 nohead lc rgb "#B22222" lw 0.8 dt 3

# Plot correct and incorrect as separate filtered series
plot 'docs/figures/data/layer27_vs_31_scatter.dat' \
        using ($4==1 ? $2 : 1/0):($4==1 ? $3 : 1/0) with points \
        pt 7 ps 1.0 lc rgb "#228B22" title 'Correct ($n$=60)', \
     '' using ($4==0 ? $2 : 1/0):($4==0 ? $3 : 1/0) with points \
        pt 5 ps 1.0 lc rgb "#B22222" title 'Incorrect ($n$=30)'

unset arrow 7; unset arrow 8; unset arrow 9; unset arrow 10; unset arrow 11

# ── Panel 3: Zoomed layer-by-layer progression (bottom) ─────────────────────

set title '\textbf{Separation Widening: Layers 25--32}' font ",13"
set xlabel 'Transformer Layer' font ",11"
set ylabel '$s_i = 1 - \mathrm{std}(\mathbf{h}^{[i]})$' font ",11"
set xrange [24.5:32.5]
set yrange [-1.5:0.80]
set xtics 1 font ",9"
set ytics 0.2 font ",9"
set grid ytics lt 0 lw 0.5 lc rgb "#cccccc"
set grid xtics lt 0 lw 0.5 lc rgb "#cccccc"

set key top right font ",8" spacing 1.1 box lw 0.5

set style fill transparent solid 0.15

# Highlight layers 27 and 31
set object 1 rect from 26.7,-1.5 to 27.3,0.80 fc rgb "#FFFFCC" fillstyle solid 0.20 noborder behind
set object 2 rect from 30.7,-1.5 to 31.3,0.80 fc rgb "#FFFFCC" fillstyle solid 0.20 noborder behind

set label 3 'L27' at 27,0.72 center font ",8" tc rgb "#666666"
set label 4 'L31' at 31,0.72 center font ",8" tc rgb "#666666"

# Show difference annotations
set label 5 '$\Delta$=0.019' at 27.4,0.63 font ",7" tc rgb "#555555"
set label 6 '$\Delta$=0.047' at 31.4,0.37 font ",7" tc rgb "#555555"

plot 'docs/figures/data/norm_shift_avg.dat' \
        using 1:($2-$3):($2+$3) with filledcurves lc rgb "#228B22" notitle, \
     '' using 1:2 with linespoints lc rgb "#228B22" lw 2.5 pt 7 ps 0.8 \
        title 'Correct (mean $\pm$ 1$\sigma$)', \
     '' using 1:($4-$5):($4+$5) with filledcurves lc rgb "#B22222" notitle, \
     '' using 1:4 with linespoints lc rgb "#B22222" lw 2.5 pt 5 ps 0.8 \
        title 'Incorrect (mean $\pm$ 1$\sigma$)'

unset object 1; unset object 2
unset label 3; unset label 4; unset label 5; unset label 6

unset multiplot
set output
