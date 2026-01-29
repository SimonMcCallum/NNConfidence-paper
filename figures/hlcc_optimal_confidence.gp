# HLCC Expected Return Heatmap with Optimal Confidence Curve
# E[S(c,p)] = p(1+c) + (1-p)(-2c^2)
# Optimal: c* = p / (4(1-p)), capped at [0,1]

set terminal cairolatex pdf color colortext size 5in,4in
set output 'hlcc_optimal_confidence.tex'

set xrange [0:1]
set yrange [0:1]

set isosamples 200,200
set samples 200,200

# Expected score function
E(c,p) = p * (1.0 + c) + (1.0 - p) * (-2.0 * c**2)

# Optimal confidence (capped at 1.0)
cstar(p) = (p >= 0.8) ? 1.0 : p / (4.0 * (1.0 - p))

set palette defined ( \
    -2.0 '#67001f', \
    -1.5 '#b2182b', \
    -1.0 '#d6604d', \
    -0.5 '#f4a582', \
     0.0 '#fddbc7', \
     0.3 '#f7f7f7', \
     0.5 '#d1e5f0', \
     1.0 '#92c5de', \
     1.5 '#4393c3', \
     2.0 '#2166ac' )

set cbrange [-2:2]

# ---- Multiplot: heatmap layer + overlay layer ----
set multiplot

# Layer 1: pm3d heatmap
set pm3d map interpolate 2,2
set xlabel '$c$ (Stated Confidence)'
set ylabel '$p$ (True Accuracy)'
set cblabel '$\mathbb{E}[S]$'

splot E(x,y) with pm3d notitle

# Layer 2: optimal confidence curve overlaid
unset pm3d
unset surface
unset xlabel
unset ylabel
unset cblabel
unset cbtics
unset colorbox
set xrange [0:1]
set yrange [0:1]
set format x ''
set format y ''

set key bottom right box opaque spacing 1.3

set samples 500
set parametric
set trange [0.02:0.99]

# Optimal confidence curve
plot cstar(t), t lc rgb '#000000' lw 2.5 dt 2 title '$c^{*}\!=\!\frac{p}{4(1\!-\!p)}$'

unset multiplot
