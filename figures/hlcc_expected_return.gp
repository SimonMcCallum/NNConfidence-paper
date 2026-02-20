# HLCC Expected Return Surface
# E[S(c,p)] = p(1+c) + (1-p)(-2c^2)
# 3D surface: expected score as function of true accuracy p and chosen confidence c

set terminal cairolatex pdf color colortext size 5.5in,4.5in
set output 'hlcc_expected_return.tex'

set object 1 rect from screen 0,0 to screen 1,1 behind fillcolor rgb "white" fillstyle solid noborder

set xlabel '$c$ (Confidence)' offset 0,-1
set ylabel '$p$ (Accuracy)' offset 0,-1
set zlabel '$\mathbb{E}[S]$' rotate by 90

set xrange [0:1]
set yrange [0:1]
set zrange [-2:2.1]

set xyplane at -2

set isosamples 50,50
set samples 50,50

set hidden3d front
set view 55, 315, 1, 1

set pm3d depthorder border lc rgb '#555555' lw 0.2
set palette defined ( \
    -2.0 '#67001f', \
    -1.5 '#b2182b', \
    -1.0 '#d6604d', \
    -0.5 '#f4a582', \
     0.0 '#f7f7f7', \
     0.5 '#92c5de', \
     1.0 '#4393c3', \
     1.5 '#2166ac', \
     2.0 '#053061' )

set cbrange [-2:2]
set cblabel '$\mathbb{E}[S(c,p)]$'

# The expected score function
E(c,p) = p * (1.0 + c) + (1.0 - p) * (-2.0 * c**2)

splot E(x,y) with pm3d notitle
