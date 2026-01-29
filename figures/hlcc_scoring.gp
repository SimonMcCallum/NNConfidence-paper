# HLCC Scoring Function: Reward R(c) = 1+c and Penalty P(c) = -2c^2
# Generates hlcc_scoring.tex (fragment) + hlcc_scoring-inc.pdf (graphics)
# Include in paper with: \input{figures/hlcc_scoring}

set terminal cairolatex pdf color colortext size 4.5in,3.2in
set output 'hlcc_scoring.tex'

set xlabel '$c$ (Stated Confidence)'
set ylabel 'Score $S(c)$'

set xrange [0:1]
set yrange [-2.3:2.3]

set grid lc rgb '#cccccc'
set key top left box opaque spacing 1.4

set style line 1 lc rgb '#2166ac' lw 3.0 dt 1
set style line 2 lc rgb '#b2182b' lw 3.0 dt 1
set style line 3 lc rgb '#aaaaaa' lw 0.8 dt 3

# Zero reference line
set arrow from 0,0 to 1,0 nohead ls 3

# Shading: danger zone for overconfident wrong answers
set object 1 rect from 0.5,-2.3 to 1.0,0 fc rgb '#fff0f0' fs solid 0.3 noborder behind

# Endpoint annotations
set label 1 '$S(0)=1$'  at 0.03,1.15  tc rgb '#2166ac'
set label 2 '$S(1)=2$'  at 0.84,2.15  tc rgb '#2166ac'
set label 3 '$S(0)=0$'  at 0.03,-0.20 tc rgb '#b2182b'
set label 4 '$S(1)=-2$' at 0.78,-1.82 tc rgb '#b2182b'

R(c) = 1.0 + c
P(c) = -2.0 * c**2

plot R(x) ls 1 title '$R(c) = 1 + c$ \quad (Correct)', \
     P(x) ls 2 title '$P(c) = -2c^{2}$ \quad (Incorrect)'
