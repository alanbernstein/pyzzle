https://n-e-r-v-o-u-s.com/projects/albums/laser-cut-puzzles/

https://mathematica.stackexchange.com/questions/6706/how-can-i-calculate-a-jigsaw-puzzle-cut-path

nubs (or 'locks') are defined as splines with only 3 control points, making an inverted 1-1-rt(2) triangle, rather than a square like i did in pyzzle.

# pyzzle

## Terminology
- puzzle - the whole thing
- puzzle cut - one continuous path, composed of one or more 'puzzle piece edge'
- base cut - the smooth underlying path followed by a cut
- puzzle piece edge - a single part of a cut, with one puzzle piece tab
- puzzle piece tab - an 'outie' or 'nub'
- edge segment - one part of an edge, defined between spline knots
- puzzle piece - a simple connected region, bounded by puzzle piece edges (not explicitly defined in base class here)

