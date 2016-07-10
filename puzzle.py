#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import random

from pyzzle.edge import SEGMENTS_PER_PIECE
from pyzzle.cut import PuzzleCutter

from panda.plot_utils import terminal_plot, qplot
from panda.debug import debug, pp, pm


# main TODO:
# - quadratic baseline
# - spline baseline
# - conform to baseline better
# - arbitrary pieces
# - multi-cell pieces
# x fix missing endpoint
# x finish big refactor

def main():
    test_puzzle()


@pm
def test_puzzle():
    size = 1
    cols, rows = 4, 5
    #puzz = Puzzle(puzzle_dim=(cols, rows), piece_dim=(size, size))
    puzz = SquarePuzzle(puzzle_dim=(cols, rows), piece_dim=(size, size), baseline_type='curved')

    pp(puzz.__dict__)

    plt.ion()
    fig = plt.figure()
    fig.add_subplot(111, aspect='equal')
    puzz.plot(color='k')
    plt.axis([-.1, size * cols + .1, -.1, size * rows + .1])
    plt.show(block=False)
    debug()


class Puzzle(object):
    """primary class for simple public interface:
    just instantiate, plot, write svg
    might be clearer to separate into base class and example implementation class,
    although other examples should make it clear enough"""
    svg_filename = 'puzzle_cuts.svg'
    svg_scale = 100  # svg default unit is .01" (in corel at least)

    def __init__(self, piece_dim=None, puzzle_dim=None, cut_type=None,
                 tab_pattern=None, tab_parameters=None):
        """handle inputs, run top-level generate method"""

        self.piece_dim = piece_dim or [1, 1]    # measurement units
        self.puzzle_dim = puzzle_dim or [4, 6]  # puzzle pieces
        self.cut_type = cut_type or 'straight'
        self.tab_pattern = tab_pattern or 'random'
        self.tab_parameters = tab_parameters

        self.generate_cuts()

    def generate_cuts(self):
        """subclass entry point. must define three things:
        - cuts - any internal, simply defined cuts, that have no tabs
        - cut baselines - internal puzzle piece edge cuts, to have tabs added
        - perimeter - outer edge of puzzle"""
        unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        self.perimeter = unit_square * self.piece_dim * self.puzzle_dim

        self.cuts = []
        self.generate_simple_cuts()
        self.generate_cut_baselines()
        self._add_tabs_to_baselines()

    def generate_simple_cuts(self):
        pass

    def _add_tabs_to_baselines(self):
        for n, base in enumerate(self.baselines):
            self.cuts.append(PuzzleCutter(**base).generate())

    def generate_cut_baselines(self):
        """core method - this defines the shapes of the pieces. must replace in child.
        this function should use any means to define:
        - self.baselines - dict containing kwarg inputs for PuzzleCutter
        dict contains:
        - path - the baseline path of the cut
        - num_tabs - the number of tabs to add"""
        self.baselines = []

        W, H = self.puzzle_dim
        for n in range(1, H):
            # horizontal cuts run left-right, stack up-down
            hbase = {'path': np.array([[0, n], [W, n]]) * self.piece_dim[0],
                     'num_tabs': W}

            self.baselines.append(hbase)

        for n in range(1, W):
            # horizontal cuts run left-right, stack up-down
            vbase = {'path': np.array([[n, 0], [n, H]]) * self.piece_dim[1],
                     'num_tabs': H}

            self.baselines.append(vbase)

    def plot(self, **kwargs):
        # caller is responsible for plt.show() etc
        ph = []
        for cut, base in zip(self.cuts, self.baselines):
            ph.append(plt.plot(cut[:, 0], cut[:, 1], **kwargs))
            ph.append(plt.plot(base['path'][:, 0], base['path'][:, 1], 'r--'))

        # add perimeter last
        ph.append(plt.plot(self.perimeter[:, 0], self.perimeter[:, 1], **kwargs))
        return ph

    def write_svg(self, fname=None):
        import svgwrite
        fname = fname or self.svg_filename
        dwg = svgwrite.Drawing(fname, profile='tiny')
        for cut in self.cuts:
            polyline = svgwrite.shapes.Polyline(points=cut * self.svg_scale,
                                                stroke='black', fill='white')
            dwg.add(polyline)
        polyline = svgwrite.shapes.Polyline(points=self.perimeter * self.svg_scale,
                                            stroke='black')
        polyline.fill('white', opacity=0)
        dwg.add(polyline)
        dwg.save()

    def __repr__(self):
        return '%d x %d %s' % (self.puzzle_dim[0],
                               self.puzzle_dim[1],
                               self.__class__.__name__)


class SquarePuzzle(Puzzle):
    def __init__(self, baseline_type=None, **kwargs):
        if baseline_type == 'curved':
            baseline_type = 'quadratic'
        self.baseline_type = baseline_type or 'quadratic'
        super(SquarePuzzle, self).__init__(**kwargs)

    def generate_cut_baselines(self):
        # straight baselines don't care about tab pattern,
        # curved baselines do...
        self.baselines = []

        W, H = self.puzzle_dim
        for n in range(1, H):
            if self.baseline_type == 'straight':
                hbase = {'path': np.array([[0, n], [W, n]]) * self.piece_dim[0],
                         'num_tabs': W}
            elif self.baseline_type == 'quadratic':
                base = self.quadratic_baseline()
                hbase = {'path': base ,
                         'num_tabs': W}
            elif self.baseline_type == 'spline':
                pass

            self.baselines.append(hbase)

        for n in range(1, W):
            if self.baseline_type == 'straight':
                vbase = {'path': np.array([[n, 0], [n, H]]) * self.piece_dim[1],
                         'num_tabs': H}
            elif self.baseline_type == 'quadratic':
                vbase = {'path': [],
                         'num_tabs': W}
            elif self.baseline_type == 'spline':
                pass

            self.baselines.append(vbase)

    def spline_baseline(self):
        # TODO: implement
        # TODO: might be hard to pass tab_positions for this, with different
        # lengths for each tab... try using index, rather than distance along curve
        # TODO: base curve should flatten out rather than have sharp points
        """1. choose tab orientations
        2. for each tab, define the spline knots for its edge:
           left-tab right-tab -> knot-slope
           [-1 1] -> -1
           [1 1]  ->  0
           [1 -1] ->  1
           [-1 -1] -> 0
           knot-slope ~ (left-right) ~ -diff(tabs)
        3. compile all knots into one spline
        """
        tab_signs = self.num_pieces

        pass

    def quadratic_baseline(self, pts_per_piece):
        """define a baseline for a curved cut, on which the edge cut can be applied.
        uses a simple piecewise quadratic function"""

        pts_total = self.num_pieces * pts_per_piece
        t = np.linspace(0, self.num_pieces, pts_total + 1)

        y_seg = quadratic_base(pts_per_piece)

        # concat y_seg's
        y = []
        for n in range(self.num_pieces - 1):
            y.append(random.choice([-1, 1]) * y_seg[0:-1])
        # remove last point from all but last piece
        y.append(random.choice([-1, 1]) * y_seg)
        y = np.hstack(y)

        base_curve = np.vstack((t, y)).T
        return base_curve


def quadratic_base(pts=25):
    t = np.linspace(0, 1, pts + 1)
    y = 0.25 * t * (1 - t)
    base = np.hstack((t, y)).T
    return base


class IrregularSquarePuzzle(Puzzle):
    """every piece is composed of several normal grid cells"""

    def __init__(self):
        pass

    def generate(self):
        self.basecuts = []
        self.cuts = []


# TODO: these
# class SquarePuzzleWithArbitraryPieces(Puzzle)  # fully arbitrary shapes (heart etc)
# class VoronoiPuzzle(Puzzle)
# class SquarePuzzleWithLargeCenter(Puzzle)      # plain grid except center piece
# class TriangleTiledPuzzle(Puzzle):
# class HexagonTiledPuzzle(Puzzle):
# idea: big central piece that looks like a giant puzzle piece
# idea: fractal puzzle - use giant jigsaw piece, add tabs to outside of that

class SpiralPuzzle(Puzzle):

    def __init__(self,
                 spiral_spacing=1,
                 spiral_turns=4,
                 radial_spacing=1,
                 edge_orientation=None,
                 tab_parameters=None):
        self.spiral_spacing = spiral_spacing
        self.spiral_turns = spiral_turns
        self.radial_spacing = radial_spacing
        self.edge_orientation = edge_orientation or 'random'
        self.tab_parameters = tab_parameters

    def generate(self):
        # TODO: spiral cut
        # TODO: radial cuts
        unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        self.perimeter = []  # TODO: figure this out

    def spiral_baseline(t):
        r = 2 * t
        th = t * 2 * np.pi
        x = r * np.cos(th)
        y = r * np.sin(th)
        return np.vstack((x, y)).T





if __name__ == '__main__':
    main()
