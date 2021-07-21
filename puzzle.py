#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import random

from geometry.spline import slope_controlled_bezier_curve

from edge import SEGMENTS_PER_PIECE
from cut import PuzzleCutter, get_tab_sign

# from panda.plot_utils import terminal_plot, qplot
from ipdb import iex, set_trace as debug


# as of 2018/12/20, last edit was 2016/07/28

# main TODO:

# x spline baseline - needs smarter normal handling (or dumber)
# - tab direction should match edge curve
# - conform to baseline better
# - arbitrary pieces
# - multi-cell pieces
# x quadratic baseline
# x fix conformal map problem
# x big center piece
# x fix missing endpoint
# x finish big refactor

def main():
    test_puzzle()


@iex
def test_puzzle():
    size = 2
    cols, rows = 8, 6
    random.seed(0)  # cheap way to make this repeatable
    # puzz = Puzzle(puzzle_dim=(cols, rows), piece_dim=(size, size))
    # puzz = SquarePuzzle(puzzle_dim=(cols, rows), piece_dim=(size, size), baseline_type='straight')
    # puzz = SquarePuzzle(puzzle_dim=(cols, rows), piece_dim=(size, size), baseline_type='curved')
    # puzz = VoronoiPuzzle()
    # puzz = LargeCenterSquarePuzzle(puzzle_dim=(cols, rows), center_dim=(3, 3))

    puzz = LargeCenterSquarePuzzle(puzzle_dim=(cols, rows), center_dim=(2, 2))

    # puzz = HeartRingPuzzle()

    # puzz.write_svg(fname='heart-center-puzzle.svg', scale=size)

    # plt.ion()edge
    fig = plt.figure()
    fig.add_subplot(111, aspect='equal')
    puzz.plot(color='k', scale=size)
    plt.axis([-.1, size * cols + .1, -.1, size * rows + .1])
    plt.show()


class Puzzle(object):
    """primary shapes class for simple public interface:
    just instantiate, plot, write svg
    might be clearer to separate into base class and example implementation class,
    although other examples should make it clear enough"""
    svg_filename = 'puzzle_cuts.svg'
    svg_scale = 100  # svg default unit is .01" (in corel at least)

    def __init__(self, piece_dim=None, puzzle_dim=None, cut_type=None,
                 tab_pattern=None, tab_parameters=None):
        """handle inputs, run top-level generate method"""

        self.piece_dim = piece_dim or [1, 1]    # measurement units
        self.puzzle_dim = puzzle_dim or [6, 4]  # puzzle pieces
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
            # vertical cuts run up-down, stack left-right
            vbase = {'path': np.array([[n, 0], [n, H]]) * self.piece_dim[1],
                     'num_tabs': H}

            self.baselines.append(vbase)

    def plot(self, **kwargs):
        # caller is responsible for plt.show() etc
        scale = kwargs.pop('scale', 1)

        ph = []
        for cut in self.cuts:
            ph.append(plt.plot(scale * cut[:, 0], scale * cut[:, 1], **kwargs))

        for base in self.baselines:
            ph.append(plt.plot(scale * base['path'][:, 0], scale * base['path'][:, 1], 'r--'))

        # add perimeter last
        ph.append(plt.plot(scale * self.perimeter[:, 0], scale * self.perimeter[:, 1], **kwargs))
        return ph

    def write_svg(self, fname=None, scale=1):
        import svgwrite
        fname = fname or self.svg_filename
        dwg = svgwrite.Drawing(fname, profile='tiny')
        for cut in self.cuts:
            polyline = svgwrite.shapes.Polyline(points=cut * scale * self.svg_scale,
                                                stroke='black', fill='white')
            dwg.add(polyline)
            break
        polyline = svgwrite.shapes.Polyline(points=self.perimeter * scale * self.svg_scale,
                                            stroke='black')
        polyline.fill('white', opacity=0)
        # dwg.add(polyline)
        debug()
        try:
            dwg.save()
        except Exception as exc:
            print(exc)
            debug()

    def __repr__(self):
        return '%d x %d %s' % (self.puzzle_dim[0],
                               self.puzzle_dim[1],
                               self.__class__.__name__)


class SquarePuzzle(Puzzle):
    def __init__(self, baseline_type=None, **kwargs):
        if baseline_type == 'curved':
            baseline_type = 'spline'
        self.baseline_type = baseline_type or 'quadratic'
        super(SquarePuzzle, self).__init__(**kwargs)

    def generate_cut_baselines(self):
        # straight baselines don't care about tab pattern,
        # curved baselines do...
        self.baselines = []

        pts_per_piece = 200  # needs to match tab definition, until i improve interpolation

        W, H = self.puzzle_dim
        for n in range(1, H):
            if self.baseline_type == 'straight':
                hbase = {'path': np.array([[0, n], [W, n]]) * self.piece_dim[0],
                         'num_tabs': W}
            elif self.baseline_type == 'quadratic':
                base = self.quadratic_baseline(W, pts_per_piece) + np.array([0, n])
                hbase = {'path': base,
                         'num_tabs': W}
            elif self.baseline_type == 'spline':
                base = self.spline_baseline(W, pts_per_piece) + np.array([0, n])
                hbase = {'path': base,
                         'num_tabs': W}

            self.baselines.append(hbase)

        for n in range(1, W):
            if self.baseline_type == 'straight':
                vbase = {'path': np.array([[n, 0], [n, H]]) * self.piece_dim[1],
                         'num_tabs': H}
            elif self.baseline_type == 'quadratic':
                base = np.fliplr(self.quadratic_baseline(H, pts_per_piece)) + np.array([n, 0])
                vbase = {'path': base,
                         'num_tabs': H}
            elif self.baseline_type == 'spline':
                base = np.fliplr(self.spline_baseline(H, pts_per_piece)) + np.array([n, 0])
                vbase = {'path': base,
                         'num_tabs': W}

            self.baselines.append(vbase)

    # TODO: split out to function? want to be reusable
    def quadratic_baseline(self, num_pieces, pts_per_piece=25):
        """define a baseline for a curved cut, on which the edge cut can be applied.
        uses a simple piecewise quadratic function"""

        pts_total = num_pieces * pts_per_piece
        t = np.linspace(0, num_pieces, pts_total + 1)

        y_seg, _ = quadratic_base(pts_per_piece)

        # concat y_seg's - remove last point from all but last piece
        y = []
        for n in range(num_pieces - 1):
            y.append(get_tab_sign(self.tab_pattern, n) * y_seg[0:-1])

        y.append(get_tab_sign(self.tab_pattern, num_pieces - 1) * y_seg)
        y = np.hstack(y)

        base_curve = np.vstack((t, y)).T
        return base_curve

    def spline_baseline(self, num_pieces, pts_per_piece=25):
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
        tab_signs = np.random.choice([-1, 1], num_pieces)

        angle = 15 * np.pi / 180
        power = .4

        knots = [[0, 0, power, 0]]
        for n, (left, right) in enumerate(zip(tab_signs, tab_signs[1:])):
            pos = [n + 1, 0]
            slope = left - right
            dirvec = [power * np.cos(angle), power * np.sin(slope * angle)]
            knots.append(pos + dirvec)
        knots.append([num_pieces, 0, power, 0])

        base_curve = slope_controlled_bezier_curve(np.array(knots), pts_per_piece)
        return base_curve

        # fig = plt.figure()
        # fig.add_subplot(111, aspect='equal')
        # plt.ion()
        # plt.plot(base_curve[:, 0], base_curve[:, 1])
        # plt.plot(np.arange(num_pieces + 1), np.zeros(num_pieces + 1), 'ko')
        # for n, sign in enumerate(tab_signs):
        #     plt.plot([n + 0.5], [0.5 * sign], 'r.')
        # plt.axis([-.1, num_pieces + .1, -.6, .6])
        # plt.show(block=False)
        # debug()


def quadratic_base(pts=25):
    t = np.linspace(0, 1, pts + 1)
    y = 0.25 * t * (1 - t)
    return y, t
    #base = np.hstack((t, y)).T
    #return base


class ArbitraryPieceSquarePuzzle(Puzzle):
    # idea: big central piece that looks like a giant puzzle piece
    # idea: fractal puzzle - use giant jigsaw piece, add tabs to outside of that
    def generate_cuts(self):
        """subclass entry point. must define three things:
        - cuts - any internal, simply defined cuts, that have no tabs
        - cut baselines - internal puzzle piece edge cuts, to have tabs added
        - perimeter - outer edge of puzzle"""
        unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        self.perimeter = unit_square * self.piece_dim * self.puzzle_dim

        self.cuts = []
        self.generate_tabless_pieces()
        self.generate_cut_baselines()
        self._add_tabs_to_baselines()

    def generate_tabless_pieces(self):
        from geometry.shapes import heart_curve_square
        heart = heart_curve_square() + np.array([2, 3])
        self.cuts.append(heart)

    def generate_cut_baselines(self):
        simple_cuts = self.cuts
        grid_baselines = self.generate_grid_baselines()
        clipped = []
        for grid_cut in grid_baselines:
            clipped.append(self.clip_grid_cut(grid_cut, simple_cuts))

    def clip_cut(self, grid_cut, simple_cuts):
        # each simple_cut should remove either 0 or 1 chunk from each grid_cut - just max to min
        for cut in simple_cuts:
            find_intersections()

    def generate_grid_baselines(self):
        grid = []

        W, H = self.puzzle_dim
        for n in range(1, H):
            # horizontal cuts run left-right, stack up-down
            hbase = {'path': np.array([[0, n], [W, n]]) * self.piece_dim[0],
                     'num_tabs': W}

            grid.append(hbase)

        for n in range(1, W):
            # vertical cuts run up-down, stack left-right
            vbase = {'path': np.array([[n, 0], [n, H]]) * self.piece_dim[1],
                     'num_tabs': H}

            grid.append(vbase)

        return grid


def find_intersections(c1, c2):
    pass


class IrregularSquarePuzzle(Puzzle):
    """every piece is composed of several adjacent normal grid cells"""

    def __init__(self):
        pass

    def generate_cut_baselines(self):
        self.basecuts = []
        self.cuts = []


class HeartRingPuzzle(Puzzle):
    # http://stackoverflow.com/questions/32772638/python-how-to-get-the-x-y-coordinates-of-a-offset-spline-from-a-x-y-list-of-poi
    def __init__(self, rings=3, **kwargs):
        self.rings = rings
        Puzzle.__init__(self, **kwargs)

    def generate_cut_baselines(self):
        from geometry.shapes import heart_curve_square
        # http://alanbernstein.net/wiki/Integral_equations#physical
        heart1 = heart_curve_square(arc_degrees=200)
        heart1 = heart1 + [0, .6]
        heart2 = heart1 * 2
        heart3 = heart1 * 3

        # plt.ion()
        fig = plt.figure()
        fig.add_subplot(111, aspect='equal')
        plt.plot(heart1[:, 0], heart1[:, 1])
        plt.plot(heart2[:, 0], heart2[:, 1])
        plt.plot(heart3[:, 0], heart3[:, 1])
        # plt.show(block=False)
        plt.show()

        debug()



class LargeCenterSquarePuzzle(Puzzle):
    def __init__(self, center_dim=None, **kwargs):
        self.center_dim = center_dim or (2, 2)

        Puzzle.__init__(self, **kwargs)  # this makes so much more sense to me...
        # super(Puzzle, self).__init__(**kwargs)

        if self.puzzle_dim[0] - self.center_dim[0] < 2 or \
           self.puzzle_dim[1] - self.center_dim[1] < 2:
            print('invalid dimensions for large center piece')
            raise Exception

    def generate_simple_cuts(self):
        HEART = False
        if HEART:
            from geometry.shapes import heart_curve_square
            heart = heart_curve_square(arc_degrees=200)
            heart = heart * 0.6 + np.array(self.puzzle_dim) / 2.0 + [0, 0.25]
            self.cuts.append(heart)

    def generate_cut_baselines(self):
        # idea 1: generate square grid, recombine edges
        # bad - recombining is painful
        #
        # idea 2: generate full cuts, generate center piece,
        # split up the cuts that hit the center piece
        # bad - splitting cuts is painful
        #
        # idea 3: define everything from closed-form expressions
        # need:
        # - start and end points of split
        #   - which of the plain grid cuts get split
        #   - starting point/ending point of the two split cuts
        # - how many tabs in each split cut
        #
        #         6
        #  0  1  2  3  4  5  6
        #  x--x--x--x--x--x--x
        #  |  |  |  |  |  |  |  4
        #  x--x--x--x--x--x--x
        #  |  |  |  2  |  |  |
        #  x--x--x    2x--x--x
        #  |  |  |     |  |  |
        #  x--x--x--x--x--x--x
        #  |  |  |  |  |  |  |
        #  x--x--x--x--x--x--x
        #
        # expression for x_splits
        #
        #  W Wc xsplit          range  W/2  Wc/2  (Wc-1)/2 (W+1)/2
        #  6 2  [3]             3, 4   3    1     .5
        #  6 4  [2, 3, 4]       2, 5   3    2     1.5
        #  8 2  [4]             4, 5   4    1     .5
        #  8 4  [3, 4, 5]       3, 6   4    2     1.5
        #  8 6  [2, 3, 4, 5, 6] 2, 7   4    3     2.5
        # 10 2  [5]             5, 6   5    1     .5
        # 10 4  [4, 5, 6]       4, 7   5    2     2.5
        # range((W+1)/2 - (Wc-1)/2, (W+1)/2 + (Wc-1)/2)
        # range((W-Wc)/2 + 1, (W+Wc)/2)

        # TODO: deduplicate somehow...

        self.baselines = []

        # common variables
        W, H = self.puzzle_dim
        Wc, Hc = self.center_dim
        x_split_start = int((W - Wc) / 2 + 1)
        x_split_end = int((W + Wc) / 2 - 1)
        x_split = range(x_split_start, x_split_end + 1)
        x_center_pieces = range(x_split_start - 1, x_split_end + 1)
        y_split_start = int((H - Hc) / 2 + 1)
        y_split_end = int((H + Hc) / 2 - 1)
        y_split = range(y_split_start, y_split_end + 1)
        y_center_pieces = range(y_split_start - 1, y_split_end + 1)

        # horizontal cuts
        for n in range(1, H):
            pattern = 'random'
            # get a half-random, half-outie pattern
            if n == y_split_start - 1:
                pattern = self.get_center_border_tab_pattern(W, x_center_pieces, -1)
            elif n == y_split_end + 1:
                pattern = self.get_center_border_tab_pattern(W, x_center_pieces, 1)

            if n in y_split:
                # split this cut into two
                hbases = [{'path': np.array([[0, n], [x_split_start - 1, n]]) * self.piece_dim[0],
                           'num_tabs': (W - Wc) / 2,
                           'tab_pattern': pattern},
                          {'path': np.array([[x_split_end + 1, n], [W, n]]) * self.piece_dim[0],
                           'num_tabs': (W - Wc) / 2}]
            else:
                # single cut
                hbases = [{'path': np.array([[0, n], [W, n]]) * self.piece_dim[0],
                           'num_tabs': W,
                           'tab_pattern': pattern}]

            self.baselines.extend(hbases)

        # vertical cuts
        for n in range(1, W):
            pattern = 'random'
            if n == x_split_start - 1:
                pattern = self.get_center_border_tab_pattern(H, y_center_pieces, 1)
            elif n == x_split_end + 1:
                pattern = self.get_center_border_tab_pattern(H, y_center_pieces, -1)

            if n in x_split:
                vbases = [{'path': np.array([[n, 0], [n, y_split_start - 1]]) * self.piece_dim[1],
                           'num_tabs': (H - Hc) / 2,
                           'tab_pattern': pattern},
                          {'path': np.array([[n, y_split_end + 1], [n, H]]) * self.piece_dim[1],
                           'num_tabs': (H - Hc) / 2,
                           'tab_pattern': pattern}]
            else:
                vbases = [{'path': np.array([[n, 0], [n, H]]) * self.piece_dim[1],
                           'num_tabs': H,
                           'tab_pattern': pattern}]

            self.baselines.extend(vbases)

    def get_center_border_tab_pattern(self, width, center_list, center_sign):
        pattern = []
        for n in range(width):
            if n in center_list:
                pattern.append(center_sign)
            else:
                pattern.append(random.choice([-1, 1]))
        return pattern



# worry about these later
class VariableSizePuzzle(Puzzle):
    # center piece is biggest, smaller pieces around that, smaller around that, etc
    pass


class VoronoiPuzzle(Puzzle):
    # https://www.j-raedler.de/projects/polygon/
    # http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.KDTree.html
    # https://en.wikipedia.org/wiki/K-d_tree
    def generate_cut_baselines(self):
        from scipy.spatial import Voronoi
        points = np.random.random((7, 2))
        vor = Voronoi(points)
        debug()


class TrianglePuzzle(Puzzle):
    # could be an instance of VoronoiPuzzle
    pass


class HexagonPuzzle(Puzzle):
    # could be an instance of VoronoiPuzzle
    pass


class SpiralPuzzle(Puzzle):

    def __init__(self,
                 spiral_spacing=1,
                 spiral_turns=4,
                 radial_spacing=1,
                 edge_orientation=None,
                 tab_parameters=None, **kwargs):
        self.spiral_spacing = spiral_spacing
        self.spiral_turns = spiral_turns
        self.radial_spacing = radial_spacing
        self.edge_orientation = edge_orientation or 'random'
        self.tab_parameters = tab_parameters
        super(Puzzle, self).__init__(**kwargs)

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
