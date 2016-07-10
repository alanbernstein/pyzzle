#!/usr/bin/python
"""

"""
import numpy as np
import matplotlib.pyplot as plt
import random

from panda.plot_utils import terminal_plot, qplot
from geometry.curves import (frenet_frame_2D_corrected,
                             add_frenet_offset_2D,
                             curve_length)

from pyzzle.edge import (create_puzzle_piece_edge,
                         get_default_tab_parameters,
                         SEGMENTS_PER_PIECE)

from panda.debug import debug, pm


def main():
    test_puzzle()
    #test_square_tiled_puzzle()

@pm
def test_puzzle():
    size = 1
    cols, rows = 4, 5
    puzz = Puzzle(puzzle_dim=(cols, rows),
                  piece_dim=(size, size))

    fig = plt.figure()
    fig.add_subplot(111, aspect='equal')
    puzz.plot(color='k')
    plt.axis([-.1, size * cols + .1, -.1, size * rows + .1])
    plt.show()


@pm
def test_square_tiled_puzzle():
    size = 1
    cols, rows = 4, 5
    puzz = SquareTiledPuzzle(puzzle_dim=(cols, rows),
                             piece_dim=(size, size),
                             cut_type='curved')
    #puzz.write_svg()
    fig = plt.figure()
    fig.add_subplot(111, aspect='equal')
    puzz.plot(color='k')
    plt.axis([-.1, size * cols + .1, -.1, size * rows + .1])
    plt.show()


# TODO: move to base.py, move basecuts out into example class SquarePuzzle
class Puzzle(object):
    svg_filename = 'puzzle_cuts.svg'
    svg_scale = 100  # svg default unit is .01" (in corel at least)

    def __init__(self, piece_dim=None, puzzle_dim=None, cut_type=None,
                 tab_pattern=None, tab_parameters=None):
        """handle inputs, run top-level generate method"""

        self.piece_dim = piece_dim or [1, 1]    # measurement units
        self.puzzle_dim = puzzle_dim or [4, 6]  # puzzle pieces
        self.cut_type = cut_type or 'straight'
        self.tab_pattern = tab_pattern or 'random'
        self.tab_parameters = tab_parameters or get_default_tab_parameters()

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
        for base in self.baselines:
            # ***************************
            # finish this and refactor is complete, then can move on to new features
            # somewhere in here is a problem that causes the last sample of a
            # cut to be omitted
            self.cuts.append(PuzzleCutter(**base).generate())
            print('len cut = %d' % len(self.cuts[-1]))
            print(self.cuts[-1][0::201])


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
        return '%d x %d rectangular puzzle' % self.puzzle_dim


# TODO: put this inside SquarePuzzle, with baseline_type or something
class CurvedBaselineSquarePuzzle(Puzzle):
    def generate_cut_baselines(self):
        self.baselines = []

        W, H = self.puzzle_dim
        for n in range(1, H):
            hbase = {'path': np.array([[0, n], [W, n]]) * self.piece_dim[0],
                     'num_tabs': W}

            self.baselines.append(hbase)

        for n in range(1, W):
            vbase = {'path': np.array([[n, 0], [n, H]]) * self.piece_dim[1],
                     'num_tabs': H}

            self.baselines.append(vbase)

    def quadratic_baseline(self):
        # TODO: implement - not quite the same as the earlier version
        pass

    def spline_baseline(self):
        # TODO: implement
        # TODO: might be hard to pass tab_positions for this, with different
        # lengths for each tab... try using index, rather than distance along curve
        pass




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
        self.tab_parameters = tab_parameters or get_default_tab_parameters()

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


# split into cut.py
# TODO: understand why the conformed edge is on both sides of the base curve...
# the reason: the straight segment of the edge definition does not correspond
# to the straight segment of the null_base_curve - the parameter moves faster
# on the edge
# this could be "fixed" by making the parameterization match for the straight segments
# NOT by naturalizing the curve parameter

class PuzzleCutter(object):
    """Accepts a plane curve, and transforms it into a "puzzle cut"
    with specified number of pieces, tab parameters, etc
    different ways to init:
    PuzzleCutter(path, num_tabs) - equally spaced tabs
    PuzzleCutter(path, positions) - arbitrary control of tab positions, relative to arc length along path
    PuzzleCutter(path) - one tab in center"""
    # Line - path=straight line, num_tabs=whatever
    # Curved - path=curve
    # IrregularStraight - positions=[...]
    # IrregularCurved - path=curve, positions=[...]
    
    def __init__(self, path=None, num_tabs=None, positions=None,
                 tab_pattern=None, tab_parameters=None):
        self.path = path if path is not None else np.array([[0, 0], [length, 0]])
        # TODO: consider saving the curve object into this object
        # that might be too much coupling
        self.base_length = curve_length(path)
        self.tab_pattern = tab_pattern or 'random'
        self.tab_parameters = tab_parameters or get_default_tab_parameters()

        # handle variable init methods
        if positions is not None:
            self.positions = positions
            self.num_tabs = len(self.positions)
            # TODO: handle arbitrary positions properly
        else:
            if num_tabs:
                self.num_tabs = num_tabs
            else:
                # this assumes greater than 1
                self.num_tabs = np.floor(self.base_length)
            self.positions = get_monospaced_positions(self.base_length, self.num_tabs)


    def generate(self):
        pts_per_segment = 25 # TODO find a better place for this - maybe tab_parameters
        pts_per_tab = pts_per_segment * SEGMENTS_PER_PIECE

        self.base_curve = resample_curve(self.path, pts_per_tab * self.num_tabs)
        T, N = frenet_frame_2D_corrected(self.base_curve)

        xy = []
        for n in range(self.num_tabs):
            # get a piece edge
            self.tab_parameters.randomize()
            pxy, dxy, _ = create_puzzle_piece_edge(self.tab_parameters, pts_per_segment)

            # handle tab direction pattern
            if self.tab_pattern == 'random':
                dxy[:, 1] *= random.choice([-1, 1])
            elif self.tab_pattern == 'alternating':
                dxy[:, 1] *= (-1) ** n

            # conform it to the current segment of the curve
            start = pts_per_tab * n
            end = pts_per_tab * (n + 1) + 1
            base_section = self.base_curve[start:end]
            T_segment = T[start:end]
            N_segment = N[start:end]
            new = add_frenet_offset_2D(base_section, dxy, T_segment, N_segment)
            # TODO: might need to remove last point or something
            xy.extend(new)

        #cut = np.vstack(xy)
        #qplot(cut)
        #debug()

        self.points = np.vstack(xy)
        return self.points


def resample_curve(r, new_length):
    """resample NxD curve to MxD, where
    M is the second input new_length"""
    # TODO: consider better resampling? interp might have something
    t = np.linspace(0, 1, len(r))
    ti = np.linspace(0, 1, new_length) # TODO double check, off by one
    return np.vstack([np.interp(ti, t, c) for c in r.T]).T


def get_monospaced_positions(L, N):
    """center-justified, equally-spaced points
    interval of length L, N points"""
    return np.arange(0, L, L/N) + L/(2*N)

    
# deprecated
class SquareTiledPuzzle(Puzzle):
    # TODO: this should define baselines:
    # - path
    # - piece_spacing OR piece_positions
    # - type: {'line', 'curve'} (because line can't be handled by frenet_frame)
    #   idea: just write a function line_frenet() with same signature as frenet_frame,
    #         but produces some well-defined normal and binormal for a straight line
    def __init__(self,
                 piece_dim=None,
                 puzzle_dim=None,
                 cut_type=None,
                 edge_orientation=None,
                 tab_parameters=None):
        self.piece_dim = piece_dim or [1, 1]    # in measurement units
        self.puzzle_dim = puzzle_dim or [4, 6]  # in puzzle pieces
        self.cut_type = cut_type or 'straight'
        self.edge_orientation = edge_orientation or 'random'
        self.tab_parameters = tab_parameters or get_default_tab_parameters()
        self.generate()

    def generate(self):
        # TODO: match parent API
        self.cuts = []
        self.baselines = []
        cutter = PuzzleGridCutter(num_pieces=self.puzzle_dim[0],
                                  piece_size=self.piece_dim[0],
                                  cut_type=self.cut_type,
                                  edge_orientation=self.edge_orientation)

        for n in range(1, self.puzzle_dim[1]):
            # horizontal cuts run left-right, stack up-down
            cut, base = cutter.generate()
            hcut = (cut + np.array([0, n])) * self.piece_dim[0]
            hbase = (base + np.array([0, n])) * self.piece_dim[0]
            self.cuts.append(hcut)
            self.baselines.append(hbase)
            print('len cut = %d' % len(self.cuts[-1]))
            print(self.cuts[-1][0::201])

        cutter.num_pieces = self.puzzle_dim[1]
        cutter.piece_sizes = self.piece_dim[1]
        for n in range(1, self.puzzle_dim[0]):
            # vertical cuts run up-down, stack left-right
            cut, base = cutter.generate()
            vcut = (np.fliplr(cut) + np.array([n, 0])) * self.piece_dim[1]
            vbase = (np.fliplr(base) + np.array([n, 0])) * self.piece_dim[1]
            self.cuts.append(vcut)
            self.baselines.append(vbase)
            print('len cut = %d' % len(self.cuts[-1]))
            print(self.cuts[-1][0::201])


        unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        self.perimeter = unit_square * self.piece_dim * self.puzzle_dim

    def plot(self, **kwargs):
        # caller is responsible for plt.show() etc
        ph = []
        for cut, base in zip(self.cuts, self.baselines):
            ph.append(plt.plot(cut[:, 0], cut[:, 1], **kwargs))
            ph.append(plt.plot(base[:, 0], base[:, 1], 'r--'))

        # add perimeter last
        ph.append(plt.plot(self.perimeter[:, 0], self.perimeter[:, 1], **kwargs))
        return ph


# should be deprecated
def create_puzzle_cut_curved(base_curve, num_pieces, pts_per_segment, tab_parameters=None):
    """given a base curve, maps puzzle pieces onto that curve smoothly"""
    # TODO: have this use corrected normal, then can eliminate cut_straight
    #T, N, B = frenet_frame(base_curve)
    T, N = frenet_frame_2D_corrected(base_curve)
    tab_parameters = tab_parameters or get_default_tab_parameters()
    pts_per_piece = pts_per_segment * SEGMENTS_PER_PIECE

    straights = []
    xy = []
    for n in range(num_pieces):
        # get a piece edge
        tab_parameters.randomize()
        pxy, dxy, _ = create_puzzle_piece_edge(tab_parameters, pts_per_segment)
        straights.append(pxy + [n, 0])

        start = pts_per_piece * n
        end = pts_per_piece * (n + 1) + 1
        base_segment = base_curve[start:end]
        T_segment = T[start:end]
        N_segment = N[start:end]

        new = add_frenet_offset_2D(base_segment, dxy, T_segment, N_segment)
        xy.extend(new)

    return np.vstack(xy), np.vstack(straights)

    
                    

# this should be deprecated when PuzzleCutter is done

class PuzzleGridCutter(object):
    """Generates a single 'puzzle cut'
    by default:
    - cut is straight, and along the x-axis
    - length = num_pieces * piece_size"""
    def __init__(self,
                 num_pieces=1,
                 piece_size=1,
                 cut_type=None,
                 edge_orientation=None,
                 tab_parameters=None):
        self.num_pieces = num_pieces
        self.piece_size = piece_size
        self.cut_type = cut_type or 'straight'
        self.edge_orientation = edge_orientation or 'random'
        self.tab_parameters = tab_parameters or get_default_tab_parameters()
        self.generate()

    def generate(self):
        if self.cut_type == 'straight':
            return self.generate_straight()
        elif self.cut_type == 'curved':
            return self.generate_curved()

    # TODO: try to unify these two?
    # they can probably also bubble up to a parent class.
    # can't actually use a straight line as the baseline because of undefined
    # normal vector, so going to have to split it up somehow anyway.
    def generate_curved(self):
        base_curve, t, pts_per_segment = self.quadratic_baseline()  # this is grid-specific
        xy, straight = create_puzzle_cut_curved(base_curve,
                                                self.num_pieces,
                                                pts_per_segment,
                                                self.tab_parameters)  # this is not grid-specific

        self.points = np.vstack(xy)
        return self.points, base_curve

    def generate_straight(self):  # this is grid-specific
        xy = []
        for n in range(self.num_pieces):
            self.tab_parameters.randomize()
            xy0, dxy, _ = create_puzzle_piece_edge(self.tab_parameters)
            xy0 *= self.piece_size
            xy0[:, 0] += self.piece_size * n
            if self.edge_orientation == 'random':
                xy0[:, 1] *= random_sign()
            elif self.edge_orientation == 'alternating':
                xy0[:, 1] *= (-1) ** n
            xy.append(xy0)

        self.points = np.vstack(xy)
        return self.points, None

    def quadratic_baseline(self, pts_per_segment=25):
        """ define a baseline for a curved cut, on which the edge cut can be applied.
        uses a simple piecewise quadratic function"""

        pts_per_piece = pts_per_segment * SEGMENTS_PER_PIECE
        pts_total = self.num_pieces * pts_per_piece

        t0 = np.linspace(0, 1, pts_per_piece + 1)
        t = np.linspace(0, self.num_pieces, pts_total + 1)
        x = t
        y_seg = 0.25 * t0 * (1 - t0)
        # concat y_seg's
        y = []
        for n in range(self.num_pieces - 1):
            y.append(random_sign() * y_seg[0:-1])
        y.append(random_sign() * y_seg)
        y = np.hstack(y)

        base_curve = np.vstack((x, y)).T

        return base_curve, t, pts_per_segment

    def spline_baseline(self, pts_per_segment=25):
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


random_sign = lambda: random.choice([1, -1])


if __name__ == '__main__':
    main()
