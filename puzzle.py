#!/usr/bin/python
"""
Terminology
- puzzle - the whole thing
- puzzle cut - one continuous path, composed of one or more 'puzzle piece edge'
- puzzle piece edge - 
- puzzle piece - a simple connected region, bounded by puzzle piece edges
"""
import numpy as np
import matplotlib.pyplot as plt
import random

from plot_tools import terminal_plot
from geometry.curves import frenet_frame, add_frenet_offset_2D

from pyzzle.edge import (create_puzzle_piece_edge,
                         get_default_tab_parameters,
                         SEGMENTS_PER_PIECE)

from panda.debug import debug, pm


def main():
    test_puzzle()


def test_puzzle():
    cols, rows = 6, 4
    puzz = SquareTiledPuzzle(puzzle_dim=(cols, rows),
                             cut_type='curved')
    puzz.write_svg()
    fig = plt.figure()
    fig.add_subplot(111, aspect='equal')
    puzz.plot(color='k')
    plt.axis([-.1, cols + .1, -.1, rows + .1])
    plt.show()


class Puzzle(object):
    svg_filename = 'puzzle_cuts.svg'
    svg_scale = 25.4  # TODO: fix this

    def __init__(self, **kwargs):
        pass

    def generate_baselines(self):
        """must define in child.
        this function should use any means to define both:
        - self.baselines
        - self.perimeter"""
        self.baselines = []
        self.perimeter = []

    def plot(self, **kwargs):
        # caller is responsible for plt.show() etc
        ph = []
        for cut, base in zip(self.cuts, self.base):
            ph.append(plt.plot(cut[:, 0], cut[:, 1], **kwargs))
            ph.append(plt.plot(base[:, 0], base[:, 1], 'r--'))

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
        debug()
        dwg.save()


class SquareTiledPuzzle(Puzzle):
    # TODO: this should define baselines:
    # - path
    # - piece_spacing OR piece_positions
    # - type: {'line', 'curve'} (because line can't be handled by 
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
        self.cuts = []
        self.base = []
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
            self.base.append(hbase)

        cutter.num_pieces = self.puzzle_dim[1]
        cutter.piece_sizes = self.piece_dim[1]
        for n in range(1, self.puzzle_dim[0]):
            # vertical cuts run up-down, stack left-right
            cut, base = cutter.generate()
            vcut = (np.fliplr(cut) + np.array([n, 0])) * self.piece_dim[1]
            vbase = (np.fliplr(base) + np.array([n, 0])) * self.piece_dim[1]
            self.cuts.append(vcut)
            self.base.append(vbase)

        unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        self.perimeter = unit_square * self.piece_dim * self.puzzle_dim

    def quadratic_baseline(self):
        # TODO: move this to here from PuzzleCutter
        pass


# TODO: these
# class TriangleTiledPuzzle(Puzzle):
# class HexagonTiledPuzzle(Puzzle):

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


# class CurvedPuzzleCut(PuzzleCut):
# class IrregularStraightPuzzleCut(PuzzleCut):  # non-even piece spacing
# class IrregularCurvedPuzzleCut(PuzzleCut):    # similar
# all four of these can be unified, by adding:
# - cut_path -> straight line if not supplide
# - piece_size -> list gives full spacing control
# in that case, why not use a function?

# split into LineCutter (handled trivially)
# and CurveCutter (handled via 

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
        xy, straight = create_puzzle_cut_curved(base_curve, t,
                                                self.num_pieces,
                                                pts_per_segment,
                                                self.tab_parameters)  # this is not grid-specific

        self.points = np.vstack(xy)
        return self.points, base_curve

    def generate_straight(self): # this is grid-specific
        xy = []
        for n in range(self.num_pieces):
            self.tab_parameters.randomize()
            xy0, _ = create_puzzle_piece_edge(self.tab_parameters)
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
        pass


random_sign = lambda: random.choice([1, -1])


def create_puzzle_cut_straight_irregular(piece_spacing):
    pass


def create_puzzle_cut_straight(num_pieces):
    """generate a cut on a straight baseline, with multiple edges"""
    # return xy of horizontal cut
    pset = get_default_tab_parameters()
    xy = []
    for n in range(num_pieces):
        pset.randomize()
        xy0, _ = create_puzzle_piece_edge(pset)
        xy0[:, 0] += n
        xy0[:, 1] *= random_sign()
        xy.append(xy0)

    return np.vstack(xy)


def create_puzzle_cut_curved(base_curve, t, num_pieces, pts_per_segment, tab_parameters=None):
    """given a base curve, maps puzzle pieces onto that curve smoothly"""
    T, N, B = frenet_frame(base_curve)
    tab_parameters = tab_parameters or get_default_tab_parameters()
    pts_per_piece = pts_per_segment * SEGMENTS_PER_PIECE

    tt = np.linspace(0, 1, pts_per_piece + 1)
    null_base_curve = np.vstack((tt, 0 * tt)).T

    straights = []
    xy = []
    for n in range(num_pieces):
        # get a piece edge
        tab_parameters.randomize()
        pxy, _ = create_puzzle_piece_edge(tab_parameters, pts_per_segment)
        dxy = pxy - null_base_curve
        straights.append(pxy + [n, 0])

        # conform it to the current segment of the curve
        # TODO: understand why the conformed edge is on both sides of the base curve...
        # the reason: the straight segment of the edge definition does not correspond
        # to the straight segment of the null_base_curve - the parameter moves faster
        # on the edge
        # this could be "fixed" by making the parameterization match for the straight segments
        # NOT by naturalizing the curve parameter
        start = pts_per_piece * n
        end = pts_per_piece * (n + 1) + 1
        base_segment = base_curve[start:end]
        T_segment = T[start:end]
        N_segment = N[start:end]

        new = add_frenet_offset_2D(base_segment, dxy, T_segment, N_segment)
        xy.extend(new)

    return np.vstack(xy), np.vstack(straights)


if __name__ == '__main__':
    main()
