#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import random

from plot_tools import terminal_plot
from geometry.curves import frenet_frame, add_frenet_offset_2D

from pyzzle.edge import (create_puzzle_piece_edge,
                         get_default_nub_parameters,
                         SEGMENTS_PER_PIECE)

from panda.debug import debug, pm


# i was able to draw a decent puzzle piece edge using illustrator's pen tool
# - 9 control points, each with controlled slope
# - illustrator uses bezier

# 8x10 pieces that are 2"x2" should be plenty - 16"x20" full size


class Puzzle(object):
    svg_filename = 'puzzle_cuts.svg'
    svg_scale = 25.4  # TODO: fix this

    def __init__(self, **kwargs):
        pass

    def generate(self):
        """must define in child.
        this function should define paths for both self.cuts and self.perimeter,
        by any means. """
        self.cuts = []
        self.perimeter = []

    def plot(self, **kwargs):
        # caller is responsible for plt.show() etc
        ph = []
        for cut in self.cuts:
            ph.append(plt.plot(cut[:, 0], cut[:, 1], **kwargs))

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


class SquareTiledPuzzle(Puzzle):
    def __init__(self,
                 piece_dim=None,
                 puzzle_dim=None,
                 cut_type=None,
                 edge_orientation=None,
                 nub_parameters=None):
        self.piece_dim = piece_dim or [1, 1]    # in measurement units
        self.puzzle_dim = puzzle_dim or [4, 6]  # in puzzle pieces
        self.cut_type = cut_type or 'straight'
        self.edge_orientation = edge_orientation or 'random'
        self.nub_parameters = nub_parameters or get_default_nub_parameters()
        self.generate()

    def generate(self):
        self.cuts = []
        cutter = PuzzleCutter(num_pieces=self.puzzle_dim[0],
                              piece_size=self.piece_dim[0],
                              cut_type=self.cut_type,
                              edge_orientation=self.edge_orientation)

        for n in range(1, self.puzzle_dim[1]):
            # horizontal cuts run left-right, stack up-down
            cut = cutter.generate()
            hcut = cut + np.array([0, n])
            self.cuts.append(hcut)

        cutter.num_pieces = self.puzzle_dim[1]
        cutter.piece_sizes = self.piece_dim[1]
        for n in range(1, self.puzzle_dim[0]):
            # vertical cuts run up-down, stack left-right
            cut = cutter.generate()
            vcut = np.fliplr(cut) + np.array([n, 0])
            self.cuts.append(vcut)

        unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        self.perimeter = unit_square * self.piece_dim * self.puzzle_dim


# TODO: these
# class TriangleTiledPuzzle(Puzzle):
# class HexagonTiledPuzzle(Puzzle):
# class SpiralPuzzle(Puzzle):


# class CurvedPuzzleCut(PuzzleCut):
# class IrregularStraightPuzzleCut(PuzzleCut):  # non-even piece spacing
# class IrregularCurvedPuzzleCut(PuzzleCut):    # similar
# all four of these can be unified, by adding:
# - cut_path -> straight line if not supplide
# - piece_size -> list gives full spacing control
# in that case, why not use a function?
class PuzzleCutter(object):
    """Generates a single 'puzzle cut'
    by default:
    - cut is straight, and along the x-axis
    - length = num_pieces * piece_size"""
    def __init__(self,
                 num_pieces=1,
                 piece_size=1,
                 cut_type=None,
                 edge_orientation=None,
                 nub_parameters=None):
        self.num_pieces = num_pieces
        self.piece_size = piece_size
        self.cut_type = cut_type or 'straight'
        self.edge_orientation = edge_orientation or 'random'
        self.nub_parameters = nub_parameters or get_default_nub_parameters()
        self.generate()

    def generate(self):
        if self.cut_type == 'straight':
            return self.generate_straight()
        elif self.cut_type == 'curved':
            return self.generate_curved()

    # TODO: try to unify these two?
    def generate_curved(self):
        base_curve, t, pts_per_segment = cut_baseline_quadratic(self.num_pieces)
        xy, straight = create_curved_cut(base_curve, t,
                                         self.num_pieces,
                                         pts_per_segment,
                                         self.nub_parameters)

        self.points = np.vstack(xy)
        return self.points

    def generate_straight(self):
        xy = []
        for n in range(self.num_pieces):
            self.nub_parameters.randomize()
            xy0, _ = create_puzzle_piece_edge(self.nub_parameters)
            xy0 *= self.piece_size
            xy0[:, 0] += self.piece_size * n
            if self.edge_orientation == 'random':
                xy0[:, 1] *= random_sign()
            elif self.edge_orientation == 'alternating':
                xy0[:, 1] *= (-1) ** n
            xy.append(xy0)

        self.points = np.vstack(xy)
        return self.points


def main():
    # plot_grid_puzzle_old()
    # plot_curvy_grid_puzzle()
    # draw_svg_curvy_grid_puzzle()

    plot_grid_puzzle(cut_type='curved')


random_sign = lambda: random.choice([1, -1])


def plot_grid_puzzle(**kwargs):
    cols, rows = 6, 4
    puzz = SquareTiledPuzzle(puzzle_dim=(cols, rows), **kwargs)
    fig = plt.figure()
    fig.add_subplot(111, aspect='equal')
    puzz.plot(color='k')
    plt.axis([-.1, cols + .1, -.1, rows + .1])
    plt.show()


def draw_svg_curvy_grid_puzzle(rows=6, cols=8):
    import svgwrite
    import svgtools
    svg_name = 'puzzle.svg'
    dwg = svgwrite.Drawing(svg_name, profile='tiny')

    piece_size = 2
    scale = 25.4
    for n in range(1, rows):
        print('horizontal cut %d' % n)
        cut, straight, base_curve = create_puzzle_cut_curved_quadratic(cols)
        cut[:, 1] += n
        cut *= piece_size * scale
        polyline = svgwrite.shapes.Polyline(points=cut, stroke='black', fill='white')
        #path = svgtools.path_from_array(cut, stroke='black', fill='white')
        dwg.add(polyline)

    for n in range(1, cols):
        print('vertical cut %d' % n)
        cut, straight, base_curve = create_puzzle_cut_curved_quadratic(rows)
        cut[:, 1] += n
        cut = np.fliplr(cut)  # transpose each vector
        cut *= piece_size * scale
        polyline = svgwrite.shapes.Polyline(points=cut, stroke='black', fill='white')
        #path = svgtools.path_from_array(cut, stroke='black', fill='white')
        dwg.add(polyline)

    print('perimeter cut')
    perimeter = np.array([[0, 0],
                          [piece_size*cols, 0],
                          [piece_size*cols, piece_size*rows],
                          [0, piece_size*rows],
                          [0, 0]]) * scale
    polyline = svgwrite.shapes.Polyline(points=perimeter, stroke='black')
    polyline.fill('white', opacity=0)
    dwg.add(polyline)

    print('finished drawing %s' % svg_name)
    dwg.save()


def plot_curvy_grid_puzzle(rows=6, cols=8):
    piece_size = 2
    fig = plt.figure()
    fig.add_subplot(111, aspect='equal')
    for n in range(1, rows):
        cut, straight, base_curve = create_puzzle_cut_curved_quadratic(cols)
        plt.plot(piece_size*(cut[:, 0]), piece_size*(cut[:, 1] + n), 'b-')
        plt.plot(piece_size*base_curve[:, 0], piece_size*(base_curve[:, 1] + n), 'r--')

    for n in range(1, cols):
        cut, straight, base_curve = create_puzzle_cut_curved_quadratic(rows)
        plt.plot(piece_size*(cut[:, 1] + n), piece_size*cut[:, 0], 'b-')
        plt.plot(piece_size*(base_curve[:, 1] + n), piece_size*base_curve[:, 0], 'r--')

    perim_x = [0, piece_size*cols, piece_size*cols, 0, 0]
    perim_y = [0, 0, piece_size*rows, piece_size*rows, 0]
    plt.plot(perim_x, perim_y, 'k-')
    d = .02
    plt.subplots_adjust(bottom=d, left=d, top=1-d, right=1-d)
    #plt.axis([-.5, cols+.5, -.5, rows+.5])
    #terminal_plot()
    plt.show()


def cut_baseline_quadratic(num_pieces, pts_per_segment=25):
    """ define a baseline for a curved cut, on which the edge cut can be applied
    using a quadratic function"""

    pts_per_piece = pts_per_segment * SEGMENTS_PER_PIECE
    pts_total = num_pieces * pts_per_piece
    
    t0 = np.linspace(0, 1, pts_per_piece + 1)
    t = np.linspace(0, num_pieces, pts_total + 1)
    x = t
    y_seg = 0.25 * t0 * (1-t0)
    # concat y_seg's
    y = []
    for n in range(num_pieces - 1):
        y.append(random_sign() * y_seg[0:-1])
    y.append(random_sign() * y_seg)
    y = np.hstack(y)

    base_curve = np.vstack((x, y)).T

    return base_curve, t, pts_per_segment
    

def cut_baseline_spline(num_pieces, pts_per_segment=25):
    # TODO: base curve should flatten out rather than have sharp points
    pass
    

def create_puzzle_cut_curved_quadratic(num_pieces, nub_parameters=None):
    """ returns a single cut for the specified number of pieces,
    using a curved baseline"""
    
    base_curve, t, pts_per_segment = cut_baseline_quadratic(num_pieces)
    cut, straight = create_curved_cut(base_curve, t, num_pieces, pts_per_segment, nub_parameters)
    return cut, straight, base_curve


def create_curved_cut(base_curve, t, num_pieces, pts_per_segment, nub_parameters=None):
    """ """
    # TODO: reparameterize the curve on arclength
    T, N, B = frenet_frame(base_curve)
    pieces = []
    nub_parameters = nub_parameters or get_default_nub_parameters()
    pts_per_piece = pts_per_segment * SEGMENTS_PER_PIECE

    tt = np.linspace(0, 1, pts_per_piece + 1)
    null_base_curve = np.vstack((tt, 0 * tt)).T

    straights = []
    xy = []
    for n in range(num_pieces):
        # get a piece edge
        nub_parameters.randomize()
        pxy, _ = create_puzzle_piece_edge(nub_parameters, pts_per_segment)
        dxy = pxy - null_base_curve
        straights.append(pxy + [n, 0])

        # conform it to the current segment of the curve
        # TODO: understand why the conformed edge is on both sides of the base curve...
        # the reason: the straight segment of the edge definition does not correspond
        # to the straight segment of the null_base_curve - the parameter moves faster
        # on the edge
        start = pts_per_piece * n
        end = pts_per_piece * (n + 1) + 1
        base_segment = base_curve[start:end]
        T_segment = T[start:end]
        N_segment = N[start:end]

        new = add_frenet_offset_2D(base_segment, dxy, T_segment, N_segment)
        xy.extend(new)

    return np.vstack(xy), np.vstack(straights)


def archimedean_spiral(t):
    # use as base for spiral puzzle cut
    r = 2 * t
    th = t * 2 * np.pi
    x = r * np.cos(th)
    y = r * np.sin(th)
    return np.vstack((x, y)).T


if __name__ == '__main__':
    main()


# deprecated

def plot_grid_puzzle_old(rows=4, cols=6):
    fig = plt.figure()
    fig.add_subplot(111, aspect='equal')
    for n in range(1, rows):
        hcut = create_puzzle_cut_straight(cols)
        plt.plot(hcut[:, 0], hcut[:, 1] + n, 'k-')

    for n in range(1, cols):
        vcut = create_puzzle_cut_straight(rows)
        plt.plot(vcut[:, 1] + n, vcut[:, 0], 'k-')

    plt.plot([0, cols, cols, 0, 0], [0, 0, rows, rows, 0], 'k-')
    plt.axis([-.5, cols + .5, -.5, rows + .5])
    plt.show()


def create_puzzle_cut_straight(num_pieces):
    """generate a cut on a straight baseline, with multiple edges"""
    # return xy of horizontal cut
    pset = get_default_nub_parameters()
    xy = []
    for n in range(num_pieces):
        pset.randomize()
        xy0, _ = create_puzzle_piece_edge(pset)
        xy0[:, 0] += n
        xy0[:, 1] *= random_sign()
        xy.append(xy0)

    return np.vstack(xy)


