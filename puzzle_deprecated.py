#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import random

from geometry.curves import (frenet_frame_2D_corrected,
                             add_frenet_offset_2D)
from pyzzle.edge import (create_puzzle_piece_edge,
                         get_default_tab_parameters,
                         SEGMENTS_PER_PIECE)

from panda.debug import debug, pm


def main():
    test_square_tiled_puzzle()


@pm
def test_square_tiled_puzzle():
    size = 1
    cols, rows = 1, 2
    puzz = SquareTiledPuzzle(puzzle_dim=(cols, rows),
                             piece_dim=(size, size),
                             cut_type='curved')
    fig = plt.figure()
    fig.add_subplot(111, aspect='equal')
    puzz.plot(color='k')
    plt.axis([-.1, size * cols + .1, -.1, size * rows + .1])
    plt.show()
    debug()


# this was the first design that basically worked,
# but it's been 95% rewritten in puzzle.py and cut.py
# still keeping it around for reference until the new
# version is verified, but can delete after that
class SquareTiledPuzzle(object):
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

        cutter.num_pieces = self.puzzle_dim[1]
        cutter.piece_sizes = self.piece_dim[1]
        for n in range(1, self.puzzle_dim[0]):
            # vertical cuts run up-down, stack left-right
            cut, base = cutter.generate()
            vcut = (np.fliplr(cut) + np.array([n, 0])) * self.piece_dim[1]
            vbase = (np.fliplr(base) + np.array([n, 0])) * self.piece_dim[1]
            self.cuts.append(vcut)
            self.baselines.append(vbase)

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


def create_puzzle_cut_curved(base_curve, num_pieces, pts_per_segment, tab_parameters=None):
    """given a base curve, maps puzzle pieces onto that curve smoothly"""
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
        print('  piece %d - %d points' % (n, len(new)))
        xy.extend(new)

    pts = np.vstack(xy)

    return pts, np.vstack(straights)


random_sign = lambda: random.choice([1, -1])


if __name__ == '__main__':
    main()
