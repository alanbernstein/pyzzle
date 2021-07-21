import numpy as np
import random

from geometry.curves import (frenet_frame_2D,
                             frenet_frame_2D_corrected,
                             add_frenet_offset_2D,
                             curve_length)

from edge import (create_puzzle_piece_edge,
                  get_default_tab_parameters,
                  SEGMENTS_PER_PIECE)

from ipdb import iex, set_trace as debug


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

    pts_per_segment = 25  # TODO find a better place for this - maybe tab_parameters
    def __init__(self, path=None, num_tabs=None, positions=None,
                 tab_pattern=None, tab_parameters=None):
        # TODO: consider saving the curve object into this object
        # that might be too much coupling
        self.base_length = curve_length(path)
        self.path = path if path is not None else np.array([[0, 0], [self.base_length, 0]])
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
                self.num_tabs = int(np.floor(self.base_length))
            self.positions = get_monospaced_positions(self.base_length, self.num_tabs)

    def generate(self):
        pts_per_tab = self.pts_per_segment * SEGMENTS_PER_PIECE
        self.base_curve = resample_curve(self.path, pts_per_tab * self.num_tabs + 1)
        # TODO: resampling seems like a good idea, but order-1 resample is no good -
        # it screws up the derivative. need something more sophisticated to work
        T, N = frenet_frame_2D_corrected(self.base_curve)
        self.tangent = T
        self.normal = N

        xy = []
        news = []
        for n in range(int(self.num_tabs)):
            # get a piece edge
            self.tab_parameters.randomize()
            pxy, dxy, _ = create_puzzle_piece_edge(self.tab_parameters, self.pts_per_segment)

            # handle tab direction pattern
            sign = get_tab_sign(self.tab_pattern, n)
            dxy[:, 1] *= sign

            # conform it to the current segment of the curve
            start = pts_per_tab * n
            end = pts_per_tab * (n + 1) + 1
            base_section = self.base_curve[start:end]
            T_segment = T[start:end]
            N_segment = N[start:end]
            new = add_frenet_offset_2D(base_section, dxy, T_segment, N_segment)
            # remove final point, for all but the last segment
            if n < self.num_tabs - 1:
                new = new[0:-1, :]

            news.append(new)
            xy.extend(new)

        self.points = np.vstack(xy)

        return self.points

    def plot(self):
        if not hasattr(self, 'points'):
            self.generate()

        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.ion()
        fig.add_subplot(111, aspect='equal')

        plt.plot(self.path[:, 0], self.path[:, 1], 'k-')
        plt.plot(self.base_curve[:, 0], self.base_curve[:, 1], 'r--')
        plt.plot(self.points[:, 0], self.points[:, 1], 'g')

        num_vecs = 12
        step = int((len(self.base_curve)-1) / num_vecs)
        for p, t0, n0 in zip(self.base_curve, self.tangent, self.normal)[::step]:
            txy = np.vstack((p, p + t0[0:2]))
            plt.plot(txy[:, 0], txy[:, 1], 'c-')
            nxy = np.vstack((p, p + n0[0:2]))
            plt.plot(nxy[:, 0], nxy[:, 1], 'm-')


#        fig.add_subplot(212, aspect='equal')
#        for pxy in self.pxys:
#            plt.plot(pxy[:, 0], pxy[:, 1], 'g')

        plt.show(block=False)
        debug()


def get_tab_sign(pattern, n):
    if pattern == 'random':
        return random.choice([-1, 1])
    elif pattern == 'alternating':
        return (-1) ** n
    elif type(pattern) == list:
        return pattern[n]


# TODO; move these into a utility module?
def resample_curve(r, new_length):
    """resample NxD curve to MxD, where
    M is the second input new_length"""
    # TODO: consider better resampling? interp might have something
    t = np.linspace(0, 1, len(r))
    ti = np.linspace(0, 1, int(new_length))  # TODO double check, off by one
    return np.vstack([np.interp(ti, t, c) for c in r.T]).T


def get_monospaced_positions(L, N):
    """center-justified, equally-spaced points
    interval of length L, N points"""
    return np.arange(0, L, L / N) + L / (2 * N)
