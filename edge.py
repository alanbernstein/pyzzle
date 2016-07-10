import random
import numpy as np
from geometry.spline import slope_controlled_bezier_curve


# TODO: rewrite this as ParameterDistribution,
# with method sample(), that does the same thing as randomize() here,
# but returns a random sample rather than setting
# not sure if that would work for the interactive plot, have to think about it

# TODO: refactor all this stuff into EdgeCutter class
# this should encapsulate everything...
#
# cutter = EdgeCutter()
# edge = cutter.generate()
#
# different edges would have different mappings from parameters to knots
# might not be any good way to generalize that


# i was able to draw a decent puzzle piece edge using illustrator's pen tool
# - 9 control points, each with controlled slope
# - illustrator uses bezier, so that should work


class Param(object):
    def __init__(self, name, val, min, max, relative=False):
        self.name = name
        self.val = val
        self.min = min
        self.max = max
        self.relative = relative

    def __str__(self):
        return '%s in [%s %s]: %s' % (self.name, self.min, self.max, self.val)


class ParamSet(object):
    """ugly thing that allows a relatively clean interface:
    pset = Paramset([Param(...), Param(...), ...])  # init with value and bounds
    pset.paramname.val                              # get value (known name)
    pset.paramname.min                              # get min bound
    pset.set_param(paramname, val)                    # set val of parameter, programmatically (unknown name)
    this interface allows looping over the set of parameter members,
    which greatly simplifies the interactive plot.
    """
    def __init__(self, paramlist):
        for param in paramlist:
            self.__setattr__(param.name, param)

    def randomize(self):
        for paramname in self.get_param_names():
            param = self.get_param(paramname)
            val = random.uniform(param.min, param.max)
            self.set_param(paramname, val)

    def get_param_names(self):
        """get list of all params contained within"""
        return self.__dict__.keys()

    def get_param(self, paramname):
        """get full param object by name"""
        return self.__getattribute__(paramname)

    def set_param(self, paramname, val):
        """set the `val` attribute of a param object, by name"""
        param = self.__getattribute__(paramname)
        param.val = val
        self.__setattr__(paramname, param)


def get_default_tab_parameters():
    # TODO: make these look better
    pset = ParamSet([Param('pos', 0.5, .4, .6),
                     Param('height', 0.25, .2, 0.3),
                     Param('head_width', 0.25, .2, 0.3),
                     Param('head_height', 0.75, 0.7, .8, relative=True),
                     Param('neck_width', 0.5, .4, .6, relative=True),
                     Param('neck_height', 0.25, .2, 0.3, relative=True),
                     Param('neck_knot_strength', 0.05, .03, 0.07),
                     Param('head_knot_strength', 0.05, .03, 0.07),
                     ])
    return pset


SEGMENTS_PER_PIECE = 8  # this should reflect the knots array here
def create_puzzle_piece_edge(params=None, pts_per_segment=25):
    # in units of edge length
    if params:
        pos = params.pos.val
        height = params.height.val  # height of tallest point

        head_width = params.head_width.val               # width of widest point
        head_height = params.head_height.val * height    # height of widest point
        neck_width = params.neck_width.val * head_width  # width of narrow point
        neck_height = params.neck_height.val * height    # height of narrow point

        neck_knot_strength = params.neck_knot_strength.val
        head_knot_strength = params.head_knot_strength.val
    else:
        # want to continue being able to use this without all the mess of the paramset class
        pos = .5
        height = .25
        head_width = .25
        head_height = .75 * height
        neck_width = .5 * head_width
        neck_height = .25 * height

        neck_knot_strength = 0.05
        head_knot_strength = 0.05

    # define control knots
    knots = np.array([[0, 0, .2, 0],
                      [pos - head_width / 2, 0, .05, 0],
                      [pos - neck_width / 2, neck_height, 0, neck_knot_strength],
                      [pos - head_width / 2, head_height, 0, head_knot_strength],
                      [pos, height, .05, 0],
                      [pos + head_width / 2, head_height, 0, -head_knot_strength],
                      [pos + neck_width / 2, neck_height, 0, -neck_knot_strength],
                      [pos + head_width / 2, 0, .05, 0],
                      [1, 0, .2, 0]
                      ])

    # generate spline curve
    xy = slope_controlled_bezier_curve(knots, pts_per_segment)

    # generate offset vector from baseline
    edge_length = 1
    pts_per_tab = pts_per_segment * SEGMENTS_PER_PIECE
    tt = np.linspace(0, edge_length, pts_per_tab + 1)
    null_base_curve = np.vstack((tt, 0 * tt)).T
    dxy = xy - null_base_curve

    return xy, dxy, knots
