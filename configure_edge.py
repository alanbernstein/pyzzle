import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pyzzle import (get_default_nub_parameters,
                    create_puzzle_piece_edge)


def configure_edge_parameters_interactive():
    """plot a puzzle piece edge, with slider widgets for each of the
    control parameters. useful for tweaking the parameter distributions"""
    # initialize paramset and data
    pset = get_default_nub_parameters()
    pset.randomize()
    xy, knots = create_puzzle_piece_edge(pset)

    # initialize graphics objects
    plt.subplot(111)
    plt.subplots_adjust(bottom=.6)
    l0, = plt.plot(xy[:, 0], xy[:, 1], 'k')
    l1 = []
    l2 = []
    for x, y, mx, my in knots:
        ll, = plt.plot(x, y, 'ro')
        l1.append(ll)
        ll, = plt.plot([x, x + mx], [y, y + my], 'r-')
        l2.append(ll)
    plt.axis([0, 1, 0, .4])

    # define sliders
    axs = {}
    sliders = {}
    for n, pname in enumerate(pset.get_param_names()):
        axs[pname] = plt.axes([.25, .05 + n * .05, .65, .03])
        param = pset.__getattribute__(pname)
        sliders[pname] = Slider(axs[pname], pname.replace('_', ' '),
                                param.min, param.max, valinit=param.val)

    # define callback
    def update(val):
        # read sliders, update paramset
        for k, v in sliders.items():
            pset.set_param(k, v.val)

        # generate new data
        xy, knots = create_puzzle_piece_edge(pset)

        # attach data to graphics objects
        l0.set_data(xy[:, 0], xy[:, 1])
        for n, k in enumerate(knots):
            l1[n].set_data(k[0], k[1])
            l2[n].set_data([k[0], k[0] + k[2]], [k[1], k[1] + k[3]])

        plt.draw()

    # attach callbacks
    for slider in sliders.values():
        slider.on_changed(update)

    plt.show()


configure_edge_parameters_interactive()
