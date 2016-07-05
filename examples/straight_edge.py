import matplotlib.pyplot as plt
from pyzzle import create_puzzle_piece_edge


def plot_straight_edge():
    """generate and plot one puzzle piece edge, with a straight baseline"""
    xy, knots = create_puzzle_piece_edge()
    fig = plt.figure()
    fig.add_subplot(111, aspect='equal')
    plt.plot(xy[:, 0], xy[:, 1], 'k-')
    for x, y, mx, my in knots:
        plt.plot(x, y, 'ro')
        plt.plot([x, x + mx], [y, y + my], 'r-')
    plt.axis([0, 1, -0.05, .5])
    plt.show()

plot_straight_edge()
