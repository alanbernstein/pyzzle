import random
import matplotlib.pyploy as plt
from pyzzle import (get_default_nub_parameters,
                    create_puzzle_piece_edge)


random_sign = lambda: random.choice([1, -1])


def plot_square_piece_simple():
    """simple demo - makes more sense for a 'cut' to be the next
    level of abstraction after an 'edge' though"""
    pset = get_default_nub_parameters()
    pset.randomize()
    bxy, _ = create_puzzle_piece_edge(pset)
    pset.randomize()
    txy, _ = create_puzzle_piece_edge(pset)
    pset.randomize()
    lxy, _ = create_puzzle_piece_edge(pset)
    pset.randomize()
    rxy, _ = create_puzzle_piece_edge(pset)
    fig = plt.figure()
    fig.add_subplot(111, aspect='equal')
    plt.plot(bxy[:, 0], random_sign() * bxy[:, 1], 'k-')
    plt.plot(txy[:, 0], random_sign() * txy[:, 1] + 1, 'k-')
    plt.plot(random_sign() * lxy[:, 1], lxy[:, 0], 'k-')
    plt.plot(random_sign() * rxy[:, 1] + 1, rxy[:, 0], 'k-')
    plt.show()
