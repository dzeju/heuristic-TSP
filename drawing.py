import matplotlib.pyplot as plt
import networkx as nx
import numpy
from networkx.drawing.nx_agraph import graphviz_layout


def draw_fitness_curve(fitness_curve, folder_name):
    plt.figure(1, figsize=(7, 7))
    plt.plot(fitness_curve)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.title('fitness curve')
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.savefig('results/' + folder_name + '/fitness_curve.png', dpi=500)
    # plt.show()


def draw_path(paths_dict, cities_coord, name, folder_name):
    plt.figure(2)
    plt.tight_layout(pad=0.5)
    paths = list(paths_dict.values())
    names = list(paths_dict.keys())

    length = int(len(paths) / 2) + len(paths) % 2

    fig, ax = plt.subplots(
        length, length, constrained_layout=True, figsize=(13, 13))
    fig.suptitle(name, fontsize=12)
    cities_coord2 = numpy.array(cities_coord)

    for i, item in enumerate(paths):
        xy = numpy.array(item)
        # ax[i % length][int(i / length)].
        ax[i % length][int(i / length)].plot(xy[:, 0], xy[:, 1],
                                             'ro-', markersize=0
                                             )
        ax[i % length][int(i / length)].plot(cities_coord2[:,
                                                           0], cities_coord2[:, 1], 'bo', markersize=3.5
                                             )
        ax[i % length][int(i / length)].plot(xy[0]
                                             [0], xy[0][1], 'go', markersize=4
                                             )
        ax[i % length][int(i / length)].set_title(names[i])
    plt.savefig('results/' + folder_name + '/' + name +
                '.png', dpi=500)
    # plt.show()


def draw_tree(nodes, edges, labels, folder_name):
    plt.figure(3, figsize=(7, 7))
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = graphviz_layout(g, prog="dot")
    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels, font_size=6)
    plt.savefig('results/' + folder_name + '/tree.png', dpi=500)
    # plt.show()
